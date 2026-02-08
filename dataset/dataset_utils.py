import os
import json
import glob
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
import random
from PIL import Image
from torchvision import transforms
from braceexpand import braceexpand
import webdataset as wds
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image

def _pil_to_rgb(pil_img: Image.Image) -> Image.Image:
    # Some LAION images are L/LA/P etc.
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return pil_img

def _tensor_to_pil(target_images):
    """Convert normalized tensor batch back to list of PIL images."""
    # Denormalize from [-1, 1] to [0, 1]
    images = (target_images + 1) / 2
    images = images.clamp(0, 1)
    
    # Convert each image in batch to PIL
    pil_images = [to_pil_image(img) for img in images]
    return pil_images

def build_laion2B_dataloader(shard_pattern, feature_extractor, config, split='train', rank=0, world_size=1):
    
    def collate_fn(batch):
        # After wds.batched(), batch is a tuple: (stacked_images_tensor, [captions_list])
        # batch[0] = tensor of shape (batch_size, C, H, W)
        # batch[1] = list of caption strings
        target_images, captions = batch[0], batch[1]
        
        # Skip if empty batch (shouldn't happen but just in case)
        if target_images is None or len(target_images) == 0:
            return None
        # Denormalize from [-1, 1] to [0, 1] for CLIP processor
        images_for_clip = (target_images + 1) / 2
        images_for_clip = images_for_clip.clamp(0, 1)
        condition_img_clip = feature_extractor(images=images_for_clip, return_tensors="pt", do_rescale=False).pixel_values
        result = {
            "images": target_images,
            "clip_images": condition_img_clip,
            "prompts": captions,
        }
        # Only include PIL images for test split (accelerator can't gather non-tensors)
        if split != 'train':
            result["pil_target_images"] = _tensor_to_pil(target_images)
            result["pil_condition_images"] = _tensor_to_pil(target_images)  # same as target for laion2B
        return result
    
    transform = transforms.Compose([
        transforms.Lambda(_pil_to_rgb),
        transforms.Resize(config.dataset.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.dataset.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    if "{" in shard_pattern and "}" in shard_pattern:
        shards = list(braceexpand(shard_pattern))
    else:
        shards = shard_pattern
        
    # For multi-GPU: split shards across processes so each GPU gets different data
    if world_size > 1 and isinstance(shards, list):
        # Each rank gets a subset of shards: rank 0 gets [0, world_size, 2*world_size, ...], etc.
        shards = shards[rank::world_size]
        
    if config.dataset.resampled:
        dataset = wds.ResampledShards(shards)
    else:
        # nodesplitter=lambda src: src tells webdataset that node splitting is already handled
        # (we split shards manually above via shards[rank::world_size])
        dataset = wds.WebDataset(shards, shardshuffle=True, nodesplitter=lambda src: src)
        
    dataset = (
        dataset
        # Removed pre-decode shuffle - shardshuffle=True already handles shard-level randomization
        # Having a large buffer here was contributing to memory usage
        .decode("pil", handler=wds.handlers.warn_and_continue)
        .to_tuple("jpg;png;jpeg;webp", "txt", handler=wds.handlers.warn_and_continue)
        .map_tuple(transform, lambda s: s.strip())
    )
    
    loader = torch.utils.data.DataLoader(
        dataset.batched(config.train.batch_size if split == 'train' else 1, partial=not config.dataset.drop_last),
        batch_size=None,  # IMPORTANT: dataset already batched
        num_workers=config.dataset.train_dataloader_num_workers if split == 'train' else config.dataset.test_dataloader_num_workers,
        prefetch_factor=config.dataset.train_prefetch_factor if split == 'train' else config.dataset.test_prefetch_factor,
        pin_memory=True,
        persistent_workers=False,  # Disabled to prevent OOM
        collate_fn=collate_fn,
    )
    return loader

def build_subject200k_dataloader(data, feature_extractor, config, split='train'):
    
    def collate_fn(data):
        
        images = torch.stack([example["image"] for example in data])
        condition_images = torch.stack([example["condition_image"] for example in data])
        position_delta = torch.cat([example["position_delta"] for example in data])
        prompts = [example["prompt"] for example in data]
        results = {
            "images": images,
            "condition_images": condition_images,
            "prompts": prompts,
            "position_delta": position_delta,
        }
        if "condition_clip" in data[0]:
            clip_images = torch.cat([example["condition_clip"] for example in data], dim=0)
            results["clip_images"] = clip_images
            
        if 'pil_target_image' in data[0]:
            target_images = [example["pil_target_image"] for example in data]
            condition_images = [example["pil_condition_image"] for example in data]
            results["pil_target_images"] = target_images
            results["pil_condition_images"] = condition_images
            return results
        return results
    
    dataset = Subject200KDataset(
        data,
        condition_size=tuple(config.dataset.condition_size),
        target_size=tuple(config.dataset.target_size),
        image_size=config.dataset.image_size,
        padding=config.dataset.padding,
        condition_type=config.dataset.type,
        drop_text_prob=config.dataset.drop_text_prob if split == 'train' else 0.0,
        drop_image_prob=config.dataset.drop_image_prob if split == 'train' else 0.0,
        clip_image_processor=feature_extractor,
        return_pil_image=(split != 'train'),
    )
        
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False, #for debugging only #True if split == 'train' else False,
        collate_fn=collate_fn,
        batch_size=config.train.batch_size if split == 'train' else 1,
        num_workers=config.dataset.train_dataloader_num_workers if split == 'train' else config.dataset.test_dataloader_num_workers,
        
    ) #[N, C, H, W]
    
    return dataloader

class Subject200KDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size=(512, 512),
        target_size=(512, 512),
        image_size: int = 512,
        padding: int = 0,
        condition_type: str = "subject",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        clip_image_processor: CLIPImageProcessor = None,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.image_size = image_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()
        self.clip_image_processor = clip_image_processor

    def __len__(self):
        return len(self.base_dataset) * 2

    def __getitem__(self, idx):
        # If target is 0, left image is target, right image is condition
        target = idx % 2
        item = self.base_dataset[idx // 2]

        # Crop the image to target and condition
        image = item["image"]
        left_img = image.crop(
            (
                self.padding,
                self.padding,
                self.image_size + self.padding,
                self.image_size + self.padding,
            )
        )
        right_img = image.crop(
            (
                self.image_size + self.padding * 2,
                self.padding,
                self.image_size * 2 + self.padding * 2,
                self.image_size + self.padding,
            )
        )

        # Get the target and condition image
        target_image, condition_img = (
            (left_img, right_img) if target == 0 else (right_img, left_img)
        )
        if self.clip_image_processor is not None:
            condition_img_clip = self.clip_image_processor(images=condition_img, return_tensors="pt").pixel_values

        # Resize the image
        target_image = target_image.resize(self.target_size).convert("RGB")
        condition_img = condition_img.resize(self.condition_size).convert("RGB")

        # Get the description
        description = item["description"][
            "description_0" if target == 0 else "description_1"
        ]

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new("RGB", self.condition_size, (0, 0, 0))
            if self.clip_image_processor is not None:
                condition_img_clip = self.clip_image_processor(images=condition_img, return_tensors="pt").pixel_values
        position_delta = np.array([[0, -self.condition_size[0] // 16]])
        outputs = {
            "image": self.to_tensor(target_image),
            "condition_image": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "prompt": description,
            "position_delta": self.to_tensor(position_delta),
            **({"pil_target_image": target_image,
                'pil_condition_image': condition_img} if self.return_pil_image else {}),
        }
        if self.clip_image_processor is not None:
            outputs["condition_clip"] = condition_img_clip
        return outputs
