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
from .util import _pil_to_rgb, _tensor_to_pil

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
        pin_memory=True,
        persistent_workers=False,  # Disabled to prevent memory leak with WebDataset
        collate_fn=collate_fn,
    )
    return loader

def build_laion2B_local_dataloader(data_dir, feature_extractor, config, split='train'):
    """
    Build dataloader for LAION2B data from extracted folder.
    Simpler alternative to WebDataset - no memory leak issues.
    """
    def collate_fn(data):
        images = torch.stack([example["image"] for example in data])
        clip_images = torch.cat([example["condition_clip"] for example in data], dim=0)
        prompts = [example["prompt"] for example in data]
        
        result = {
            "images": images,
            "clip_images": clip_images,
            "prompts": prompts,
        }
        
        if 'pil_target_image' in data[0]:
            result["pil_target_images"] = [example["pil_target_image"] for example in data]
            result["pil_condition_images"] = [example["pil_condition_image"] for example in data]
        
        return result
    
    dataset = Laion2BFolderDataset(
        data_dir=data_dir,
        target_size=tuple(config.dataset.target_size),
        drop_text_prob=config.dataset.drop_text_prob if split == 'train' else 0.0,
        drop_image_prob=config.dataset.drop_image_prob if split == 'train' else 0.0,
        clip_image_processor=feature_extractor,
        return_pil_image=(split != 'train'),
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=(split == 'train'),
        collate_fn=collate_fn,
        batch_size=config.train.batch_size if split == 'train' else 1,
        num_workers=config.dataset.train_dataloader_num_workers if split == 'train' else config.dataset.test_dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=(split == 'train'),
    )
    
    return dataloader

def build_subject200k_dataloader(data, feature_extractor, config, split='train'):
    
    def collate_fn(data):
        images = torch.stack([example["image"] for example in data])
        clip_images = torch.cat([example["condition_clip"] for example in data], dim=0)
        prompts = [example["prompt"] for example in data]
        if 'pil_target_image' in data[0]:
            target_images = [example["pil_target_image"] for example in data]
            condition_images = [example["pil_condition_image"] for example in data]
            return {
                "images": images,
                "clip_images": clip_images,
                "prompts": prompts,
                "pil_target_images": target_images,
                "pil_condition_images": condition_images,
            }
        return {
            "images": images,
            "clip_images": clip_images,
            "prompts": prompts,
        }
    
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
    )
        
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True if split == 'train' else False,
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
        condition_img_clip = self.clip_image_processor(images=condition_img, return_tensors="pt").pixel_values

        # Resize the image
        target_image = target_image.resize(self.target_size).convert("RGB")

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
            condition_img_clip = self.clip_image_processor(images=condition_img, return_tensors="pt").pixel_values
        
        return {
            "image": self.to_tensor(target_image),
            "condition_clip": condition_img_clip,
            "condition_type": self.condition_type,
            "prompt": description,
            **({"pil_target_image": target_image,
                'pil_condition_image': condition_img} if self.return_pil_image else {}),
        }

class Laion2BLocalDataset(Dataset):
    """
    Simple map-style dataset for LAION data extracted from tar files.
    
    Expected folder structure:
        data_dir/
            000000000.jpg
            000000000.txt  (or .json with "caption" key)
            000000001.jpg
            000000001.txt
            ...
    """
    def __init__(
        self,
        data_dir: str,
        target_size=(512, 512),
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        clip_image_processor: CLIPImageProcessor = None,
        return_pil_image: bool = False,
    ):
        self.data_dir = data_dir
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.clip_image_processor = clip_image_processor
        self.return_pil_image = return_pil_image
        
        # Find all images
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.image_paths += sorted(glob.glob(os.path.join(data_dir, "*.png")))
        self.image_paths += sorted(glob.glob(os.path.join(data_dir, "*.webp")))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"Found {len(self.image_paths)} images in {data_dir}")
        
        self.to_tensor = T.ToTensor()
    
    def __len__(self):
        return len(self.image_paths)
    
    def _load_caption(self, img_path: str) -> str:
        """Load caption from .txt or .json file."""
        base = os.path.splitext(img_path)[0]
        
        # Try .txt first
        txt_path = base + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                return f.read().strip()
        
        # Try .json
        json_path = base + ".json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                return data.get("caption", data.get("text", ""))
        
        return ""  # No caption found
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path)
            image = _pil_to_rgb(image)
            image = image.resize(self.target_size)
        except Exception as e:
            # Return a black image on error
            print(f"Error loading {img_path}: {e}")
            image = Image.new("RGB", self.target_size, (0, 0, 0))
        
        caption = self._load_caption(img_path)
        
        # Random drops
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        
        if drop_text:
            caption = ""
        
        condition_img = image
        if drop_image:
            condition_img = Image.new("RGB", self.target_size, (0, 0, 0))
        
        # Process for CLIP
        condition_img_clip = self.clip_image_processor(
            images=condition_img, return_tensors="pt"
        ).pixel_values
        
        # Normalize to [-1, 1] for diffusion
        target_tensor = self.to_tensor(image) * 2.0 - 1.0
        
        result = {
            "image": target_tensor,
            "condition_clip": condition_img_clip,
            "prompt": caption,
        }
        
        if self.return_pil_image:
            result["pil_target_image"] = image
            result["pil_condition_image"] = condition_img
        
        return result

