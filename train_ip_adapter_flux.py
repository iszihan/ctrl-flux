import os
import torch 
from omegaconf import OmegaConf 
import math
import copy
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from diffusers.optimization import get_scheduler
from diffusers.pipelines import FluxPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from datasets import load_dataset
from ipadapters.ipadapter import setup_ip_adapter
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import numpy as np
from PIL import Image
import wandb

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("wandb").setLevel(logging.WARNING)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger = get_logger(__name__)

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

def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids

def log_test_sample(pipeline, 
                    test_sample, 
                    accelerator, 
                    weight_dtype,
                    config,
                    is_final_validation=False):
    
    logger.info("Running test sample...")
    pipeline.set_progress_bar_config(disable=True)
    
    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(config.test.seed) if config.test.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type, weight_dtype) if not is_final_validation else nullcontext()
    
    with torch.no_grad():
        with autocast_ctx:
            image = pipeline(
                width=config.dataset.image_size,
                height=config.dataset.image_size,
                prompt=test_sample["prompts"],
                negative_prompt="",
                true_cfg_scale=config.train.guidance_scale,
                generator=generator,
                ip_adapter_image=test_sample["pil_condition_images"],
                ).images[0]
            image_no_ip = pipeline(
                width=config.dataset.image_size,
                height=config.dataset.image_size,
                prompt=test_sample["prompts"],
                negative_prompt="",
                true_cfg_scale=config.train.guidance_scale,
                generator=generator,
                ).images[0]
            
        
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{test_sample['prompts']}") 
                    ],
                    'test_no_ip': [wandb.Image(image_no_ip, caption=f"{test_sample['prompts']}")],
                    'condition_image': [wandb.Image(test_sample["pil_condition_images"][0], caption="Conditioning")],
                    'target_image': [wandb.Image(test_sample["pil_target_images"][0], caption="Target")],
                }
            )
    return image

def main():
    
    config_path = os.environ.get('CONFIG_PATH')
    assert config_path is not None, "Please set the CONFIG_PATH environment variable."
    config = OmegaConf.load(config_path)
    
    # Initialize the accelerator
    logging_dir = Path(config.logging.output_dir, config.logging.logging_dir)
    accelerator = Accelerator(
        mixed_precision=config.dtype,
        log_with=config.logging.logger,
        project_config=ProjectConfiguration(project_dir=config.logging.output_dir, logging_dir=logging_dir),
    )
    
    if accelerator.is_main_process:
        if config.logging.output_dir is not None:
            os.makedirs(config.logging.output_dir, exist_ok=True)
            
    # Load the Flux model
    weight_dtype = DTYPE_MAP[config.dtype]
    flux_pipeline = FluxPipeline.from_pretrained(config.flux_path, torch_dtype=weight_dtype).to('cuda')
    # Freeze the Flux pipeline
    flux_pipeline.text_encoder.to(accelerator.device, dtype=weight_dtype).requires_grad_(False)
    flux_pipeline.text_encoder_2.to(accelerator.device, dtype=weight_dtype).requires_grad_(False)
    flux_pipeline.vae.to(accelerator.device, dtype=weight_dtype).requires_grad_(False)
    flux_pipeline.transformer.to(accelerator.device, dtype=weight_dtype).requires_grad_(False)
    noise_scheduler_copy = copy.deepcopy(flux_pipeline.scheduler)
    
    # Setup the IP Adapter modules
    ip_adapter_trainable_params = setup_ip_adapter(flux_pipeline, image_encoder_pretrained_model_name_or_path=config.image_encoder_path)
    total_number_of_trainable_params = sum(p.numel() for p in ip_adapter_trainable_params)
    params_to_optimize = [{'params': ip_adapter_trainable_params, 'lr': config.train.optimizer.lr}]
    
    if config.gradient_checkpointing:
        flux_pipeline.transformer.enable_gradient_checkpointing()
    
    # Dataloader 
    raw_dataset = load_dataset("Yuanshi/Subjects200K")
    # Filter function to filter out low-quality images from Subjects200K
    def filter_func(item):
        if not item.get("quality_assessment"):
            return False
        return all(
            item["quality_assessment"].get(key, 0) >= 5
            for key in ["compositeStructure", "objectConsistency", "imageQuality"]
        )
    # Filter dataset
    if not os.path.exists("./cache/dataset"):
        os.makedirs("./cache/dataset")
    data_valid = raw_dataset["train"].filter(
        filter_func,
        num_proc=4,
        cache_file_name="./cache/dataset/data_valid.arrow",
    )
    # Split into train/test                                                                             
    split = data_valid.train_test_split(test_size=0.01, seed=42)                                        
    test_data_valid = split["test"]                                                                           
                                                                                                      
    train_dataset = Subject200KDataset(
        data_valid,
        condition_size=tuple(config.dataset.condition_size),
        target_size=tuple(config.dataset.target_size),
        image_size=config.dataset.image_size,
        padding=config.dataset.padding,
        condition_type=config.dataset.type,
        drop_text_prob=config.dataset.drop_text_prob,
        drop_image_prob=config.dataset.drop_image_prob,
        clip_image_processor=flux_pipeline.feature_extractor,
    )
    test_dataset = Subject200KDataset(
        test_data_valid,
        condition_size=tuple(config.dataset.condition_size),
        target_size=tuple(config.dataset.target_size),
        image_size=config.dataset.image_size,
        padding=config.dataset.padding,
        condition_type=config.dataset.type,
        drop_text_prob=0.0,
        drop_image_prob=0.0,
        clip_image_processor=flux_pipeline.feature_extractor,
        return_pil_image=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=config.train.batch_size,
        num_workers=config.train.dataloader_num_workers,
    ) #[N, C, H, W]
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=config.test.batch_size,
        num_workers=config.test.dataloader_num_workers,
    ) #[N, C, H, W]
    test_iter = iter(test_dataloader)
    
    tokenizers = [flux_pipeline.tokenizer, flux_pipeline.tokenizer_2]
    text_encoders = [flux_pipeline.text_encoder, flux_pipeline.text_encoder_2]
    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, config.train.text_encoder.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            text_ids = text_ids.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds, text_ids
    vae_config_shift_factor = flux_pipeline.vae.config.shift_factor
    vae_config_scaling_factor = flux_pipeline.vae.config.scaling_factor
    vae_config_block_out_channels = flux_pipeline.vae.config.block_out_channels
    guidance_embeds = flux_pipeline.transformer.config.guidance_embeds
    
    # Initialize optimizer 
    if config.train.optimizer.type == 'adamw':
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        params_to_optimize,
        betas=(config.train.optimizer.beta1, config.train.optimizer.beta2),
        weight_decay=config.train.optimizer.adam_weight_decay,
        eps=config.train.optimizer.adam_epsilon,
    )
    
    # Scheduler 
    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = config.train.scheduler.warmup_steps * accelerator.num_processes
    if 'max_train_steps' not in config.train:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / config.train.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            config.train.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = config.train.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        config.train.scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=config.train.scheduler.num_cycles,
        power=config.train.scheduler.power,
    )
    
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_pipeline.transformer, optimizer, train_dataloader, lr_scheduler
        )
    # reassign the wrapped transformer to the flux pipeline
    flux_pipeline.transformer = transformer
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.train.gradient_accumulation_steps)
    if 'max_train_steps' not in config.train:
        config.train.max_train_steps = config.train.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != config.train.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    config.train.num_train_epochs = math.ceil(config.train.max_train_steps / num_update_steps_per_epoch)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "ip-adapter-flux-subject200k"
        accelerator.init_trackers(tracker_name, config=vars(config))
    
    # Train!
    total_batch_size = config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Total number of trainable parameters = {total_number_of_trainable_params/1e6:.2f}M")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {config.train.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.train.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    if config.resume_from_checkpoint:
        # TODO: Implement resume from checkpoint
        raise ValueError("Resume from checkpoint is not supported yet.")
    else:
        initial_global_step = 0
        
    progress_bar = tqdm(
        range(0, config.train.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
     
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    # Training Loop
    for epoch in range(config.train.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Compute the text embeddings from prompts
                prompts = batch["prompts"]
                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                            prompts, text_encoders, tokenizers
                        )
                
                # Compute ID-Adapter image embeddings 
                condition_clip = batch["clip_images"].to(dtype=weight_dtype)
                image_embeds = flux_pipeline.encode_image(condition_clip, accelerator.device, 1) # [B, 768]
                joint_attention_kwargs = {"ip_adapter_image_embeds": image_embeds}
                
                # Convert images to latent space
                if config.cache_latents:
                    # TODO: Implement cache latents
                    raise ValueError("Cache latents is not supported yet.")
                else:
                    images = batch["images"].to(dtype=flux_pipeline.vae.dtype)
                    model_input = flux_pipeline.vae.encode(images).latent_dist.sample()
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)
                
                # Downsampling factor
                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                # Sample noise 
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=config.train.noise_scheduler.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=config.train.noise_scheduler.logit_mean,
                    logit_std=config.train.noise_scheduler.logit_std,
                    mode_scale=config.train.noise_scheduler.mode_scale,
                ) 
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                
                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                
                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                # Use guidance scale if guidance is enabled in the model
                if guidance_embeds:
                    guidance = torch.tensor([config.train.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None
                    
                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    joint_attention_kwargs=joint_attention_kwargs,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=config.train.noise_scheduler.weighting_scheme, sigmas=sigmas)
                
                # flow matching loss
                target = noise - model_input
                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(ip_adapter_trainable_params, config.train.optimizer.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % config.logging.checkpointing_steps == 0:
                        if config.logging.checkpoints_total_limit > 0:
                            checkpoints = os.listdir(config.logging.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.logging.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config.logging.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        save_path = os.path.join(config.logging.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            
            logs = {"loss": loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= config.train.max_train_steps:
                break 
            if step % config.logging.test_steps == 0:
                transformer.eval()
                test_sample = next(test_iter)
                log_test_sample(pipeline=flux_pipeline, 
                                test_sample=test_sample, 
                                accelerator=accelerator, 
                                weight_dtype=weight_dtype, 
                                config=config, 
                                is_final_validation=False)
                transformer.train()
                
                
        
        # After one epoch, generate test examples for validation  
        if accelerator.is_main_process:
            if epoch % config.logging.test_steps == 0:
                transformer.eval()
                test_smaple = next(iter(test_dataloader))
                log_test_sample(pipeline=flux_pipeline, 
                                test_sample=test_sample, 
                                accelerator=accelerator, 
                                weight_dtype=weight_dtype, 
                                config=config, 
                                is_final_validation=False)
                
                
                
                  
                
if __name__ == "__main__":
    main()