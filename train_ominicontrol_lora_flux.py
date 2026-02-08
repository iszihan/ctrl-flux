import os
import gc
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf 
import math
from typing import Optional, Union                                                                     
import safetensors.torch
import copy
import prodigyopt
import shutil
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.models.transformers.transformer_flux import FluxIPAdapterAttnProcessor              
from diffusers.optimization import get_scheduler
from diffusers.pipelines import FluxPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from datasets import load_dataset
from ipadapters.ipadapter import setup_ip_adapter
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
import random
import numpy as np
from PIL import Image
import wandb
from arrgh import arrgh
from dataset.ipadapter_dataset import build_subject200k_dataloader, build_laion2B_dataloader, build_laion2B_local_dataloader
from peft import LoraConfig
from ominilora.transformer_flux_omini import omini_transformer_forward
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

def init_loras(adapters, flux_pipeline, lora_config):
    for adapter in adapters:
        flux_pipeline.transformer.add_adapter(LoraConfig(**lora_config), adapter_name=adapter)
    lora_layers = filter(
        lambda p: p.requires_grad, flux_pipeline.transformer.parameters()
    )
    return list(lora_layers)
    
def main():
    
    config_path = os.environ.get('CONFIG_PATH')
    assert config_path is not None, "Please set the CONFIG_PATH environment variable."
    config = OmegaConf.load(config_path)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(config, cli)
    
    # Initialize the accelerator
    config.logging.output_dir = Path(config.logging.output_dir, config.expname)
    if not os.path.exists(config.logging.output_dir):
        os.makedirs(config.logging.output_dir, exist_ok=True)
    logging_dir = Path(config.logging.output_dir, config.logging.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
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
    if config.train.text_encoder_offload:
        flux_pipeline.text_encoder.to('cpu', dtype=weight_dtype).requires_grad_(False)
        flux_pipeline.text_encoder_2.to('cpu', dtype=weight_dtype).requires_grad_(False)
    else:
        flux_pipeline.text_encoder.to(accelerator.device, dtype=weight_dtype).requires_grad_(False).eval()
        flux_pipeline.text_encoder_2.to(accelerator.device, dtype=weight_dtype).requires_grad_(False).eval()
    flux_pipeline.vae.to(accelerator.device, dtype=weight_dtype).requires_grad_(False).eval()
    flux_pipeline.transformer.to(accelerator.device, dtype=weight_dtype).requires_grad_(False).train() # train mode for drop out and norm layers 
    noise_scheduler_copy = copy.deepcopy(flux_pipeline.scheduler)
    
    # Setup the LoRAs (Single Condition only for now)
    adapters = [None,None,config.train.lora.name]
    lora_layers = init_loras([adapters[-1]], flux_pipeline, config.train.lora.config)
    params_to_optimize = lora_layers
    total_number_of_trainable_params = sum(p.numel() for p in params_to_optimize)
    
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
    # Build dataloaders
    train_dataloader = build_subject200k_dataloader(data_valid, flux_pipeline.feature_extractor, config, split='train')
    test_dataloader = build_subject200k_dataloader(test_data_valid, flux_pipeline.feature_extractor, config, split='test')                                                                                                                                
    test_iter = iter(test_dataloader)
    
    tokenizers = [flux_pipeline.tokenizer, flux_pipeline.tokenizer_2]
    text_encoders = [flux_pipeline.text_encoder, flux_pipeline.text_encoder_2]
    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            if config.train.text_encoder_offload:
                # Manual text encoder cpu offloading to save VRAM
                # Move text encoders back to gpu
                text_encoders[0].to(accelerator.device, dtype=weight_dtype)
                text_encoders[1].to(accelerator.device, dtype=weight_dtype)
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, config.train.text_encoder.max_sequence_length
            )
            if config.train.text_encoder_offload:
                # Move text encoders back to cpu
                text_encoders[0].to('cpu', dtype=weight_dtype)
                text_encoders[1].to('cpu', dtype=weight_dtype)
                torch.cuda.empty_cache()
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
    elif config.train.optimizer.type == 'Prodigy':
        optimizer_cls = prodigyopt.Prodigy
    optimizer = optimizer_cls(
        params_to_optimize,
        **config.train.optimizer.params,
    )
    
    # TODO
    # save_model_hook, load_model_hook = create_ip_adapter_hooks(accelerator, flux_pipeline.transformer)
    # accelerator.register_save_state_pre_hook(save_model_hook)
    # accelerator.register_load_state_pre_hook(load_model_hook)
    
    
    transformer, optimizer, train_dataloader = accelerator.prepare(
        flux_pipeline.transformer, optimizer, train_dataloader
    )
    # reassign the wrapped transformer to the flux pipeline
    flux_pipeline.transformer = transformer

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.train.gradient_accumulation_steps)
    if 'max_train_steps' not in config.train:
        config.train.max_train_steps = config.train.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.train.num_train_epochs = math.ceil(config.train.max_train_steps / num_update_steps_per_epoch)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "ominicontrol-flux-subject200k"
        accelerator.init_trackers(tracker_name, config=vars(config))
    
    # Train!
    total_batch_size = config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Total number of trainable parameters = {total_number_of_trainable_params/1e6:.2f}M")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {config.train.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.train.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    initial_global_step = 0
    
    progress_bar = tqdm(
        range(initial_global_step, config.train.max_train_steps),
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
    
    # Pre-load validation images once (avoids reloading each validation step)
    validation_data = None
    if hasattr(config, 'validations') and config.validations is not None:
        validation_images_pil = []
        validation_clip_images = []
        
        # Transform: resize shortest edge to target, then center crop
        val_transform = T.Compose([
            T.Resize(config.dataset.image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(config.dataset.image_size),
        ])
        
        for img_path in config.validations.images:
            pil_img = Image.open(img_path).convert("RGB")
            pil_img = val_transform(pil_img)  # Resize + center crop (no distortion)
            validation_images_pil.append(pil_img)
        
        validation_data = {
            "pil_images": validation_images_pil,
            "prompts": config.validations.prompts,
        }
        logger.info(f"Pre-loaded {len(validation_images_pil)} validation images")
    
    # Training Loop
    for epoch in range(first_epoch, config.train.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            if batch is None:
                continue
            with accelerator.accumulate(transformer):
                with torch.no_grad():
                    # Convert input images to latent space
                    if config.cache_latents:
                        # TODO: Implement cache latents
                        raise ValueError("Cache latents is not supported yet.")
                    else:
                        images = batch["images"].to(device=accelerator.device, dtype=flux_pipeline.vae.dtype)
                        images = flux_pipeline.image_processor.preprocess(images)
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
                    packed_model_input = FluxPipeline._pack_latents(
                        model_input,
                        batch_size=model_input.shape[0],
                        num_channels_latents=model_input.shape[1],
                        height=model_input.shape[2],
                        width=model_input.shape[3],
                    )
                
                    # Sample noise and add to input
                    bsz = model_input.shape[0]
                    t = torch.sigmoid(torch.randn((bsz,), device=accelerator.device))
                    t_ = t.unsqueeze(1).unsqueeze(1)
                    noise = torch.randn_like(packed_model_input).to(accelerator.device)
                    packed_noisy_model_input = ((1 - t_) * packed_model_input + t_ * noise).to(accelerator.device, dtype=weight_dtype)
                
                    # Compute the text embeddings from prompts
                    prompts = batch["prompts"]
                    prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                                prompts, text_encoders, tokenizers
                            )
                    
                    # Prepare conditions
                    position_delta = batch["position_delta"]
                    position_scale = batch.get("position_scale", [1.0])[0]
                    condition_images = batch["condition_images"].to(device=accelerator.device, dtype=flux_pipeline.vae.dtype)
                    condition_model_input = flux_pipeline.image_processor.preprocess(condition_images)
                    condition_model_input = flux_pipeline.vae.encode(condition_images).latent_dist.sample()
                    condition_model_input = (condition_model_input - vae_config_shift_factor) * vae_config_scaling_factor
                    condition_model_input = condition_model_input.to(dtype=weight_dtype)
                
                    packed_condition_model_input = FluxPipeline._pack_latents(
                        condition_model_input,
                        batch_size=condition_model_input.shape[0],
                        num_channels_latents=condition_model_input.shape[1],
                        height=condition_model_input.shape[2],
                        width=condition_model_input.shape[3],
                    )
                    condition_latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                        condition_model_input.shape[0],
                        condition_model_input.shape[2] // 2,
                        condition_model_input.shape[3] // 2,
                        accelerator.device,
                        weight_dtype,
                    )
                    
                    # Position encoding scaling and shifting 
                    if position_scale != 1.0:
                        scale_bias = (position_scale - 1.0) / 2
                        packed_condition_model_input[:, 1:] *= position_scale
                        packed_condition_model_input[:, 1:] += scale_bias
                    condition_latent_image_ids[:, 1] += position_delta[0][0][0].to(dtype=condition_latent_image_ids.dtype)
                    condition_latent_image_ids[:, 2] += position_delta[0][0][1].to(dtype=condition_latent_image_ids.dtype)
                    if len(position_delta) > 1:
                        print("Warning: only the first position delta is used.")
                    condition_latents = [packed_condition_model_input] # Only one condition type
                    condition_ids = [condition_latent_image_ids] # Only one condition type
                    
                    # Prepare guidance
                    guidance = (
                        torch.ones_like(t).to(accelerator.device)
                        if guidance_embeds
                        else None
                    )
                    
                # Prepare group mask for attention mechanism
                branch_n = 2 + len(condition_latents)
                group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool).to(accelerator.device)
                # Disable the attention cross different condition branches
                group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(condition_latents)))
                # Disable the attention from condition branches to image branch and text branch
                if config.train.independent_condition:
                    group_mask[2:, :2] = False
                    
                # Predict the noise residual
                model_pred = omini_transformer_forward(
                    transformer=transformer,
                    hidden_states=[packed_noisy_model_input, *condition_latents],
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                    timesteps=[t, t] + [torch.zeros_like(t)] * len(condition_latents),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    joint_attention_kwargs={},
                    txt_ids=text_ids,
                    img_ids=[latent_image_ids, *condition_ids],
                    return_dict=False,
                    adapters=adapters,
                    group_mask=group_mask,
                )[0]
            
                # Compute loss
                target = packed_noisy_model_input - packed_model_input
                loss = torch.mean(
                    ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, config.train.optimizer.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % config.logging.checkpointing_steps == 0:
                        if config.logging.checkpoints_total_limit > 0:
                            checkpoints = os.listdir(config.logging.output_dir)
                            checkpoints = [d for d in checkpoints if os.path.isdir(d) and d.starts("ip_adapter-")]
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
                        accelerator.save_state(f"{config.logging.output_dir}/ip_adapter-{global_step:06d}", safe_serialization=True)
                        logger.info(f"Saved state to {config.logging.output_dir}/ip_adapter-{global_step:06d}.safetensors")
                        
            logs = {"loss": loss.detach().item(), 
                    "lr": optimizer.param_groups[0]["lr"]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
                
            if global_step >= config.train.max_train_steps:
                break 
        
                
            
if __name__ == "__main__":
    main()