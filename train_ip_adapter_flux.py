import os
import gc
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf 
import math
from typing import Optional, Union                                                                     
import safetensors.torch
import copy
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
import random
import numpy as np
from PIL import Image
import wandb
from dataset.ipadapter_dataset import build_subject200k_dataloader, build_laion2B_dataloader, build_laion2B_local_dataloader

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

def save_ip_adapter(                                                                                   
      pipeline,                
      accelerator,                                                                                
      save_directory: Union[str, os.PathLike],                                                           
      weight_name: str = "ip_adapter.safetensors",                                                       
      safe_serialization: bool = True,                                                                   
      adapter_index: int = 0,                                                                            
  ):                                                                                                     
      """                                                                                                
      Save IP-Adapter weights in the format expected by load_ip_adapter() for Flux.                      
                                                                                                         
      Args:                                                                                              
          pipeline: FluxPipeline with IP-Adapter loaded                                                  
          save_directory: Directory to save the weights                                                  
          weight_name: Filename for the saved weights                                                    
          safe_serialization: Whether to use safetensors (True) or torch.save (False)                    
          adapter_index: Which IP adapter to save if multiple are loaded (default: 0)                    
      """                                                                                                
                                                                                                         
      transformer = pipeline.transformer                                                                 
                                                                                                         
      # Unwrap accelerate-wrapped model                                                                  
      transformer = accelerator.unwrap_model(transformer)                                                            
                                                                                                         
      # Check if IP adapter is loaded                                                                    
      if transformer.encoder_hid_proj is None:                                                           
          raise ValueError("No IP-Adapter loaded in the pipeline.")                                      
                                                                                                         
      state_dict = {                                                                                     
          "image_proj": {},                                                                              
          "ip_adapter": {},                                                                              
      }                                                                                                  
                                                                                                         
      # ========== Extract image_proj weights ==========                                                 
      # encoder_hid_proj is MultiIPAdapterImageProjection with image_projection_layers                   
      image_proj_layer = transformer.encoder_hid_proj.image_projection_layers[adapter_index]             
                                                                                                         
      # ImageProjection has: image_embeds (nn.Linear), norm (nn.LayerNorm)                               
      # Need to convert image_embeds -> proj for the expected format                                     
      image_proj_state = image_proj_layer.state_dict()                                                   
                                                                                                         
      for key, value in image_proj_state.items():                                                        
          # Convert "image_embeds.weight" -> "proj.weight", etc.                                         
          new_key = key.replace("image_embeds", "proj")                                                  
          state_dict["image_proj"][new_key] = value                                                      
                                                                                                         
      # ========== Extract ip_adapter weights ==========                                                 
      # Iterate through attention processors and extract IP adapter weights                              
      key_id = 0                                                                                         
      for name, attn_processor in transformer.attn_processors.items():                                   
          # Skip single_transformer_blocks - they don't have IP adapter                                  
          if name.startswith("single_transformer_blocks"):                                               
              continue                                                                                   
                                                                                                         
          if not isinstance(attn_processor, FluxIPAdapterAttnProcessor):                                 
              raise ValueError(                                                                          
                  f"Expected FluxIPAdapterAttnProcessor for {name}, "                                    
                  f"got {type(attn_processor).__name__}"                                                 
              )                                                                                          
                                                                                                         
          # Extract to_k_ip and to_v_ip for the specified adapter_index                                  
          # to_k_ip and to_v_ip are nn.ModuleList                                                        
          to_k_ip = attn_processor.to_k_ip[adapter_index]                                                
          to_v_ip = attn_processor.to_v_ip[adapter_index]                                                
                                                                                                         
          state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"] = to_k_ip.weight.data.clone()             
          state_dict["ip_adapter"][f"{key_id}.to_k_ip.bias"] = to_k_ip.bias.data.clone()                 
          state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"] = to_v_ip.weight.data.clone()             
          state_dict["ip_adapter"][f"{key_id}.to_v_ip.bias"] = to_v_ip.bias.data.clone()                 
                                                                                                         
          key_id += 1                                                                                    
                                                                                                         
      # ========== Save to disk ==========                                                               
      os.makedirs(save_directory, exist_ok=True)                                                         
      save_path = os.path.join(save_directory, weight_name)                                              
                                                                                                         
      if safe_serialization:                                                                             
          # Flatten the nested dict for safetensors                                                      
          flat_state_dict = {}                                                                           
          for prefix, sub_dict in state_dict.items():                                                    
              for key, value in sub_dict.items():                                                        
                  flat_state_dict[f"{prefix}.{key}"] = value                                             
                                                                                                         
          safetensors.torch.save_file(flat_state_dict, save_path)                                        
      else:                                                                                              
          torch.save(state_dict, save_path)                                                              
                                                                                                         
      print(f"IP-Adapter saved to {save_path}")                                                          
      return save_path 
  
def create_ip_adapter_hooks(accelerator, transformer):                                                 
      """                                                                                                
      Create save/load hooks for IP adapter training with accelerate.                                    
                                                                                                         
      Args:                                                                                              
          accelerator: The accelerate Accelerator instance                                               
          transformer: The FluxTransformer2DModel (can be accelerate-wrapped)                            
                                                                                                         
      Returns:                                                                                           
          Tuple of (save_model_hook, load_model_hook)                                                    
      """                                                                                                
                                                                                                         
      def save_model_hook(models, weights, output_dir):                                                  
          if accelerator.is_main_process:                                                                
              for model in models:                                                                       
                  unwrapped = accelerator.unwrap_model(model)                                                        
                                                                                                         
                  # Check if this is the transformer with IP adapter                                     
                  if isinstance(unwrapped, type(accelerator.unwrap_model(transformer))):                             
                      state_dict = {                                                                     
                          "image_proj": {},                                                              
                          "ip_adapter": {},                                                              
                      }                                                                                  
                                                                                                         
                      # ========== Extract image_proj weights ==========                                 
                      if unwrapped.encoder_hid_proj is None:                                             
                          raise ValueError("No IP-Adapter loaded in the transformer.")                   
                                                                                                         
                      # Support single IP adapter (index 0)                                              
                      image_proj_layer = unwrapped.encoder_hid_proj.image_projection_layers[0]           
                      image_proj_state = image_proj_layer.state_dict()                                   
                                                                                                         
                      for key, value in image_proj_state.items():                                        
                          # Convert "image_embeds.weight" -> "proj.weight"                               
                          new_key = key.replace("image_embeds", "proj")                                  
                          state_dict["image_proj"][new_key] = value.cpu()                                
                                                                                                         
                      # ========== Extract ip_adapter weights ==========                                 
                      key_id = 0                                                                         
                      for name, attn_processor in unwrapped.attn_processors.items():                     
                          # Skip single_transformer_blocks - they don't have IP adapter                  
                          if name.startswith("single_transformer_blocks"):                               
                              continue                                                                   
                                                                                                         
                          if not isinstance(attn_processor, FluxIPAdapterAttnProcessor):                 
                              raise ValueError(                                                          
                                  f"Expected FluxIPAdapterAttnProcessor for {name}, "                    
                                  f"got {type(attn_processor).__name__}"                                 
                              )                                                                          
                                                                                                         
                          # Extract to_k_ip and to_v_ip for adapter index 0                              
                          to_k_ip = attn_processor.to_k_ip[0]                                            
                          to_v_ip = attn_processor.to_v_ip[0]                                            
                                                                                                         
                          state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"] = to_k_ip.weight.data.cpu().clone()                                                                      
                          state_dict["ip_adapter"][f"{key_id}.to_k_ip.bias"] = to_k_ip.bias.data.cpu().clone()                                                                      
                          state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"] = to_v_ip.weight.data.cpu().clone()                                                                      
                          state_dict["ip_adapter"][f"{key_id}.to_v_ip.bias"] = to_v_ip.bias.data.cpu().clone()                                                                        
                                                                                                         
                          key_id += 1                                                                    
                                                                                                         
                      # ========== Save to disk ==========                                               
                      # Flatten for safetensors format                                                   
                      flat_state_dict = {}                                                               
                      for prefix, sub_dict in state_dict.items():                                        
                          for key, value in sub_dict.items():                                            
                              flat_state_dict[f"{prefix}.{key}"] = value                                 

                      safetensors.torch.save_file(flat_state_dict, f"{output_dir}/ip_adapter.safetensors")                            
                                                                                                         
                  # Pop weight so accelerate doesn't save in its default format                          
                  weights.pop()                                                                          
                                                                                                         
      def load_model_hook(models, input_dir):                                                            
          while len(models) > 0:                                                                         
              model = models.pop()                                                                       
              unwrapped = accelerator.unwrap_model(model)                                                            
                                                                                                         
              # Check if this is the transformer with IP adapter                                         
              if isinstance(unwrapped, type(accelerator.unwrap_model(transformer))):                                 
                  load_path = os.path.join(input_dir, "ip_adapter.safetensors")                          
                                                                                                         
                  if not os.path.exists(load_path):                                                      
                      raise ValueError(f"IP adapter checkpoint not found at {load_path}")                
                                                                                                         
                  # Load the flat state dict                                                             
                  with safetensors.torch.safe_open(load_path, framework="pt", device="cpu") as f:        
                      state_dict = {"image_proj": {}, "ip_adapter": {}}                                  
                                                                                                         
                      for key in f.keys():                                                               
                          if key.startswith("image_proj."):                                              
                              new_key = key.replace("image_proj.", "")                                   
                              state_dict["image_proj"][new_key] = f.get_tensor(key)                      
                          elif key.startswith("ip_adapter."):                                            
                              new_key = key.replace("ip_adapter.", "")                                   
                              state_dict["ip_adapter"][new_key] = f.get_tensor(key)                      
                                                                                                         
                  # ========== Load image_proj weights ==========                                        
                  image_proj_layer = unwrapped.encoder_hid_proj.image_projection_layers[0]               
                                                                                                         
                  # Convert "proj.weight" -> "image_embeds.weight"                                       
                  converted_image_proj_state = {}                                                        
                  for key, value in state_dict["image_proj"].items():                                    
                      new_key = key.replace("proj", "image_embeds")                                      
                      converted_image_proj_state[new_key] = value                                        
                                                                                                         
                  image_proj_layer.load_state_dict(converted_image_proj_state)                           
                                                                                                         
                  # ========== Load ip_adapter weights ==========                                        
                  key_id = 0                                                                             
                  for name, attn_processor in unwrapped.attn_processors.items():                         
                      if name.startswith("single_transformer_blocks"):                                   
                          continue                                                                       
                                                                                                         
                      if not isinstance(attn_processor, FluxIPAdapterAttnProcessor):                     
                          continue                                                                       
                                                                                                         
                      to_k_ip = attn_processor.to_k_ip[0]                                                
                      to_v_ip = attn_processor.to_v_ip[0]                                                
                                                                                                         
                      to_k_ip.weight.data.copy_(state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"])    
                      to_k_ip.bias.data.copy_(state_dict["ip_adapter"][f"{key_id}.to_k_ip.bias"])        
                      to_v_ip.weight.data.copy_(state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"])    
                      to_v_ip.bias.data.copy_(state_dict["ip_adapter"][f"{key_id}.to_v_ip.bias"])        
                                                                                                         
                      key_id += 1                                                                        
                                                                                                         
      return save_model_hook, load_model_hook                                                            
                                                           
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
                ip_adapter_image=test_sample["clip_images"],
                ).images[0]
            image_no_ip = pipeline(
                width=config.dataset.image_size,
                height=config.dataset.image_size,
                prompt=test_sample["prompts"],
                negative_prompt="",
                true_cfg_scale=config.train.guidance_scale,
                generator=generator,
                ).images[0]
    
    # Compute CLIP image-to-image similarity using torchmetrics
    target_image_pil = test_sample["pil_target_images"][0]
    condition_image_pil = test_sample["pil_condition_images"][0]
    
    # Convert PIL images to tensors [N, C, H, W] with values in [0, 255] as uint8
    to_tensor = T.ToTensor()
    gen_tensor = (to_tensor(image) * 255).to(torch.uint8).unsqueeze(0).to(accelerator.device)
    target_tensor = (to_tensor(target_image_pil) * 255).to(torch.uint8).unsqueeze(0).to(accelerator.device)
    
    with torch.no_grad():
        # pipeline's image_encoder + cosine similarity
        gen_inputs = pipeline.feature_extractor(images=image, return_tensors="pt").pixel_values.to(accelerator.device, dtype=weight_dtype)
        target_inputs = pipeline.feature_extractor(images=target_image_pil, return_tensors="pt").pixel_values.to(accelerator.device, dtype=weight_dtype)
        
        gen_embed = pipeline.image_encoder(gen_inputs).image_embeds
        target_embed = pipeline.image_encoder(target_inputs).image_embeds
        gen_embed = gen_embed / gen_embed.norm(dim=-1, keepdim=True)
        target_embed = target_embed / target_embed.norm(dim=-1, keepdim=True)
        
        clip_sim_pipeline = F.cosine_similarity(gen_embed, target_embed).item() * 100
        
        # Log comparison for debugging
        logger.info(f"CLIP sim (pipeline): {clip_sim_pipeline:.4f}")
        
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
                    'clip_sim_pipeline': clip_sim_pipeline,
                }
            )
    return image

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
    if config.dataset.type == 'subject':
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
    elif config.dataset.type == 'laion2b':
        # Pass rank/world_size for multi-GPU shard splitting (since we don't use accelerator.prepare for WebDataset)
        train_dataloader = build_laion2B_dataloader(
            config.dataset.shards_path + "/" + config.dataset.train_shard_pattern, 
            flux_pipeline.feature_extractor, 
            config, 
            split='train',
            rank=accelerator.process_index,
            world_size=accelerator.num_processes
        )
        # Don't shard test dataloader - all GPUs see same test data (fewer shards than GPUs typically)
        test_dataloader = build_laion2B_dataloader(
            config.dataset.shards_path + "/" + config.dataset.test_shard_pattern, 
            flux_pipeline.feature_extractor, 
            config, 
            split='test',
            rank=0,  # no sharding for test
            world_size=1
        )
    elif config.dataset.type == 'laion2b_local':
        train_dataloader = build_laion2B_local_dataloader(
            config.dataset.train_data_dir,
            flux_pipeline.feature_extractor,
            config,
            split='train'
        )
        test_dataloader = build_laion2B_local_dataloader(
            config.dataset.test_data_dir,
            flux_pipeline.feature_extractor,
            config,
            split='test'
        )                                                                          
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
    
    save_model_hook, load_model_hook = create_ip_adapter_hooks(accelerator, flux_pipeline.transformer)
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    # For WebDataset (IterableDataset), don't prepare the dataloader - accelerate can't
    # concatenate non-tensor data (strings) across processes. WebDataset handles its own sharding.
    if config.dataset.type == 'laion2b':
        transformer, optimizer, lr_scheduler = accelerator.prepare(
            flux_pipeline.transformer, optimizer, lr_scheduler
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            flux_pipeline.transformer, optimizer, train_dataloader, lr_scheduler
        )
    # reassign the wrapped transformer to the flux pipeline
    flux_pipeline.transformer = transformer
    
    if config.dataset.type in ['subject', 'laion2b_local']:
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
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    # logger.info(f"  Num Epochs = {config.train.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.train.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    if config.resume_from_checkpoint:
        if config.resume_path is not None:
            logger.info(f"Resuming from checkpoint {config.resume_path}.")
            accelerator.load_state(config.resume_path)
            global_step = int(os.path.basename(config.resume_path).split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            # TODO: implement resume from latest checkpoint by default
            raise ValueError("Resume path is not set.")
    else:
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
    
    # Training Loop
    # for epoch in range(first_epoch, config.train.num_train_epochs):
    while global_step < config.train.max_train_steps:
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            if batch is None:
                continue
            with accelerator.accumulate(transformer):
                # Compute the text embeddings from prompts
                prompts = batch["prompts"]
                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                            prompts, text_encoders, tokenizers
                        )
                
                # Compute ID-Adapter image embeddings 
                condition_clip = batch["clip_images"].to(device=accelerator.device, dtype=weight_dtype)
                image_embeds = flux_pipeline.encode_image(condition_clip, accelerator.device, 1) # [B, 768]
                joint_attention_kwargs = {"ip_adapter_image_embeds": image_embeds}
                
                # Convert images to latent space
                if config.cache_latents:
                    # TODO: Implement cache latents
                    raise ValueError("Cache latents is not supported yet.")
                else:
                    images = batch["images"].to(device=accelerator.device, dtype=flux_pipeline.vae.dtype)
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
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
                
            if global_step >= config.train.max_train_steps:
                break 
            
            # Generate test samples every N steps
            if global_step % config.logging.test_steps == 0:
                # Sync all GPUs before test generation to prevent OOM
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    transformer.eval()
                    test_sample = next(test_iter)
                    # Temporarily unwrap the transformer for inference
                    # DDP wrapping breaks pipeline access to transformer.config
                    flux_pipeline.transformer = accelerator.unwrap_model(transformer)
                    log_test_sample(pipeline=flux_pipeline,
                                    test_sample=test_sample,
                                    accelerator=accelerator,
                                    weight_dtype=weight_dtype,
                                    config=config,
                                    is_final_validation=False)
                    # Restore the wrapped transformer for training
                    flux_pipeline.transformer = transformer
                    transformer.train()
                    # Clear cache to prevent OOM
                    torch.cuda.empty_cache()
                # Sync again before resuming training
                accelerator.wait_for_everyone()
                      
if __name__ == "__main__":
    main()