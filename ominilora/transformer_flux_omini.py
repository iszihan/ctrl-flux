import torch
from typing import List, Union, Optional, Dict, Any, Callable, Type, Tuple

from diffusers.pipelines import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    FluxTransformer2DModel,
    calculate_shift,
    retrieve_timesteps,
    np,
)
from diffusers.models.attention_processor import Attention, F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import _get_qkv_projections
from transformers import pipeline

from peft.tuners.tuners_utils import BaseTunerLayer
from accelerate.utils import is_torch_version
from diffusers.models.attention_dispatch import dispatch_attention_fn

from contextlib import contextmanager
from arrgh import arrgh

import cv2

from PIL import Image, ImageFilter
from torchvision.transforms.functional import to_pil_image

'''
What I really wanna do is to have a Omini-version of 
FluxOminiTransformer2DModel, 
FluxOminiTransformerBlock, 
FluxOminiSingleTransformerBlock, 
FluxOminiAttention etc. 
and re-instantiate the transformer within the pipeline with these modules and run forward.

Ideally, the modules in diffusers should be modified to support Omini-type of lora mechanism.

For now, we will do the easier thing of just overriding the forward method but follow the diffuse style.
'''


def clip_hidden_states(hidden_states: torch.FloatTensor) -> torch.FloatTensor:
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return hidden_states

@contextmanager
def specify_lora(lora_modules: List[BaseTunerLayer], specified_lora):
    # Filter valid lora modules
    valid_lora_modules = [m for m in lora_modules if isinstance(m, BaseTunerLayer)]
    # Save original scales
    original_scales = [
        {
            adapter: module.scaling[adapter]
            for adapter in module.active_adapters
            if adapter in module.scaling
        }
        for module in valid_lora_modules
    ]
    # Enter context: adjust scaling
    for module in valid_lora_modules:
        for adapter in module.active_adapters:
            if adapter in module.scaling:
                module.scaling[adapter] = 1 if adapter == specified_lora else 0
    try:
        yield
    finally:
        # Exit context: restore original scales
        for module, scales in zip(valid_lora_modules, original_scales):
            for adapter in module.active_adapters:
                if adapter in module.scaling:
                    module.scaling[adapter] = scales[adapter]

# TODO: should just have a FluxOminiAttentionProcessor
def omini_attn_forward(
    self: Attention,
    hidden_states: List[torch.Tensor],
    encoder_hidden_states: torch.Tensor = None,
    position_embs: Optional[List[torch.Tensor]] = None,
    adapters: List[str] = None,
    group_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs: dict,
):   
    if encoder_hidden_states is not None:
        adapter_offset = 1
    else:
        adapter_offset = 0
        
    # Compute QKV for each branch
    queries, keys, values = [], [], []
    for i, hidden_state in enumerate(hidden_states):
        with specify_lora((self.to_q, self.to_k, self.to_v), adapters[i+adapter_offset]):
            query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
                self, hidden_state, encoder_hidden_states if encoder_hidden_states is not None and i == 0 else None
                )
            query = query.unflatten(-1, (self.heads, -1))
            key = key.unflatten(-1, (self.heads, -1))
            query, key = self.norm_q(query), self.norm_k(key)
            value = value.unflatten(-1, (self.heads, -1))
            if i ==0 and encoder_hidden_states is not None:
                encoder_query = encoder_query.unflatten(-1, (self.heads, -1))
                encoder_key = encoder_key.unflatten(-1, (self.heads, -1))
                encoder_query, encoder_key = self.norm_added_q(encoder_query), self.norm_added_k(encoder_key)
                encoder_value = encoder_value.unflatten(-1, (self.heads, -1))
                queries.append(encoder_query)
                keys.append(encoder_key)
                values.append(encoder_value)
            queries.append(query)
            keys.append(key)
            values.append(value)
    
    # Apply rotary embedding
    assert position_embs is not None
    queries = [apply_rotary_emb(q, position_embs[i], sequence_dim=1) for i, q in enumerate(queries)]
    keys = [apply_rotary_emb(k, position_embs[i], sequence_dim=1) for i, k in enumerate(keys)]
    
    # Attention
    attention_outputs= []
    for i, query in enumerate(queries):
        keys_, values_ = [], []
        for j, (k, v) in enumerate(zip(keys, values)):
            if (group_mask is not None) and not (group_mask[i][j].item()):
                continue
            keys_.append(k)
            values_.append(v)
        attn_output = dispatch_attention_fn(
            query,
            torch.cat(keys_, dim=1),
            torch.cat(values_, dim=1),
            attn_mask=attention_mask,
            backend=self.processor._attention_backend,
            parallel_config=self.processor._parallel_config,
        )
        attn_output = attn_output.flatten(2, 3).to(query.dtype)
        attention_outputs.append(attn_output)
    
    hidden_states_out = []
    for i, _ in enumerate(hidden_states):
        hidden_states[i] = attention_outputs[i+adapter_offset]
        if getattr(self, "to_out", None) is not None:
            with specify_lora((self.to_out[0],), adapters[i+adapter_offset]):
                hidden_states[i] = self.to_out[0](hidden_states[i])
        hidden_states_out.append(hidden_states[i])
    if encoder_hidden_states is not None:
        encoder_hidden_state_out = self.to_add_out(attention_outputs[0])
        return hidden_states_out, encoder_hidden_state_out
    else:
        return hidden_states_out

def omini_single_block_forward(
    self,
    hidden_states: List[torch.Tensor],
    tembs: List[torch.Tensor],
    position_embs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    adapters: List[str] = None,
    attn_forward=omini_attn_forward,
    **kwargs: dict,
):  
    mlp_hidden_states, gates = [], []
    hidden_state_norm = []
    for i, hidden_state in enumerate(hidden_states):
        with specify_lora((self.norm.linear, self.proj_mlp), adapters[i]):
            h_norm, gate = self.norm(hidden_state, emb=tembs[i])
            mlp_hidden_states.append(self.act_mlp(self.proj_mlp(h_norm)))
            gates.append(gate)
            hidden_state_norm.append(h_norm)
    
    attn_outputs = attn_forward(
        self.attn, 
        hidden_states=hidden_state_norm, 
        encoder_hidden_states=None,
        adapters=adapters, 
        position_embs=position_embs, 
        **joint_attention_kwargs,
        **kwargs)
   
    hidden_states_out = []
    for i, hidden_state in enumerate(hidden_states):
        with specify_lora((self.proj_out,), adapters[i]):
            h = torch.cat([attn_outputs[i], mlp_hidden_states[i]], dim=2)
            h = gates[i].unsqueeze(1) * self.proj_out(h) + hidden_state
            hidden_states_out.append(clip_hidden_states(h))
    return hidden_states_out

def omini_block_forward(
    self,
    hidden_states: List[torch.Tensor],
    encoder_hidden_states: torch.Tensor,
    tembs: List[torch.Tensor],
    position_embs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    adapters: List[str] = None,
    attn_forward=omini_attn_forward,
    **kwargs: dict,
):
    # Normalization for hidden states
    hidden_states_variables = []
    for i, hidden_state in enumerate(hidden_states):
        with specify_lora((self.norm1.linear,), adapters[1+i]):
            hidden_states_variables.append(self.norm1(hidden_state, emb=tembs[1+i]))
    
    # Normalization for encoder hidden states
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=tembs[0]
        )
    
    attention_outputs = attn_forward(
        self.attn,
        hidden_states=[hidden_state[0] for hidden_state in hidden_states_variables],
        encoder_hidden_states=norm_encoder_hidden_states,
        position_embs=position_embs,
        adapters=adapters,
        **joint_attention_kwargs,
        **kwargs,
    )
    
    
    # Apply MLP and gating mechanisms to encoder hidden states 
    encoder_hidden_states = encoder_hidden_states + attention_outputs[1] * c_gate_msa.unsqueeze(1)
    encoder_hidden_states = self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    encoder_hidden_states = self.ff_context(encoder_hidden_states) * c_gate_mlp.unsqueeze(1) + encoder_hidden_states
    encoder_hidden_states = clip_hidden_states(encoder_hidden_states)
    
    # Apply MLP and gating mechanisms to hidden states
    hidden_states_out = []
    for i, hidden_state_variable in enumerate(hidden_states_variables):
        _, gate_msa, shift_mlp, scale_mlp, gate_mlp = hidden_state_variable
        hidden_state = hidden_states[i] + attention_outputs[0][i] * gate_msa.unsqueeze(1)
        hidden_state = self.norm2(hidden_state) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        with specify_lora((self.ff.net[2],), adapters[1+i]):
            hidden_state = hidden_state + self.ff(hidden_state) * gate_mlp.unsqueeze(1)
        hidden_states_out.append(clip_hidden_states(hidden_state))
    
    return hidden_states_out, encoder_hidden_states

def omini_transformer_forward(
    transformer: FluxTransformer2DModel,
    hidden_states: List[torch.Tensor], # Need to be list of take in condition inputs
    encoder_hidden_states: torch.Tensor,
    pooled_projections: torch.Tensor,
    timesteps: List[torch.LongTensor] = None,
    img_ids: List[torch.Tensor] = None, # Need to be list of take in condition inputs
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples: Optional[List[torch.Tensor]] = None,
    controlnet_single_block_samples: Optional[List[torch.Tensor]] = None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
    # Omini-specific parameters
    adapters: List[str] = None,
    attn_forward=omini_attn_forward,
    single_block_forward=omini_single_block_forward,
    block_forward=omini_block_forward,
    **kwargs: dict,
):
    self = transformer
    assert len(adapters) == len(timesteps)
    
    # Embed image tokens for each branch
    hidden_states_all = []
    for i, hidden_state in enumerate(hidden_states):
        with specify_lora((self.x_embedder,), adapters[1+i]):
            hidden_states_all.append(self.x_embedder(hidden_state))
    
    # Embed text tokens
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    
    # Timestep embedding (with guidance and pooled_projections)
    tembs = [
        self.time_text_embed(timesteps[i], pooled_projections)
        if guidance is None
        else self.time_text_embed(timesteps[i], guidance, pooled_projections)
        for i in range(len(timesteps))
    ]
    
    # Position Embedding
    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
        
    for i in range(len(img_ids)):
        if img_ids[i].ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids[i] = img_ids[0]
    position_embs = [self.pos_embed(each) for each in (*[txt_ids], *img_ids)]
    
    # Prepare the gradient checkpointing kwargs
    gckpt_kwargs: Dict[str, Any] = (
        {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
    )
    # Dual-branch blocks forward pass
    for block in self.transformer_blocks:
        block_kwargs = {
            "self": block,
            "hidden_states": hidden_states_all,
            "encoder_hidden_states": encoder_hidden_states,
            "tembs": tembs,
            "position_embs": position_embs,
            "joint_attention_kwargs": joint_attention_kwargs,
            "adapters": adapters,
            "attn_forward": attn_forward,
        }
        if self.training and self.gradient_checkpointing:
            hidden_states_all, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                block_forward, **block_kwargs, **gckpt_kwargs
            )
        else:
            hidden_states_all, encoder_hidden_states = block_forward(**block_kwargs)
    
    # Single-branch blocks forward pass
    hidden_states_all = [encoder_hidden_states, *hidden_states_all]
    for block in self.single_transformer_blocks:
        block_kwargs = {
            "self": block,
            "hidden_states": hidden_states_all,
            "tembs": tembs,
            "position_embs": position_embs,
            "joint_attention_kwargs": joint_attention_kwargs,
            "adapters": adapters,
            "attn_forward": attn_forward,
            **kwargs,
        }
        if self.training and self.gradient_checkpointing:
            hidden_states_all = torch.utils.checkpoint.checkpoint(
                single_block_forward, **block_kwargs, **gckpt_kwargs
            )
        else:
            hidden_states_all = single_block_forward(**block_kwargs)
    
    hidden_states = self.norm_out(hidden_states_all[1], tembs[1])
    output = self.proj_out(hidden_states)
    return (output,)
