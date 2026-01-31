import torch 
from diffusers.pipelines.flux import FluxPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.embeddings import ImageProjection, MultiIPAdapterImageProjection
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT
from typing import Optional
from diffusers.utils import logging, is_torch_version, is_accelerate_available  
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import itertools

logger = logging.get_logger(__name__)

def setup_ip_adapter(
    pipeline: FluxPipeline,
    image_encoder_pretrained_model_name_or_path: Optional[str] = "image_encoder",
    image_encoder_subfolder: Optional[str] = "",
    image_encoder_dtype: torch.dtype = torch.float16,
    **kwargs,):
    
    """
    Parameters:
        image_encoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `./image_encoder`):
            Can be either:

                - A string, the *model id* (for example `openai/clip-vit-large-patch14`) of a pretrained model
                    hosted on the Hub.
                - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                    with [`ModelMixin.save_pretrained`].
        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
            is not used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.

        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether to only load local model weights and configuration files or not. If set to `True`, the model
            won't be downloaded from the Hub.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
            `diffusers-cli login` (stored in `~/.huggingface`) is used.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
            allowed by Git.
        low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
            Speed up model loading only loading the pretrained weights and not initializing the weights. This also
            tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
            Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
            argument to `True` will raise an error.
    """

    self = pipeline
    
    # Load the main state dict first.
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", None)
    token = kwargs.pop("token", None)
    revision = kwargs.pop("revision", None)
    low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
    
    if low_cpu_mem_usage and not is_accelerate_available():
        low_cpu_mem_usage = False
        logger.warning(
            "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
            " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
            " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
            " install accelerate\n```\n."
        )

    if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError(
            "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
            " `low_cpu_mem_usage=False`."
        )
    
    # load CLIP image encoder here if it has not been registered to the pipeline yet
    if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is None:
        if image_encoder_pretrained_model_name_or_path is not None:
            logger.info(f"loading image_encoder from {image_encoder_pretrained_model_name_or_path}")
            image_encoder = (
                CLIPVisionModelWithProjection.from_pretrained(
                    image_encoder_pretrained_model_name_or_path,
                    subfolder=image_encoder_subfolder,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    torch_dtype=image_encoder_dtype,
                )
                .to(self.device)
                .eval()
            )
            self.register_modules(image_encoder=image_encoder)
        else:
            raise ValueError(
                "`image_encoder` cannot be loaded because `pretrained_model_name_or_path_or_dict` is a state dict."
            )
    else:
        logger.warning(
            "image_encoder is not loaded since `image_encoder_folder=None` passed. You will not be able to use `ip_adapter_image` when calling the pipeline with IP-Adapter."
            "Use `ip_adapter_image_embeds` to pass pre-generated image embedding instead."
        )

    # create feature extractor if it has not been registered to the pipeline yet
    if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is None:
        # FaceID IP adapters don't need the image encoder so it's not present, in this case we default to 224
        default_clip_size = 224
        clip_image_size = (
            self.image_encoder.config.image_size if self.image_encoder is not None else default_clip_size
        )
        feature_extractor = CLIPImageProcessor(size=clip_image_size, crop_size=clip_image_size)
        self.register_modules(feature_extractor=feature_extractor)
    
    # set attention processor to IPAdapterAttnProcessor and initialize with existing weights
    from diffusers.models.transformers.transformer_flux import FluxIPAdapterAttnProcessor
    attn_procs = {}
    state_dict = self.transformer.state_dict()
    # for k, v in state_dict.items():
    #     if 'single_transformer_blocks' not in k:
    #         print(k)
    #         print(v.shape)
    
    for name in self.transformer.attn_processors.keys():
        if name.startswith("single_transformer_blocks"):
            attn_processor_class = self.transformer.attn_processors[name].__class__
            attn_procs[name] = attn_processor_class()
        else:
            cross_attention_dim = self.transformer.config.joint_attention_dim
            hidden_size = self.transformer.inner_dim
            attn_processor_class = FluxIPAdapterAttnProcessor
            num_image_text_embeds = 4
            attn_procs[name] = attn_processor_class(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=1.0,
                num_tokens=num_image_text_embeds,
                dtype=self.dtype,
                device=self.device,
            )
            W_akp = state_dict[name.split(".processor")[0]+'.add_k_proj.weight']
            W_avp = state_dict[name.split(".processor")[0]+'.add_v_proj.weight']
            b_akp = state_dict[name.split(".processor")[0]+'.add_k_proj.bias']
            b_avp = state_dict[name.split(".processor")[0]+'.add_v_proj.bias']
            W_ctx = state_dict['context_embedder.weight']
            b_ctx = state_dict['context_embedder.bias']
            attn_weights = {
                'to_k_ip.0.weight': W_akp @ W_ctx,
                'to_v_ip.0.weight': W_avp @ W_ctx,
                'to_k_ip.0.bias': W_akp @ b_ctx + b_akp,
                'to_v_ip.0.bias': W_avp @ b_ctx + b_avp
            }
            attn_procs[name].load_state_dict(attn_weights)
    self.transformer.set_attn_processor(attn_procs)
    
    ip_modules = [p for p in self.transformer.attn_processors.values() if isinstance(p, torch.nn.Module)]
    ip_modules = torch.nn.ModuleList(ip_modules)
 
    # initialize the image projection model, i.e. encoder_hid_proj in the transformer 
    num_image_text_embeds = 4 # TODO: get this from the config, 16 when IP-Plus
    # Use projection_dim (output dim of image_embeds), not hidden_size (internal transformer dim)
    image_projection = ImageProjection(
        cross_attention_dim=self.transformer.config.joint_attention_dim,
        image_embed_dim=self.image_encoder.config.projection_dim,
        num_image_text_embeds=num_image_text_embeds,
    )
    image_projection_layers = [image_projection]
    self.transformer.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
    self.transformer.config.encoder_hid_dim_type = "ip_image_proj"
    
    trainable_params = itertools.chain(self.transformer.encoder_hid_proj.parameters(), ip_modules.parameters())
    return trainable_params
 