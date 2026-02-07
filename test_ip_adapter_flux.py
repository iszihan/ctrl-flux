import torch
from diffusers import FluxPipeline
from diffusers.utils import load_image
from torchvision import transforms as T
import diffusers


pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")

# Resize shortest edge to 512, then center crop to 512x512
transform = T.Compose([
    T.Resize(512, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(512),
])
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flux_ip_adapter_input.jpg")
image = transform(image)

# pipe.load_ip_adapter(
#     "XLabs-AI/flux-ip-adapter",
#     weight_name="ip_adapter.safetensors",
#     image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
# )
# exit()
pipe.load_ip_adapter(
    #"runs/ipflux-laion2b-4l40s/ip_adapter-021000",
    "runs/ipflux-laion2b-4l40s/ip_adapter-054000",
    weight_name="ip_adapter.safetensors",
    image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
)
exit()
# Move newly loaded IP adapter components to GPU with correct dtype
pipe.to("cuda", dtype=torch.bfloat16)
pipe.set_ip_adapter_scale(1.0)

image = pipe(
    width=512,
    height=512,
    prompt="wearing sunglasses",
    negative_prompt="",
    true_cfg_scale=4.0,
    generator=torch.Generator().manual_seed(42),
    ip_adapter_image=image,
).images[0]

image.save('./runs/test/flux_ours21k_ip_adapter_output.jpg')