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