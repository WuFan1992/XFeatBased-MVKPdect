import torch
import numpy as np
from validation.hloc.utils.read_colmaps import read_extrinsics_binary


def PILtoTorch(pil_image):
    resized_image_PIL = pil_image
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def image_process(image, device= "cuda"):
    resized_image_rgb = PILtoTorch(image)
    gt_image = resized_image_rgb[:3, ...]
    original_image = gt_image.clamp(0.0, 1.0)
    original_image = original_image.to(device)
    return original_image    


def get_registered_images(images_bin_path):
    images = read_extrinsics_binary(images_bin_path)

    registered_imgs = set()

    for image_id, img in images.items():
        registered_imgs.add(img.name)

    return registered_imgs