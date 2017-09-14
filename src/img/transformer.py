import torch
import numpy as np
from PIL import Image


def image_to_tensor(image, mean=0, std=1.):
    """
    Transforms an image to a tensor
    Args:
        image (np.ndarray): A RGB array image
        mean: The mean of the image values
        std: The standard deviation of the image values

    Returns:
        tensor: A Pytorch tensor
    """
    image = image.astype(np.float32)
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    tensor = torch.from_numpy(image)
    return tensor


def mask_to_tensor(mask, threshold):
    """
    Transforms a mask to a tensor
    Args:
        mask (np.ndarray): A greyscale mask array
        threshold: The threshold used to consider the mask present or not

    Returns:
        tensor: A Pytorch tensor
    """
    mask = mask
    mask = (mask > threshold).astype(np.float32)
    tensor = torch.from_numpy(mask).type(torch.FloatTensor)
    return tensor


def center_cropping_resize(img, new_size):
    """
        Resize an image and keep its aspect ratio
    Args:
        img (Image): The Pillow image to resize
        new_size (tuple): The size as tuple (h, w)

    Returns:
        Image: The resized image
    """
    largest = max(img.width, img.height)
    new_h = np.round(np.multiply(new_size[0] / largest, img.size[0])).astype(int)
    new_w = np.round(np.multiply(new_size[1] / largest, img.size[1])).astype(int)
    return img.resize((new_h, new_w), Image.ANTIALIAS)


def get_center_crop_size(img_path, img_size):
    img = Image.open(img_path)
    largest = max(img.width, img.height)
    new_h = np.round(np.multiply(img_size[0] / largest, img.size[0])).astype(int)
    new_w = np.round(np.multiply(img_size[1] / largest, img.size[1])).astype(int)
    return new_h, new_w