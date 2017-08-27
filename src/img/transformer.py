import torch
import numpy as np


def tensor_to_image(tensor, mean=0, std=1):
    image = tensor.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image * std + mean
    image = image.astype(dtype=np.uint8)
    # img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return image


def tensor_to_mask(tensor):
    label = tensor.numpy() * 255
    label = label.astype(dtype=np.uint8)
    return label


def image_to_tensor(image, mean=0, std=1.):
    image = image.astype(np.float32)
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))  # Change channels_last to channels_first
    tensor = torch.from_numpy(image)
    return tensor


def mask_to_tensor(mask, threshold=0.5):
    mask = mask
    mask = (mask > threshold).astype(np.float32)
    tensor = torch.from_numpy(mask).type(torch.FloatTensor)
    return tensor
