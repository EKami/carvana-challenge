import numpy as np
import cv2


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    """
    Args:
        mask (np.ndarray): 1 = mask, 0 = background

    Returns:
        str: run length as string formated
    """
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def get_prediction_transformer(prediction, orig_size, threshold=0.5):
    """

    Args:
        prediction (np.ndarray): An array of predicted values
        orig_size (tuple): The original size of the prediction array
        threshold (float): The threshold used to consider a mask present or not

    Returns:
        str: A length encoded version of the passed prediction
    """
    mask = cv2.resize(prediction, orig_size)
    mask = mask > threshold
    return run_length_encode(mask)

