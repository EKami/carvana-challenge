import numpy as np
import cv2


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def get_prediction_transformer(prediction, orig_size, threshold=0.5):
    """
    A function that takes in a prediction array, reshape it to
    its original size and length encore it
    :param prediction: An array of predicted values
    :param orig_size: The original size of the prediction array
    :param threshold: The threshold used to consider a mask present or not
    :return: A length encoded version of the passed prediction
    """
    mask = cv2.resize(prediction, orig_size)
    mask = mask > threshold
    return run_length_encode(mask)

