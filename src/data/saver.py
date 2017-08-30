import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


# TODO remove numpy usage and use Pytorch on the GPU
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


def get_prediction_df(predictions, orig_size, threshold=0.5):
    total = len(predictions)
    results = [None] * total

    with tqdm(total=total, desc="Reshaping the results") as pbar:
        for i, (img, name) in enumerate(predictions):
            mask = cv2.resize(img, orig_size)
            mask = mask > threshold
            encoded = run_length_encode(mask)
            results[i] = [name, encoded]
            pbar.update(1)

    return pd.DataFrame(results, columns=["img", "rle_mask"])
