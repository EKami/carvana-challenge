import gzip
import csv
import cv2
import numpy as np


class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class PredictionsSaverCallback(Callback):
    def __init__(self, to_file, origin_img_size, threshold):
        self.threshold = threshold
        self.origin_img_size = origin_img_size
        self.to_file = to_file
        self.file = gzip.open(to_file, "wt", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["img", "rle_mask"])

    # https://www.kaggle.com/stainsby/fast-tested-rle
    def run_length_encode(self, mask):
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

    def get_mask_rle(self, prediction):
        """

        Args:
            prediction (np.ndarray): An array of predicted values

        Returns:
            str: A length encoded version of the passed prediction
        """
        mask = cv2.resize(prediction, self.origin_img_size)
        mask = mask > self.threshold
        return self.run_length_encode(mask)

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != "predict":
            return

        probs = kwargs['probs']
        files_name = kwargs['files_name']
        # Save the predictions
        for (pred, name) in zip(probs, files_name):
            rle = self.get_mask_rle(pred)
            self.writer.writerow([name, rle])

    def close_saver(self):
        self.file.flush()
        self.file.close()
        print("Predictions wrote in {} file".format(self.to_file))
