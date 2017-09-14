import gzip
import csv
import cv2
import numpy as np


class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class PredictionsSaverCallback(Callback):
    def __init__(self, to_file, origin_img_size, threshold):
        """
            Callback intended to be executed at each batch iteration of
            the prediction pass. It allows to save the predictions in
            a compressed file or in an array
        Args:
            to_file (str, None): The file where to save the predictions.
                If None is given, the predictions are saved in a numpy array
            origin_img_size (tuple): The original image size
            threshold (float): The threshold used to consider the mask present or not
        """
        self.threshold = threshold
        self.origin_img_size = origin_img_size
        self.to_file = to_file
        if self.to_file:
            self.file = gzip.open(to_file, "wt", newline="")
            self.writer = csv.writer(self.file)
            self.writer.writerow(["img", "rle_mask"])
        else:
            self.predictions = []

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

    def _get_mask(self, prediction):
        mask = cv2.resize(prediction, self.origin_img_size)
        return mask > self.threshold

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != "predict":
            return

        probs = kwargs['probs']
        files_name = kwargs['files_name']
        # Save the predictions
        for (pred, name) in zip(probs, files_name):
            mask = self._get_mask(pred)
            rle = self.run_length_encode(mask)
            if self.to_file:
                self.writer.writerow([name, rle])
            else:
                self.predictions.append([name, mask])

    def get_predictions(self):
        """
            Returns either the file object or a numpy array
            if the callback wasn't meant to store the results in a file
        Returns:
            (file, array): A file object or a numpy array
        """
        if self.to_file:
            return self.file
        else:
            return self.predictions

    def close_saver(self):
        if self.to_file:
            self.file.flush()
            self.file.close()
            print("Predictions wrote in {} file".format(self.to_file))
