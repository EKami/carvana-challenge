import gzip
import csv
import cv2
import numpy as np
import bcolz


class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class KagglePredictionsSaverCallback(Callback):
    def __init__(self, to_file, origin_img_size, threshold):
        """
            Callback intended to be executed at each batch iteration of
            the prediction pass. It allows to save the predictions in
            a compressed gzip csv file
        Args:
            to_file (str): The file where to save the predictions
            origin_img_size (tuple): The original image size
            threshold (float): The threshold used to consider the mask present or not
        """
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

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != "predict":
            return

        probs = kwargs['probs']
        files_name = kwargs['files_name']
        # Save the predictions
        for (pred, name) in zip(probs, files_name):
            mask = cv2.resize(pred, self.origin_img_size)
            mask = mask > self.threshold
            rle = self.run_length_encode(mask)
            self.writer.writerow([name, rle])

    def get_predictions(self):
        """
            Returns either the file object or a numpy array
            if the callback wasn't meant to store the results in a file
        Returns:
            (file, array): A file object or a numpy array
        """
        return self.file

    def close_saver(self):
        self.file.flush()
        self.file.close()
        print("Predictions wrote in {} file".format(self.to_file))


class BcolzPredictionsSaverCallback(Callback):
    def __init__(self, to_file, origin_img_size):
        self.origin_img_size = origin_img_size
        self.to_file = to_file
        self.bc_arr = None

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != "predict":
            return

        probs = kwargs['probs']
        files_name = kwargs['files_name']

        pred_names = []
        pred_arr = np.empty((len(probs), *self.origin_img_size), dtype=np.float32)
        # Save the predictions to bcolz
        for i, (pred, name) in enumerate(zip(probs, files_name)):
            mask = cv2.resize(pred, self.origin_img_size)
            pred_names.append(name)
            pred_arr[i] = mask.T

        pred_names = np.array(pred_names, dtype=str)
        if self.bc_arr:
            self.bc_arr[0].append(pred_names)
            self.bc_arr[1].append(pred_arr)
        else:
            self.bc_arr = [bcolz.carray(pred_names,
                                        cparams=bcolz.cparams(clevel=9, cname='blosclz', shuffle=bcolz.NOSHUFFLE),
                                        rootdir=self.to_file + "_names.bc", mode='w'),
                           bcolz.carray(pred_arr,
                                        cparams=bcolz.cparams(clevel=9, cname='blosclz', shuffle=bcolz.NOSHUFFLE),
                                        rootdir=self.to_file + "_arr.bc", mode='w')
                           ]

        for b in self.bc_arr:
            b.flush()

    def get_prediction_array(self):
        bc_names = bcolz.open(rootdir=self.to_file + "_names.bc", mode='r')
        bc_arr = bcolz.open(rootdir=self.to_file + "_arr.bc", mode='r')
        return zip(bc_names, bc_arr)
