import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter


class TensorboardVisualizerCallback:
    def __init__(self, path_to_files):
        """
            Callback intended to be executed at each epoch
            of the training which goal is to display the result
            of the last validation batch in Tensorboard
        Args:
            path_to_files:
        """
        self.path_to_files = path_to_files
        writer = SummaryWriter()

    def __call__(self, *args, **kwargs):
        net = kwargs['net']
        last_images, last_targets, last_preds = kwargs['last_val_batch']
        for image, target, pred in zip(last_images, last_targets, last_preds):
            # TODO finish
            pass
