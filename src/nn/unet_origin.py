import torch.nn as nn


class UNetOriginal(nn.Module):
    def __init__(self, in_shape):
        super(UNetOriginal, self).__init__()
        channels, height, width = in_shape
