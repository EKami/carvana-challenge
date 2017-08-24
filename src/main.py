import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import img.augmentation as aug

import nn.unet as unet
from data.dataset import DatasetTools, ImageDataset

# Hyperparameters
img_resize = (128, 128)
batch_size = 16
epochs = 3


def train_unet128(trainloader: DataLoader):
    net = unet.UNet128((*img_resize, 3), 1)
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, masks = data

            # wrap them in Variable
            inputs, masks = Variable(inputs.cuda()), Variable(masks.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()


def main():
    # Download the datasets
    ds_tools = DatasetTools()
    ds_tools.download_dataset()
    X_train, y_train, X_valid, y_valid = ds_tools.get_train_valid_split()

    ds = ImageDataset(X_train, y_train, img_resize, X_transform=aug.data_transformator)
    dl = DataLoader(ds, batch_size, num_workers=4)

    # Launch train on Unet128
    train_unet128(dl)

if __name__ == "__main__":
    main()
