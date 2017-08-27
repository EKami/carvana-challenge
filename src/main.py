import torch
from torch.utils.data import DataLoader
import img.augmentation as aug
from multiprocessing import cpu_count

import nn.unet as unet
from data.dataset import DatasetTools, ImageDataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import nn.classifier


def main():
    # Hyperparameters
    img_resize = (128, 128)
    batch_size = 32
    epochs = 3
    threads = cpu_count()
    use_cuda = torch.cuda.is_available()

    # Download the datasets
    ds_tools = DatasetTools()
    train_data, test_data, metadata_csv, train_masks_csv, train_masks_data = ds_tools.download_dataset()
    X_train, y_train, X_valid, y_valid = ds_tools.get_train_valid_split()

    train_ds = ImageDataset(X_train, y_train, img_resize, X_transform=aug.data_transformator)
    train_loader = DataLoader(train_ds, batch_size,
                              sampler=RandomSampler(train_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)

    valid_ds = ImageDataset(X_valid, y_valid, img_resize)
    valid_loader = DataLoader(valid_ds, batch_size,
                              sampler=SequentialSampler(valid_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)

    test_ds = ImageDataset(test_data, img_resize=img_resize)
    test_loader = DataLoader(test_ds, batch_size, num_workers=threads,
                             pin_memory=use_cuda)

    # Launch train on Unet128
    net = unet.UNet128((3, *img_resize), 1)
    classifier = nn.classifier.CarvanaClassifier(net, train_loader, valid_loader)
    classifier.train(epochs)

    # Predict
    classifier.predict(test_loader)

if __name__ == "__main__":
    main()
