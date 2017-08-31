import torch
from torch.utils.data import DataLoader
import img.augmentation as aug
from multiprocessing import cpu_count

import nn.unet as unet
from data.dataset import DatasetTools, TrainImageDataset, TestImageDataset
import data.saver as saver
from torch.utils.data.sampler import RandomSampler
import nn.classifier


def main():
    # Hyperparameters
    img_resize = (128, 128)
    batch_size = 32
    epochs = 20
    threshold = 0.5

    # Optional parameters
    threads = cpu_count()
    use_cuda = torch.cuda.is_available()

    # Put None to work on full dataset
    sample_size = None  # 0.2

    # Download the datasets
    ds_tools = DatasetTools()
    ds_tools.download_dataset()

    # Get the path to the files for the neural net
    X_train, y_train, X_valid, y_valid = ds_tools.get_train_valid_split(sample_size=sample_size)
    X_test = ds_tools.get_test_files(sample_size)

    # Get the original images size (assuming they are all the same size)
    origin_img_size = ds_tools.get_image_size(X_train[0])

    train_ds = TrainImageDataset(X_train, y_train, img_resize, X_transform=aug.data_transformer)
    train_loader = DataLoader(train_ds, batch_size,
                              sampler=RandomSampler(train_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)

    valid_ds = TrainImageDataset(X_valid, y_valid, img_resize, threshold=threshold)
    valid_loader = DataLoader(valid_ds, batch_size,
                              num_workers=threads,
                              pin_memory=use_cuda)

    test_ds = TestImageDataset(X_test, img_resize)
    test_loader = DataLoader(test_ds, batch_size,
                             num_workers=threads,
                             pin_memory=use_cuda)

    # Launch train on Unet128
    net = unet.UNet128((3, *img_resize), 1)
    classifier = nn.classifier.CarvanaClassifier(net, train_loader, valid_loader)
    classifier.train(epochs)

    # Predict & save
    output_file = "submit.csv.gz"
    classifier.predict(test_loader, to_file=output_file,
                       t_fnc=saver.get_prediction_transformer,
                       fnc_args=[origin_img_size, 0.5])  # The get_prediction_transformer arguments
    print("Predictions wrote in {} file".format(output_file))

if __name__ == "__main__":
    main()
