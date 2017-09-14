import nn.classifier
import nn.unet as unet
import helpers

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import img.augmentation as aug
from data.dataset import DatasetTools
import nn.classifier
from nn.train_callbacks import TensorboardVisualizerCallback, TensorboardLoggerCallback, ModelSaverCallback

import os
import numpy as np
from multiprocessing import cpu_count
from sklearn.model_selection import KFold

from data.dataset import TrainImageDataset, TestImageDataset
import data.saver as saver
import img.transformer as transformer


def main():
    # Clear log dir first
    helpers.clear_logs_folder()

    # Hyperparameters
    img_resize = (1024, 1024)
    batch_size = 3
    epochs = 6
    threshold = 0.5
    n_fold = 3
    sample_size = 0.2  # None  # Put None to work on full dataset

    # -- Optional parameters
    threads = cpu_count()
    use_cuda = torch.cuda.is_available()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Training callbacks
    tb_viz_cb = TensorboardVisualizerCallback(os.path.join(script_dir, '../logs/tb_viz'))
    tb_logs_cb = TensorboardLoggerCallback(os.path.join(script_dir, '../logs/tb_logs'))
    model_saver_cb = ModelSaverCallback(os.path.join(script_dir, '../output/models/model_' +
                                                     helpers.get_model_timestamp()), verbose=True)
    kf = KFold(n_splits=n_fold, shuffle=True)

    # Download the datasets
    ds_tools = DatasetTools()
    ds_tools.download_dataset()

    # Get the path to the files for the neural net
    # We don't want to split train/valid for KFold crossval
    full_x_train, full_y_train, _, _ = ds_tools.get_train_files(sample_size=sample_size, validation_size=0)
    full_x_test = ds_tools.get_test_files(sample_size)

    # -- Computed parameters
    # Get the original images size (assuming they are all the same size)
    origin_img_size = ds_tools.get_image_size(full_x_train[0])
    # The image kept its aspect ratio so we need to recalculate the img size for the nn
    img_resize_centercrop = transformer.get_center_crop_size(full_x_train[0], img_resize)
    # Calculate epoch per fold for cross validation
    epochs_per_fold = np.maximum(1, np.round(epochs / n_fold).astype(int))

    # Define our nn architecture
    # net = unet.UNet128((3, *img_resize))
    net = unet.UNet1024((3, *img_resize_centercrop))
    classifier = nn.classifier.CarvanaClassifier(net, epochs_per_fold * n_fold)

    # Launch the training on k folds
    for i, (train_indexes, valid_indexes) in enumerate(kf.split(full_x_train)):
        X_train = full_x_train[train_indexes]
        y_train = full_y_train[train_indexes]
        X_valid = full_x_train[valid_indexes]
        y_valid = full_y_train[valid_indexes]

        train_ds = TrainImageDataset(X_train, y_train, img_resize, X_transform=aug.augment_img)
        train_loader = DataLoader(train_ds, batch_size,
                                  sampler=RandomSampler(train_ds),
                                  num_workers=threads,
                                  pin_memory=use_cuda)

        valid_ds = TrainImageDataset(X_valid, y_valid, img_resize, threshold=threshold)
        valid_loader = DataLoader(valid_ds, batch_size,
                                  sampler=SequentialSampler(valid_ds),
                                  num_workers=threads,
                                  pin_memory=use_cuda)

        if i == 0:
            print("Training on {} samples and validating on {} samples "
                  .format(len(train_loader.dataset), len(valid_loader.dataset)))

        model_saver_cb.set_suffix("_kfold_" + str(i + 1))
        classifier.train(train_loader, valid_loader, epochs_per_fold,
                         callbacks=[tb_viz_cb, tb_logs_cb, model_saver_cb])
        print("KFold {} finished.".format(str(i + 1)))

    test_ds = TestImageDataset(full_x_test, img_resize)
    test_loader = DataLoader(test_ds, batch_size,
                             sampler=SequentialSampler(test_ds),
                             num_workers=threads,
                             pin_memory=use_cuda)

    # Predict & save
    output_file = os.path.join(script_dir, '../output/submit.csv.gz')
    classifier.predict(test_loader, to_file=output_file,
                       t_fnc=saver.get_prediction_transformer,
                       fnc_args=[origin_img_size, 0.5])


if __name__ == "__main__":
    main()
