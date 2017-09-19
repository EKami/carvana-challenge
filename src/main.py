import nn.classifier
import nn.unet as unet
import helpers

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import img.augmentation as aug
from data.fetcher import DatasetFetcher
import nn.classifier
from nn.train_callbacks import TensorboardVisualizerCallback, TensorboardLoggerCallback, ModelSaverCallback
from nn.test_callbacks import PredictionsSaverCallback

import os
from multiprocessing import cpu_count

from data.dataset import TrainImageDataset, TestImageDataset
import img.transformer as transformer


def main():
    # Clear log dir first
    helpers.clear_logs_folder()

    # Hyperparameters
    img_resize = (1024, 1024)
    batch_size = 2
    epochs = 50
    threshold = 0.5
    validation_size = 0.2
    sample_size = None  # Put None to work on full dataset

    # Training on 4576 samples and validating on 512 samples
    # -- Optional parameters
    threads = cpu_count()
    use_cuda = torch.cuda.is_available()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Download the datasets
    ds_fetcher = DatasetFetcher()
    ds_fetcher.download_dataset()

    # Get the path to the files for the neural net
    # We don't want to split train/valid for KFold crossval
    X_train, y_train, X_valid, y_valid = ds_fetcher.get_train_files(sample_size=sample_size,
                                                                    validation_size=validation_size)
    full_x_test = ds_fetcher.get_test_files(sample_size)

    # -- Computed parameters
    # Get the original images size (assuming they are all the same size)
    origin_img_size = ds_fetcher.get_image_size(X_train[0])
    # The image kept its aspect ratio so we need to recalculate the img size for the nn
    img_resize_centercrop = transformer.get_center_crop_size(X_train[0], img_resize)  # Training callbacks
    tb_viz_cb = TensorboardVisualizerCallback(os.path.join(script_dir, '../logs/tb_viz'))
    tb_logs_cb = TensorboardLoggerCallback(os.path.join(script_dir, '../logs/tb_logs'))
    model_saver_cb = ModelSaverCallback(os.path.join(script_dir, '../output/models/model_' +
                                                     helpers.get_model_timestamp()), verbose=True)

    # Testing callbacks
    pred_saver_cb = PredictionsSaverCallback(os.path.join(script_dir, '../output/submit.csv.gz'),
                                             origin_img_size, threshold)

    # Define our neural net architecture
    net = unet.UNet1024((3, *img_resize_centercrop))
    classifier = nn.classifier.CarvanaClassifier(net, epochs)

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

    print("Training on {} samples and validating on {} samples "
          .format(len(train_loader.dataset), len(valid_loader.dataset)))

    classifier.train(train_loader, valid_loader, epochs,
                     callbacks=[tb_viz_cb, tb_logs_cb, model_saver_cb])

    test_ds = TestImageDataset(full_x_test, img_resize)
    test_loader = DataLoader(test_ds, batch_size,
                             sampler=SequentialSampler(test_ds),
                             num_workers=threads,
                             pin_memory=use_cuda)

    # Predict & save
    classifier.predict(test_loader, callbacks=[pred_saver_cb])
    pred_saver_cb.close_saver()


if __name__ == "__main__":
    main()
