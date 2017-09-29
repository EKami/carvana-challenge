import nn.classifier
import nn.unet_origin as unet_origin
import nn.unet as unet_custom
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
import multiprocessing


def main():
    # Hyperparameters
    input_img_resize = (572, 572)  # The resize size of the input images of the neural net
    output_img_resize = (388, 388)  # The resize size of the output images of the neural net
    # input_img_resize = (1024, 1024)
    # output_img_resize = (1024, 1024)
    batch_size = 3
    epochs = 50
    threshold = 0.5
    validation_size = 0.2
    sample_size = None  # Put 'None' to work on full dataset

    # -- Optional parameters
    threads = cpu_count()
    use_cuda = torch.cuda.is_available()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Training callbacks
    tb_viz_cb = TensorboardVisualizerCallback(os.path.join(script_dir, '../logs/tb_viz'))
    tb_logs_cb = TensorboardLoggerCallback(os.path.join(script_dir, '../logs/tb_logs'))
    model_saver_cb = ModelSaverCallback(os.path.join(script_dir, '../output/models/model_' +
                                                     helpers.get_model_timestamp()), verbose=True)

    # Download the datasets
    ds_fetcher = DatasetFetcher()
    ds_fetcher.download_dataset()

    # Get the path to the files for the neural net
    X_train, y_train, X_valid, y_valid = ds_fetcher.get_train_files(sample_size=sample_size,
                                                                    validation_size=validation_size)
    full_x_test = ds_fetcher.get_test_files(sample_size)

    # -- Computed parameters
    # Get the original images size (assuming they are all the same size)
    origin_img_size = ds_fetcher.get_image_size(X_train[0])

    # Testing callbacks
    pred_saver_cb = PredictionsSaverCallback(os.path.join(script_dir, '../output/submit.csv.gz'),
                                             origin_img_size, threshold)

    # -- Define our neural net architecture
    # The original paper has 1 input channel,
    # in our case we have 3 (RGB)
    #net = unet_custom.UNet1024((3, *input_img_resize))
    net = unet_origin.UNetOriginal((3, *input_img_resize))
    classifier = nn.classifier.CarvanaClassifier(net, epochs)

    train_ds = TrainImageDataset(X_train, y_train, input_img_resize, output_img_resize,
                                 X_transform=aug.augment_img)
    train_loader = DataLoader(train_ds, batch_size,
                              sampler=RandomSampler(train_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)

    valid_ds = TrainImageDataset(X_valid, y_valid, input_img_resize, output_img_resize,
                                 threshold=threshold)
    valid_loader = DataLoader(valid_ds, batch_size,
                              sampler=SequentialSampler(valid_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)

    print("Training on {} samples and validating on {} samples "
          .format(len(train_loader.dataset), len(valid_loader.dataset)))

    # Train the classifier
    classifier.train(train_loader, valid_loader, epochs, callbacks=[tb_viz_cb, tb_logs_cb, model_saver_cb])

    test_ds = TestImageDataset(full_x_test, input_img_resize)
    test_loader = DataLoader(test_ds, batch_size,
                             sampler=SequentialSampler(test_ds),
                             num_workers=threads,
                             pin_memory=use_cuda)

    # Predict & save
    classifier.predict(test_loader, callbacks=[pred_saver_cb])
    pred_saver_cb.close_saver()


if __name__ == "__main__":
    # Workaround for a deadlock issue on Pytorch 0.2.0: https://github.com/pytorch/pytorch/issues/1838
    multiprocessing.set_start_method('spawn', force=True)
    main()
