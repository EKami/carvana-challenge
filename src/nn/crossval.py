import os
import torch
from torch.utils.data import DataLoader
import img.augmentation as aug
from multiprocessing import cpu_count

import nn.unet as unet
from nn.callbacks import TensorboardVisualizerCallback
from sklearn.model_selection import KFold

from torch.utils.data.sampler import RandomSampler, SequentialSampler
import nn.classifier

from data.dataset import TrainImageDataset, TestImageDataset
import data.saver as saver


def run_crossval(ds_tools, img_resize, batch_size, epochs, threshold, sample_size, n_fold):
    threads = cpu_count()
    use_cuda = torch.cuda.is_available()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tb_viz_cb = TensorboardVisualizerCallback(os.path.join(script_dir, '../../logs'))
    kf = KFold(n_splits=n_fold, shuffle=True)

    # Define our nn architecture
    net = unet.UNet1024((3, *img_resize))
    classifier = None

    # Get the path to the files for the neural net
    # We don't want to split train/valid for crossval
    full_x_train, full_y_train, _, _ = ds_tools.get_train_files(sample_size=sample_size, validation_size=0)
    full_x_test = ds_tools.get_test_files(sample_size)

    for train_indexes, valid_indexes in kf.split(full_x_train):
        X_train = full_x_train[train_indexes]
        y_train = full_y_train[train_indexes]
        X_valid = full_x_train[valid_indexes]
        y_valid = full_y_train[valid_indexes]

        # Get the original images size (assuming they are all the same size)
        origin_img_size = ds_tools.get_image_size(X_train[0])

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

        classifier = nn.classifier.CarvanaClassifier(net, train_loader, valid_loader)

        # Launch training
        classifier.train(epochs, callbacks=[tb_viz_cb])

    test_ds = TestImageDataset(full_x_test, img_resize)
    test_loader = DataLoader(test_ds, batch_size,
                             sampler=SequentialSampler(test_ds),
                             num_workers=threads,
                             pin_memory=use_cuda)

    # Predict & save
    output_file = os.path.join(script_dir, '../../output/submit.csv.gz')
    classifier.predict(test_loader, to_file=output_file,
                       t_fnc=saver.get_prediction_transformer,
                       fnc_args=[origin_img_size, 0.5])  # The get_prediction_transformer arguments
    print("Predictions wrote in {} file".format(output_file))
