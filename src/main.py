from data.dataset import DatasetTools
import numpy as np
import nn.crossval as crossval
import nn.classifier
import nn.unet as unet
import utils


def main():
    # Hyperparameters
    img_resize = (128, 128)  # (1024, 1024)
    batch_size = 32
    epochs = 2
    threshold = 0.5
    n_fold = 3

    # Put None to work on full dataset
    sample_size = None  # 0.1

    # Download the datasets
    ds_tools = DatasetTools()
    ds_tools.download_dataset()

    # Calculate epoch per fold for cross validation
    epochs_per_fold = np.maximum(1, np.round(epochs / n_fold).astype(int))

    # Define our nn architecture
    net = unet.UNet128((3, *img_resize))
    #net = unet.UNet1024((3, *img_resize))
    classifier = nn.classifier.CarvanaClassifier(net, epochs_per_fold * n_fold)

    # Clear output dirs
    utils.clear_output_dirs()

    crossval.run_crossval(classifier, ds_tools, img_resize, batch_size, epochs_per_fold, threshold, sample_size, n_fold)


if __name__ == "__main__":
    main()
