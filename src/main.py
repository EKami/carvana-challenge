from data.dataset import DatasetTools
import nn.crossval as crossval


def main():
    # Hyperparameters
    img_resize = (1024, 1024)
    batch_size = 2
    epochs = 50
    threshold = 0.5
    n_fold = 5

    # Put None to work on full dataset
    sample_size = None  # 0.1

    # Download the datasets
    ds_tools = DatasetTools()
    ds_tools.download_dataset()

    crossval.run_crossval(ds_tools, img_resize, batch_size, epochs, threshold, sample_size, n_fold)


if __name__ == "__main__":
    main()
