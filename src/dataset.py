from kaggle_data.downloader import KaggleDataDownloader
import os


def download_dataset():
    competition_name = "carvana-image-masking-challenge"

    train = "train.zip"
    test = "test.zip"
    train_masks_csv = "train_masks.csv.zip"
    train_masks = "train_masks.zip"
    destination_path = "../input/"
    is_datasets_present = True

    # If the folders already exists then the files may already be extracted
    # This is a bit hacky but it's sufficient for our needs
    datasets_path = ["../input/train", "../input/test", "../input/train_masks.csv", "../input/train_masks"]
    for dir_path in datasets_path:
        if not os.path.exists(dir_path):
            is_datasets_present = False

    if not is_datasets_present:
        # Put your Kaggle user name and password in a $KAGGLE_USER and $KAGGLE_PASSWD env vars respectively
        downloader = KaggleDataDownloader(os.getenv("KAGGLE_USER"), os.getenv("KAGGLE_PASSWD"), competition_name)

        output_path = downloader.download_dataset(train, destination_path)
        downloader.decompress(output_path, destination_path)
        os.remove(output_path)

        output_path = downloader.download_dataset(test, destination_path)
        downloader.decompress(output_path, destination_path)
        os.remove(output_path)

        output_path = downloader.download_dataset(train_masks_csv, destination_path)
        downloader.decompress(output_path, destination_path)
        os.remove(output_path)

        output_path = downloader.download_dataset(train_masks, destination_path)
        downloader.decompress(output_path, destination_path)
        os.remove(output_path)
    else:
        print("All datasets are present.")