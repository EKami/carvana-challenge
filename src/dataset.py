from kaggle_data.downloader import KaggleDataDownloader
import pandas as pd
import os


class DatasetHandler:

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.train_masks_data = None
        self.train_files = None
        self.test_files = None
        self.train_masks_files = None
        self.train_ids = None
        self.masks_ids = None
        self.test_ids = None

    def download_dataset(self):
        """
        Downloads the dataset and return the input paths
        :return: [train_data, test_data, metadata_csv, train_masks_csv, train_masks_data]
        """
        competition_name = "carvana-image-masking-challenge"

        destination_path = "../input/"
        files = ["train.zip", "test.zip", "metadata.csv.zip", "train_masks.csv.zip", "train_masks.zip"]
        datasets_path = ["../input/train", "../input/test", "../input/metadata.csv", "../input/train_masks.csv",
                        "../input/train_masks"]
        is_datasets_present = True

        # If the folders already exists then the files may already be extracted
        # This is a bit hacky but it's sufficient for our needs
        for dir_path in datasets_path:
            if not os.path.exists(dir_path):
                is_datasets_present = False

        if not is_datasets_present:
            # Put your Kaggle user name and password in a $KAGGLE_USER and $KAGGLE_PASSWD env vars respectively
            downloader = KaggleDataDownloader(os.getenv("KAGGLE_USER"), os.getenv("KAGGLE_PASSWD"), competition_name)

            for file in files:
                output_path = downloader.download_dataset(file, destination_path)
                downloader.decompress(output_path, destination_path)
                os.remove(output_path)
        else:
            print("All datasets are present.")

        self.train_data = datasets_path[0]
        self.test_data = datasets_path[1]
        self.train_masks_data = datasets_path[4]
        self.train_files = os.listdir(self.train_data)
        self.test_files = os.listdir(self.test_data)
        self.train_masks_files = os.listdir(self.train_masks_data)
        self.train_ids = list(set(t.split("_")[0] for t in self.train_files))
        self.masks_ids = list(set(t.split("_")[0] for t in self.train_masks_files))
        self.test_ids = list(set(t.split("_")[0] for t in self.test_files))
        return datasets_path

    def get_car_image_files(self, car_image_id, get_mask=False):

        if get_mask:
            if car_image_id in self.masks_ids:
                return [self.train_masks_data + "/" + s for s in self.train_masks_files if car_image_id in s]
            else:
                raise Exception("No mask with this ID found")
        elif car_image_id in self.train_ids:
            return [self.train_data + "/" + s for s in self.train_files if car_image_id in s]
        elif car_image_id in self.test_ids:
            return [self.test_data + "/" + s for s in self.test_files if car_image_id in s]
        raise Exception("No image with this ID found")

    def split_train_valid(self):
        pass
