import torch.utils.data as data
from kaggle_data.downloader import KaggleDataDownloader
from sklearn.model_selection import train_test_split
import img.transformer as transformer
from PIL import Image
import numpy as np
import os


class DatasetTools:
    def __init__(self):
        """
            A tool used to automatically download, check, split and get
            relevant information on the dataset
        """
        self.train_data = None
        self.test_data = None
        self.train_masks_data = None
        self.train_files = None
        self.test_files = None
        self.train_masks_files = None
        self.train_ids = None
        self.masks_ids = None
        self.test_ids = None

    def download_dataset(self, hq_files=True):
        """
        Downloads the dataset and return the input paths
        Args:
            hq_files (bool): Whether to download the hq files or not

        Returns:
            list: [train_data, test_data, metadata_csv, train_masks_csv, train_masks_data]

        """
        competition_name = "carvana-image-masking-challenge"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        destination_path = os.path.join(script_dir, '../../input/')
        prefix = ""
        if hq_files:
            prefix = "_hq"
        files = ["train" + prefix + ".zip", "test" + prefix + ".zip", "metadata.csv.zip",
                 "train_masks.csv.zip", "train_masks.zip"]
        datasets_path = [destination_path + "train" + prefix, destination_path + "test" + prefix,
                         destination_path + "metadata.csv", destination_path + "train_masks.csv",
                         destination_path + "train_masks"]
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
        self.train_files = sorted(os.listdir(self.train_data))
        self.test_files = sorted(os.listdir(self.test_data))
        self.train_masks_files = sorted(os.listdir(self.train_masks_data))
        self.train_ids = list(set(t.split("_")[0] for t in self.train_files))
        self.masks_ids = list(set(t.split("_")[0] for t in self.train_masks_files))
        self.test_ids = list(set(t.split("_")[0] for t in self.test_files))
        return datasets_path

    def get_car_image_files(self, car_image_id, test_file=False, get_mask=False):
        if get_mask:
            if car_image_id in self.masks_ids:
                return [self.train_masks_data + "/" + s for s in self.train_masks_files if car_image_id in s]
            else:
                raise Exception("No mask with this ID found")
        elif test_file:
            if car_image_id in self.test_ids:
                return [self.test_data + "/" + s for s in self.test_files if car_image_id in s]
        else:
            if car_image_id in self.train_ids:
                return [self.train_data + "/" + s for s in self.train_files if car_image_id in s]
        raise Exception("No image with this ID found")

    def get_image_matrix(self, image_path):
        img = Image.open(image_path)
        return np.asarray(img, dtype=np.uint8)

    def get_image_size(self, image):
        img = Image.open(image)
        return img.size

    def get_train_files(self, validation_size=0.2, sample_size=None):
        """

        Args:
            validation_size (int):
                 Value between 0 and 1
            sample_size (float, None):
                Value between 0 and 1 or None.
                Whether you want to have a sample of your dataset.

        Returns:
            list :
                Returns the dataset in the form:
                [train_data, train_masks_data, valid_data, valid_masks_data]
        """
        train_ids = self.train_ids

        # Each id has 16 images but well...
        if sample_size:
            rnd = np.random.choice(self.train_ids, int(len(self.train_ids) * sample_size))
            train_ids = rnd.ravel()

        if validation_size:
            ids_train_split, ids_valid_split = train_test_split(train_ids, test_size=validation_size)
        else:
            ids_train_split = train_ids
            ids_valid_split = []

        train_ret = []
        train_masks_ret = []
        valid_ret = []
        valid_masks_ret = []

        for id in ids_train_split:
            train_ret.append(self.get_car_image_files(id))
            train_masks_ret.append(self.get_car_image_files(id, get_mask=True))

        for id in ids_valid_split:
            valid_ret.append(self.get_car_image_files(id))
            valid_masks_ret.append(self.get_car_image_files(id, get_mask=True))

        return [np.array(train_ret).ravel(), np.array(train_masks_ret).ravel(),
                np.array(valid_ret).ravel(), np.array(valid_masks_ret).ravel()]

    def get_test_files(self, sample_size):
        test_files = self.test_files

        if sample_size:
            rnd = np.random.choice(self.test_files, int(len(self.test_files) * sample_size))
            test_files = rnd.ravel()

        ret = [None] * len(test_files)
        for i, file in enumerate(test_files):
            ret[i] = self.test_data + "/" + file

        return np.array(ret)


# Reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L66
class TrainImageDataset(data.Dataset):
    def __init__(self, X_data, y_data=None, img_resize=(128, 128),
                 X_transform=None, y_transform=None, threshold=0.5):
        """
            A dataset loader taking images paths as argument and return
            as them as tensors from getitem()

            Args:
                threshold (float): The threshold used to consider the mask present or not
                X_data (list): List of paths to the training images
                y_data (list, optional): List of paths to the target images
                img_resize (tuple): Tuple containing the new size of the images
                X_transform (callable, optional): A function/transform that takes in 2 numpy arrays.
                    Assumes X_data and y_data are not None.
                    (train_img, mask_img) and returns a transformed version with the same signature
                y_transform (callable, optional): A function/transform that takes in 2 numpy arrays.
                    Assumes X_data and y_data are not None.
                    (train_img, mask_img) and returns a transformed version with the same signature
        """
        self.threshold = threshold
        self.X_train = X_data
        self.y_train_masks = y_data
        self.img_resize = img_resize
        self.y_transform = y_transform
        self.X_transform = X_transform

    def __getitem__(self, index):
        """
            Args:
                index (int): Index
            Returns:
                tuple: (image, target) where target is class_index of the target class.
        """
        img = Image.open(self.X_train[index])
        #img = transformer.center_cropping_resize(img, self.img_resize)
        img = np.asarray(img.convert("RGB"), dtype=np.float32)

        # Pillow reads gifs
        mask = Image.open(self.y_train_masks[index])
        #mask = transformer.center_cropping_resize(mask, self.img_resize)
        mask = np.asarray(mask.convert("L"), dtype=np.float32)  # GreyScale

        if self.X_transform:
            img, mask = self.X_transform(img, mask)

        if self.y_transform:
            img, mask = self.y_transform(img, mask)

        img = transformer.image_to_tensor(img)
        mask = transformer.mask_to_tensor(mask, self.threshold)
        return img, mask

    def __len__(self):
        assert len(self.X_train) == len(self.y_train_masks)
        return len(self.X_train)


class TestImageDataset(data.Dataset):
    def __init__(self, X_data, img_resize=(128, 128)):
        """
            A dataset loader taking images paths as argument and return
            as them as tensors from getitem()
            Args:
                X_data (list): List of paths to the training images
                img_resize (tuple): Tuple containing the new size of the images
        """
        self.img_resize = img_resize
        self.X_train = X_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_path = self.X_train[index]
        img = Image.open(img_path)
        #img = transformer.center_cropping_resize(img, self.img_resize)
        img = np.asarray(img.convert("RGB"), dtype=np.float32)

        img = transformer.image_to_tensor(img)
        return img, img_path.split("/")[-1]

    def __len__(self):
        return len(self.X_train)
