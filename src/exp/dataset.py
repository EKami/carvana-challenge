import torch.utils.data as data
import numpy as np
import bcolz


#
# /!\ Unfinished feature
# The data are not retrieved properly
# Usage in main.py:
#   train_ds = CacheDatasetWrapper(TrainImageDataset(X_train, y_train, img_resize, X_transform=aug.augment_img),
#                                  os.path.join(script_dir, '../output/cached_preprocessing.h5'))
class CacheDatasetWrapper(data.Dataset):
    def __init__(self, dataset, caching_file):
        """
            Wrapper that cache Dataset objects into a h5 file
            to avoid doing preprocessing across each epochs.
            Assumes the passed dataset returns its objects
            from __getitem__ as tuple of Pytorch Tensors
        Args:
            dataset (data.Dataset): The dataset to cache
            caching_file (str) : The path to the h5 file
        """
        self.caching_file = caching_file
        self.dataset = dataset
        self.cache_file = h5py.File(caching_file, "w")
        self.dsets = None
        self.dsets_size = 0
        self.dset_types = None

    def __getitem__(self, index):
        if not self.dsets:
            first_iter = self.dataset[index]
            # Now create the dataset with the type of the first preprocessed file
            self.dsets_size = len(first_iter)
            self.dsets = []
            self.dset_types = []
            # Create a dataset for each element of the tuple
            for i, file in enumerate(first_iter):
                self.dsets.append(self.cache_file.create_dataset("preprocessing_dataset_" + str(i),
                                                                 (len(self.dataset), *file.size()),
                                                                 dtype=file.numpy().dtype))
                self.dset_types.append(type(file))

        # TODO find a way to pass in first_iter here
        # TODO
        ret = [None] * self.dsets_size
        for i in range(self.dsets_size):
            curr_dset = self.dsets[i]
            curr_dset_type = self.dset_types[i]

            # TODO check if this method really works as expected
            if not np.any(curr_dset[index]):  # If the array is all 0, thus has no record
                preprocessed_obj = self.dataset[index][i]
                curr_dset[index] = preprocessed_obj.numpy()
                ret[i] = preprocessed_obj
            else:  # If the record has been found in cache
                # print("Data retrieved from cache!")
                ret[i] = curr_dset_type(curr_dset[index])  # cast file in the original type

        return ret

    def __len__(self):
        return len(self.dataset)
