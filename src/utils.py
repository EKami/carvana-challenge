import numpy as np


# TODO remove numpy usage and use Pytorch on the GPU
def rle(img):
    """
    Encore an image to the Run-length encoding format for
    the final csv file
        Ex:
            mask = np.array(Image.open('../input/train_masks/00087a6bd4dc_01_mask.gif'), dtype=np.uint8)
            mask_rle = rle(mask)
            print(mask_rle)
    :param img: the image as a numpy array
    :return: the mask rle
    """
    flat_img = img.flatten()
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix

    return starts_ix, lengths


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

