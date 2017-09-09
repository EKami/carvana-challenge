import cv2
import numpy as np
import scipy.misc as scipy
from tensorboardX import SummaryWriter


class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class TensorboardVisualizerCallback(Callback):
    def __init__(self, path_to_files):
        """
            Callback intended to be executed at each epoch
            of the training which goal is to display the result
            of the last validation batch in Tensorboard
        Args:
            path_to_files:
        """
        self.writer = SummaryWriter(path_to_files)

    def _apply_mask_overlay(self, image, mask, color=(0, 255, 0)):
        mask = np.dstack((mask, mask, mask)) * np.array(color)
        mask = mask.astype(np.uint8)
        return cv2.addWeighted(mask, 0.5, image, 0.5, 0.)  # image * α + mask * β + λ

    def _get_mask_representation(self, image, mask):
        """
         Given a mask and an image this method returns
         one image representing 3 patches of the same image.
         These patches represent:
            - The original image
            - The original mask
            - The mask applied to the original image
        Args:
            image (np.ndarray): The original image
            mask (np.ndarray): The predicted mask

        Returns (np.ndarray):
            An image of size (original_image_height, (original_image_width * 3))
            showing 3 patches of the original image
        """

        H, W, C = image.shape
        results = np.zeros((H, 3 * W, 3), np.uint8)
        p = np.zeros((H * W, 3), np.uint8)

        m = np.zeros((H * W), np.uint8)
        l = mask.reshape(-1)
        masked_img = self._apply_mask_overlay(image, mask)

        a = (2 * l + m)
        miss = np.where(a == 2)[0]
        hit = np.where(a == 3)[0]
        fp = np.where(a == 1)[0]
        p[miss] = np.array([0, 0, 255])
        p[hit] = np.array([64, 64, 64])
        p[fp] = np.array([0, 255, 0])
        p = p.reshape(H, W, 3)

        results[:, 0:W] = image
        results[:, W:2 * W] = p
        results[:, 2 * W:3 * W] = masked_img
        return results

    def __call__(self, *args, **kwargs):
        epoch_id = kwargs['epoch_id']

        last_images, last_targets, last_preds = kwargs['last_val_batch']
        for i, (image, target_mask, pred_mask) in enumerate(zip(last_images, last_targets, last_preds)):

            image = image.data.float().cpu().numpy().astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))  # Invert c, h, w to h, w, c
            target_mask = target_mask.float().data.cpu().numpy().astype(np.uint8)
            pred_mask = pred_mask.float().data.cpu().numpy().astype(np.uint8)
            if image.shape[0] > 256:  # We don't want the images on tensorboard to be too large
                image = scipy.imresize(image, (256, 256))
                target_mask = scipy.imresize(target_mask, (256, 256))
                pred_mask = scipy.imresize(pred_mask, (256, 256))

            expected_result = self._get_mask_representation(image, target_mask)
            pred_result = self._get_mask_representation(image, pred_mask)
            self.writer.add_image("Epoch_"+str(epoch_id)+'-Image_'+str(i+1)+'-Expected', expected_result, epoch_id)
            self.writer.add_image("Epoch_" + str(epoch_id)+'-Image_' + str(i+1)+'-Predicted', pred_result, epoch_id)
            if i == 1:  # 2 Images are sufficient
                break
        #self.writer.close()