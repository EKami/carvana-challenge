import cv2
import os
import numpy as np
from tensorboardX import SummaryWriter


class TensorboardVisualizerCallback:
    def __init__(self, path_to_files):
        """
            Callback intended to be executed at each epoch
            of the training which goal is to display the result
            of the last validation batch in Tensorboard
        Args:
            path_to_files:
        """
        self.writer = SummaryWriter(path_to_files)

    def _draw_contour(self, image, mask, color=(0, 255, 0), thickness=1):
        threshold = 127
        ret, thresh = cv2.threshold(mask, threshold, 255, 0)
        ret = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = ret[1]
        cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)

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

        l = np.zeros((H * W), np.uint8)
        m = np.zeros((H * W), np.uint8)
        image1 = image.copy()
        if mask is not None:
            l = mask.reshape(-1)
            self._draw_contour(image1, mask, color=(0, 0, 255), thickness=1)

        a = (2 * l + m)
        miss = np.where(a == 2)[0]
        hit = np.where(a == 3)[0]
        fp = np.where(a == 1)[0]
        p[miss] = np.array([0, 0, 255])
        p[hit] = np.array([64, 64, 64])
        p[fp] = np.array([0, 255, 0])
        p = p.reshape(H, W, 3)

        results[:, 0:W] = image1
        results[:, W:2 * W] = p
        results[:, 2 * W:3 * W] = image  # image * α + mask * β + λ
        return results

    def __call__(self, *args, **kwargs):
        net = kwargs['net']
        epoch_id = kwargs['epoch_id']

        last_images, last_targets, last_preds = kwargs['last_val_batch']
        for i, (image, target_mask, pred_mask) in enumerate(zip(last_images, last_targets, last_preds)):

            image = image.data.float().cpu().numpy().astype(np.uint8)
            image = np.transpose(image)  # Invert c, h, w to h, w, c
            target_mask = target_mask.float().data.cpu().numpy().astype(np.uint8)
            pred_mask = pred_mask.float().data.cpu().numpy().astype(np.uint8)

            expected_result = self._get_mask_representation(image, target_mask)
            pred_result = self._get_mask_representation(image, pred_mask)
            self.writer.add_image('Expected-Image_'+str(i)+"-Epoch_"+str(epoch_id), expected_result, epoch_id)
            self.writer.add_image('Predicted-Image_' + str(i) + "-Epoch_" + str(epoch_id), pred_result, epoch_id)
        #self.writer.close()
