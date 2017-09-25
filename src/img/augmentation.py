import cv2
import numpy as np


def random_shift_scale_rotate(image, angle, scale, aspect, shift_dx, shift_dy,
                              borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        if len(image.shape) == 3:  # Img or mask
            height, width, channels = image.shape
        else:
            height, width = image.shape

        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(shift_dx * width)
        dy = round(shift_dy * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode, borderValue=(0, 0, 0, 0))
    return image


def random_horizontal_flip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def augment_img(img, mask):
    rotate_limit = (-45, 45)
    aspect_limit = (0, 0)
    scale_limit = (-0.1, 0.1)
    shift_limit = (-0.0625, 0.0625)
    shift_dx = np.random.uniform(shift_limit[0], shift_limit[1])
    shift_dy = np.random.uniform(shift_limit[0], shift_limit[1])
    angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
    scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
    aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])

    img = random_shift_scale_rotate(img, angle, scale, aspect, shift_dx, shift_dy)
    mask = random_shift_scale_rotate(mask, angle, scale, aspect, shift_dx, shift_dy)

    img, mask = random_horizontal_flip(img, mask)
    return img, mask
