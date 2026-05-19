# crop_patch()

# resize_path()

# pad_patch_if_needed()

import cv2
import numpy as np

def crop_patch(image, box):
    x_center, y_center, width, height = box
    x1 = int(round(x_center - width / 2))
    y1 = int(round(y_center - height / 2))
    x2 = int(round(x_center + width / 2))
    y2 = int(round(y_center + height / 2))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]

def resize_patch(patch, target_size):
    return cv2.resize(patch, target_size)

def pad_patch_if_needed(patch, target_size):
    #clip box first, then pad if needed
    h, w = patch.shape[:2]
    target_w, target_h = target_size
    pad_w = max(0, target_w - w)
    pad_h = max(0, target_h - h)
    if pad_w > 0 or pad_h > 0:
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    return patch


