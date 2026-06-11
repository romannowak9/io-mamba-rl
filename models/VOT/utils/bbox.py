#move_box()
#scale_box()
#clip_box()
#iou()

#original 11 actions
#left, right, up, down,
#large-left, large-right, large-up, large-down,
#scale-up, scale-down,
#stop

import numpy as np

#box assumed to be [x_center, y_center, width, height]

def scale_box(box, scale_factor):
    x, y, w, h = box
    return [x, y, w * scale_factor, h * scale_factor]

def alter_box(box, action_type, alpha=0.05):
    x, y, w, h = box
    dx = alpha * w
    dy = alpha * h

    if action_type == "left":
        x -= dx
    elif action_type == "right":
        x += dx
    elif action_type == "up":
        y -= dy
    elif action_type == "down":
        y += dy
    elif action_type == "large-left":
        x -= 2 * dx
    elif action_type == "large-right":
        x += 2 * dx
    elif action_type == "large-up":
        y -= 2 * dy
    elif action_type == "large-down":
        y += 2 * dy
    elif action_type == "scale-up":
        w *= 1.2
        h *= 1.2
    elif action_type == "scale-down":
        w *= 0.8
        h *= 0.8

    return [x, y, w, h]

def clip_box(box, img_width, img_height):
    x1, y1, x2, y2 = box_to_corners(box)

    x1 = np.clip(x1, 0, img_width - 1)
    y1 = np.clip(y1, 0, img_height - 1)
    x2 = np.clip(x2, 0, img_width - 1)
    y2 = np.clip(y2, 0, img_height - 1)

    if x2 <= x1:
        x2 = min(x1 + 1, img_width - 1)
    if y2 <= y1:
        y2 = min(y1 + 1, img_height - 1)

    return corners_to_box([x1, y1, x2, y2])

def box_to_corners(box):
    x_center, y_center, width, height = box
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return [x1, y1, x2, y2]

def corners_to_box(corners):
    x1, y1, x2, y2 = corners
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return [x_center, y_center, width, height]

def iou(boxA, boxB):
    boxA_corners = box_to_corners(boxA)
    boxB_corners = box_to_corners(boxB)

    xA = max(boxA_corners[0], boxB_corners[0])
    yA = max(boxA_corners[1], boxB_corners[1])
    xB = min(boxA_corners[2], boxB_corners[2])
    yB = min(boxA_corners[3], boxB_corners[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA_corners[2] - boxA_corners[0]) * (boxA_corners[3] - boxA_corners[1])
    boxBArea = (boxB_corners[2] - boxB_corners[0]) * (boxB_corners[3] - boxB_corners[1])

    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    
    iou_value = interArea / union
    return iou_value




