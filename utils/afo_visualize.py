import cv2
import matplotlib.pyplot as plt
import torch
import os

from utils.afo_info import CLASS_NAME_BY_ID

def add_labels_to_image(img: torch.Tensor, boxes, labels):
    """
    Rysuje bounding boxy i labelki na obrazie numpy (H,W,C)
    Args:
        img: torch.Tensor (C,H,W) w zakresie [0,1]
        boxes: tensor Nx4
        labels: tensor N
    Returns:
        img: numpy array uint8 z narysowanymi bboxami
    """
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype("uint8").copy()

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, str(CLASS_NAME_BY_ID[label.item()]), (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    return img


def draw_bboxes_on_image(image: torch.Tensor, targets: dict, figsize=(10, 8)):
    """
    Rysuje pojedynczy obraz z bboxami i labelami.
    """
    img = add_labels_to_image(image, targets["boxes"], targets["labels"])
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def draw_sequence_bboxes(frames: torch.Tensor, targets: list, out_path='out/img.jpg', figsize=(45, 15)):
    """
    Rysuje sekwencję klatek w subplotach.
    """
    T = frames.shape[0]
    plt.figure(figsize=figsize)

    for i in range(T):
        img = add_labels_to_image(frames[i], targets[i]["boxes"], targets[i]["labels"])
        plt.subplot(1, T, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Frame {i}")
    
    out_dir, out_name = os.path.split(out_path)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved sequence visualization to {out_path}")