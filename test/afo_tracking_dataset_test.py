# To run:
# from repo root:
# python3 -m test.afo_tracking_dataset_test

import torch
import torchvision.transforms as T

from modules.afo_tracking_datamodule import AFOTrackingDataset
from utils.afo_visualize import draw_sequence_bboxes
from utils.afo_download import download_afo_dataset
from modules.transforms import AddGaussianNoise


def dataset_test(transforms=None):
    data_dir = "data/afo"
    download_afo_dataset(data_dir)

    sequence_length = 2
    dataset = AFOTrackingDataset(
        data_dir,
        split="train",
        sequence_length=sequence_length,
        transforms=transforms
    )

    frames, targets = dataset[0]

    print("Frames shape with augmentation:", frames.shape)
    print("Number of targets per frame:", [len(t["boxes"]) for t in targets])

    draw_sequence_bboxes(frames, targets)


if __name__ == '__main__':

    transforms = T.Compose([
        T.ToTensor(),                     # HWC [0,255] -> C,H,W [0,1]
        T.RandomHorizontalFlip(p=0.5),    # losowy flip
        T.RandomRotation(degrees=15),     # losowy obrót
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        AddGaussianNoise(std=0.05),       # nasz własny szum
    ])

    dataset_test(transforms=None)