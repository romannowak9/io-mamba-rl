# To run:
# from repo root:
# python3 -m test.afo_tracking_dataset_test

import torchvision.transforms as T

from modules.afo_tracking_datamodule import AFOTrackingDataset
from utils.afo_visualize import draw_sequence_bboxes
from utils.afo_download import download_afo_dataset
from modules.transforms import AddGaussianNoise

def dataset_test(transforms=None):
    data_dir = "data"
    download_afo_dataset(data_dir)

    sequence_length = 1
    dataset = AFOTrackingDataset(
        data_dir,
        split="train",
        sequence_length=sequence_length,
        n_classes=6,
        transforms=transforms
    )

    print(f"Number of images in train: {len(dataset.images)}")
    print(f"Sequence length: {dataset.sequence_length}")

    frames, targets = dataset[0]

    print("Frames shape:", frames.shape)
    print("Number of targets per frame:", [len(t["boxes"]) for t in targets])

    draw_sequence_bboxes(frames, targets, figsize=(120, 40))


if __name__ == '__main__':
    transforms = T.Compose([
        T.ToTensor(),                     # HWC [0,255] -> C,H,W [0,1]
        # T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # T.Resize([512, 512]),
        # AddGaussianNoise(std=0.1),       # nasz własny szum
    ])

    dataset_test(transforms=transforms)