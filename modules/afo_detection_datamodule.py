from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from utils.afo_download import download_afo_dataset


class AFODetectionDataset(Dataset):

    def __init__(self, root, split="train", transforms=None):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        self.images_dir = self.root / split / "images"
        self.labels_dir = self.root / split / "labels"

        self.images = sorted(self.images_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.images)

    def _load_labels(self, label_path, img_w, img_h):
        '''
        Returns:
        - boxes : torch.tensor of [x1,y1,x2,y2]
        - labels : torch.tensor of class idxs values
        '''
        boxes = []
        labels = []

        if not label_path.exists():
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        with open(label_path) as f:
            lines = f.readlines()

        for line in lines:
            cls, xc, yc, w, h = map(float, line.split())

            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h

            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls))

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        boxes, labels = self._load_labels(label_path, w, h)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            image = self.transforms(image)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, target


def detection_collate(batch):
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    return images, targets


class AFODetectionDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir="data/afo",
        batch_size=8,
        num_workers=8,
        transforms=None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms

    def prepare_data(self):
        download_afo_dataset(self.data_dir)

    def setup(self, stage=None):
        self.train_dataset = AFODetectionDataset(self.data_dir, split="train", transforms=self.transforms)
        self.val_dataset = AFODetectionDataset(self.data_dir, split="valid", transforms=self.transforms)
        self.test_dataset = AFODetectionDataset(self.data_dir, split="test", transforms=self.transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=detection_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=detection_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=detection_collate,
        )