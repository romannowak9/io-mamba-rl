import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pytorch_lightning as pl
import torchvision.transforms as T

from utils.afo_download import download_afo_dataset
from utils.helpers import sort_key_from_filename
from utils.afo_info import CLASS_NAME_BY_ID


class AFOTrackingDataset(Dataset):
    def __init__(self, data_dir, split="train", img_size=(1080, 1920), transforms=None, n_classes=6, sequence_length=1):
        '''
        img_size: Tuple[int] - (h, w) shape for resize of all frames
        n_classes: int - Number of classes. Possible values: 1, 2, 6
        transforms: torchvision.transforms to apply to image after already implemented ToTensor() and Resize()
        sequence_length: number of subsequent frames in one item

        Output:
        - frames: torch.tensor [T, CH, H, W], where T is sequence_length
        - targets: List[Dict[torch.tensor]] - dict keys: 'boxes', 'labels', 'image_id'
        '''
        self.root = Path(data_dir) / "afo"
        self.transforms = transforms
        self.sequence_length = sequence_length

        split_map = {"train": "train", "valid": "validation", "val": "validation", "test": "test"}
        self.split = split_map[split]

        self.images_dir = self.root / self.split / "img"
        self.labels_dir = self.root / self.split / "ann"

        # Obsługujemy jpg i png
        self.images = sorted(list(self.images_dir.glob("*.jpg")),
                             key=lambda x : sort_key_from_filename(x))

        if len(self.images) < self.sequence_length:
            raise RuntimeError(f"Not enough images in {self.images_dir} for sequence length {self.sequence_length}")
        
        self.n_classes = n_classes
        self.img_size = img_size

        match n_classes:
            case 1:
                self.valid_classes = {
                    "object",
                }
            case 2:
                self.valid_classes = {
                    "small_obj",
                    "large_obj",
                }
            case 6:
                self.valid_classes = {
                    "human",
                    "wind/sup-board",
                    "kayak",
                    "boat",
                    "bouy",
                    "sailboat",
                }
            case _:
                raise ValueError(f"Not supported number of classes: {n_classes}!")
                
    def __len__(self):
        return max(len(self.images) - self.sequence_length + 1, 0)

    def _load_labels(self, label_path):
        boxes, labels = [], []
        if not label_path.exists():
            return torch.zeros((0,4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        data = json.load(open(label_path))
        for obj in data.get("objects", []):
            exterior = obj.get("points", {}).get("exterior", None)
            x1, y1 = exterior[0]
            x2, y2 = exterior[1]
            boxes.append([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)])
            labels.append(obj.get("classId", [torch.nan]))

        # Scale boxes
        if boxes:
            scale_x = self.img_size[1] / data["size"]["width"]
            scale_y = self.img_size[0] / data["size"]["height"]
            boxes = [[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y] for x1, y1, x2, y2 in boxes]

        filtered = [(box, label) for box, label in zip(boxes, labels) if CLASS_NAME_BY_ID[label] in self.valid_classes]

        if filtered:
            boxes, labels = zip(*filtered)
            return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int32)
        else:
            return torch.zeros((0,4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int32)

    def __getitem__(self, idx):
        frames, targets = [], []

        for i in range(self.sequence_length):
            img_path = self.images[idx + i]
            label_path = self.labels_dir / f"{img_path.stem}.jpg.json"
            if not label_path.exists():
                label_path = self.labels_dir / f"{img_path.stem}.json"

            image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            boxes, labels = self._load_labels(label_path)

            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor(idx + i)
            }

            # Transforms
            if not isinstance(image, torch.Tensor):
                image = T.ToTensor()(image)
            image = T.Resize(self.img_size)(image)

            if self.transforms:
                image = self.transforms(image)

            frames.append(image)
            targets.append(target)

        frames = torch.stack(frames, dim=0)
        return frames, targets


def tracking_collate(batch):
    frames_batch, targets_batch = [], []
    for frames, targets in batch:
        frames_batch.append(frames)
        targets_batch.append(targets)
    return torch.stack(frames_batch, dim=0), targets_batch


class AFOTrackingDataModule(pl.LightningDataModule):
    """
    Parameters:
    - n_classes: int - Number of classes. Possible values: 1, 2, 6
    - batch_size
    - img_size: Tuple[int] - (h, w) shape for resize of all frames
    - sequence_length: number of subsequent frames in one item
    - num_workers
    - train_transforms: torchvision.transforms to apply to image after already implemented ToTensor() and Resize() for train dataset
    - test_transforms: torchvision.transforms to apply to image after already implemented ToTensor() and Resize() for test and validation dataset

    Output:
    - frames: Tensor(B, T, C, H, W)
    - targets: List[B][T]{boxes, labels, image_id}
    """
    def __init__(self, data_dir="data/afo", n_classes=6, batch_size=8, img_size=(1080, 1920),
                 sequence_length=1, num_workers=2, train_transforms=None, test_transforms=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.sequence_length = sequence_length

    def prepare_data(self):
        download_afo_dataset(self.data_dir.parent)

    def setup(self, stage=None):
        self.train_dataset = AFOTrackingDataset(
            self.data_dir, split="train", transforms=self.train_transforms, n_classes=self.n_classes, img_size=self.img_size, sequence_length=self.sequence_length
        )
        self.val_dataset = AFOTrackingDataset(
            self.data_dir, split="valid", transforms=self.test_transforms, n_classes=self.n_classes, img_size=self.img_size, sequence_length=self.sequence_length
        )
        self.test_dataset = AFOTrackingDataset(
            self.data_dir, split="test", transforms=self.test_transforms, n_classes=self.n_classes, img_size=self.img_size, sequence_length=self.sequence_length
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers>0, collate_fn=tracking_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers>0, collate_fn=tracking_collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers>0, collate_fn=tracking_collate
        )
    