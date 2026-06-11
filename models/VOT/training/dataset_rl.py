from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from models.VOT.utils.bbox import iou, clip_box
from models.VOT.utils.crops import crop_patch, resize_patch
from models.VOT.tracking.actions import (
    action_to_index,
    transition_box,
)



from models.VOT.training.datasets import read_groundtruth, find_frames


class VOTSequenceDataset(Dataset):
    """
    Sequence dataset for ADNet RL fine-tuning.

    Returns:
        frames: list[np.ndarray] of BGR images
        gt_boxes: FloatTensor [clip_length, 4]
        frame_paths: list[str]
        sequence: str
        sequence_id: str
        start_idx: int
    """

    def __init__(
        self,
        root_dirs: Iterable[str | Path],
        clip_length: int = 10,
        stride: int = 1,
        sequence_filter: Optional[set[str]] = None,
        skip_nan: bool = True,
        negative_sequence_filter: Optional[bool] = False,
    ):
        self.root_dirs = [Path(p) for p in root_dirs]
        self.clip_length = clip_length
        self.stride = stride
        self.sequence_filter = sequence_filter
        self.skip_nan = skip_nan
        self.negative_sequence_filter = negative_sequence_filter  # reverses sequence filter logic
        self.items = self._index_sequences()

        if len(self.items) == 0:
            raise RuntimeError(f"No valid RL sequences found in: {self.root_dirs}")

    def _is_valid_box(self, box):
        return np.all(np.isfinite(np.asarray(box, dtype=np.float32)))

    def _clip_is_valid(self, gt_boxes, start_idx):
        if not self.skip_nan:
            return True

        clip_boxes = gt_boxes[start_idx : start_idx + self.clip_length]
        return all(self._is_valid_box(box) for box in clip_boxes)

    def _index_sequences(self):
        items = []

        for root_dir in self.root_dirs:
            if not root_dir.exists():
                raise FileNotFoundError(root_dir)

            dataset_name = root_dir.name

            for sequence_dir in sorted(root_dir.iterdir()):
                if not sequence_dir.is_dir():
                    continue

                gt_path = sequence_dir / "groundtruth.txt"
                if not gt_path.exists():
                    continue

                sequence_id = f"{dataset_name}/{sequence_dir.name}"

                if not self.negative_sequence_filter:


                    if (
                        self.sequence_filter is not None
                        and sequence_id not in self.sequence_filter
                    ):
                        continue
                else:
                    if (self.sequence_filter is not None and sequence_id in self.sequence_filter):
                        continue

                frames = find_frames(sequence_dir)
                gt_boxes = read_groundtruth(gt_path)

                n = min(len(frames), len(gt_boxes))

                if n < self.clip_length:
                    continue

                for start_idx in range(0, n - self.clip_length + 1, self.stride):
                    if not self._clip_is_valid(gt_boxes, start_idx):
                        continue

                    items.append(
                        {
                            "sequence_dir": sequence_dir,
                            "sequence": sequence_dir.name,
                            "sequence_id": sequence_id,
                            "frames": frames[start_idx : start_idx + self.clip_length],
                            "gt_boxes": gt_boxes[start_idx : start_idx + self.clip_length],
                            "start_idx": start_idx,
                        }
                    )

        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        frames = []
        valid_boxes = []

        for frame_path, gt_box in zip(item["frames"], item["gt_boxes"]):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)

            if frame is None:
                raise RuntimeError(f"Could not read image: {frame_path}")

            img_h, img_w = frame.shape[:2]

            gt_box = clip_box(gt_box, img_w, img_h)

            frames.append(frame)
            valid_boxes.append(gt_box)

        gt_boxes = torch.tensor(valid_boxes, dtype=torch.float32)

        return {
            "frames": frames,
            "gt_boxes": gt_boxes,
            "frame_paths": [str(p) for p in item["frames"]],
            "sequence": item["sequence"],
            "sequence_id": item["sequence_id"],
            "start_idx": item["start_idx"],
        }


if __name__ == "__main__":
    dataset = VOTSequenceDataset(
        root_dirs = [
            "data/VOT2013",
            "data/VOT2014",
            "data/VOT2015",
            "data/VOT2016",
            "data/OTB/OTB50",
            "data/OTB/OTB100",
            "data/TrackingDataset"
        ],
        clip_length=10,
        stride=2,
        sequence_filter=None,
    )

    sample = dataset[0]

    print(len(sample["frames"]))        # 10
    print(sample["gt_boxes"].shape)     # torch.Size([10, 4])
    print(sample["sequence_id"])
    print(sample["start_idx"])