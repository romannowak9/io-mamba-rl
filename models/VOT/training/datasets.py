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


def polygon_to_bbox(poly: Sequence[float]) -> list[float]:
    xs = poly[0::2]
    ys = poly[1::2]

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    w = x2 - x1
    h = y2 - y1

    return [x1 + w / 2, y1 + h / 2, w, h]


def read_groundtruth(gt_path: Path) -> list[list[float]]:
    boxes = []

    with gt_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            values = [float(v) for v in line.replace(",", " ").split()]

            if len(values) == 8:
                box = polygon_to_bbox(values)
            elif len(values) == 4:
                x, y, w, h = values
                box = [x + w / 2, y + h / 2, w, h]
            else:
                raise ValueError(f"Unexpected groundtruth format in {gt_path}: {line}")

            boxes.append(box)

    return boxes


def find_frames(sequence_dir: Path) -> list[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    frames = []

    for ext in exts:
        frames.extend(sequence_dir.glob(ext))

    return sorted(frames)


def sample_noisy_box(gt_box: Sequence[float]) -> list[float]:
    x, y, w, h = map(float, gt_box)

    noise = np.random.normal(
        loc=0.0,
        scale=[0.25 * w, 0.25 * h, 0.1 * w, 0.1 * h],
    )

    sampled = [
        x + noise[0],
        y + noise[1],
        max(1.0, w + noise[2]),
        max(1.0, h + noise[3]),
    ]

    return sampled


def assign_action_label(
    sample_box: Sequence[float],
    gt_box: Sequence[float],
    action_set: Sequence[str],
    img_width: int,
    img_height: int,
    positive_iou_threshold: float = 0.7,
) -> int:
    current_iou = iou(sample_box, gt_box)

    if current_iou > positive_iou_threshold and "stop" in action_set:
        return action_to_index("stop", action_set)

    best_action = None
    best_iou = -1.0

    for action in action_set:
        if action == "stop":
            continue

        candidate_box = transition_box(
            sample_box,
            action,
            img_width=img_width,
            img_height=img_height,
        )

        candidate_iou = iou(candidate_box, gt_box)

        if candidate_iou > best_iou:
            best_iou = candidate_iou
            best_action = action

    if best_action is None:
        raise RuntimeError("Could not assign action label. Check action_set.")

    return action_to_index(best_action, action_set)

def list_vot_sequences(root_dirs):
    sequences = []

    for root_dir in [Path(p) for p in root_dirs]:
        for sequence_dir in sorted(root_dir.iterdir()):
            if sequence_dir.is_dir() and (sequence_dir / "groundtruth.txt").exists():
                sequences.append(sequence_dir.name)

    return sorted(set(sequences))

def sample_box_with_target_class(
    gt_box,
    img_width,
    img_height,
    positive_iou_threshold=0.7,
    target_positive=True,
    max_attempts=50,
):
    fallback_box = None

    for _ in range(max_attempts):
        sample_box = sample_noisy_box(gt_box)
        sample_box = clip_box(sample_box, img_width, img_height)

        sample_iou = iou(sample_box, gt_box)
        is_positive = sample_iou > positive_iou_threshold

        if fallback_box is None:
            fallback_box = sample_box

        if is_positive == target_positive:
            return sample_box

    return fallback_box

class VOTSupervisedDataset(Dataset):
    """
    Supervised ADNet-style dataset for VOT2013/VOT2014/VOT2015.

    Returns:
        patch: FloatTensor [3, H, W]
        history: FloatTensor [history_length * num_actions]
        action_label: LongTensor scalar
        class_label: LongTensor scalar
    """

    def __init__(
        self,
        root_dirs: Iterable[str | Path],
        action_set: Sequence[str],
        samples_per_frame: int = 10,
        input_size: tuple[int, int] = (112, 112),
        history_length: int = 10,
        positive_iou_threshold: float = 0.7,
        target_positive_ratio: Optional[float] = 0.4,
        max_resample_attempts: Optional[int] = 50,
        transform: Optional[object] = None,
        sequence_filter: Optional[set[str]] = None,
    ):
        self.root_dirs = [Path(p) for p in root_dirs]
        self.action_set = list(action_set)
        self.samples_per_frame = samples_per_frame
        self.input_size = input_size
        self.history_length = history_length
        self.positive_iou_threshold = positive_iou_threshold
        self.transform = transform
        self.sequence_filter = sequence_filter

        self.num_actions = len(self.action_set)
        self.history_dim = self.history_length * self.num_actions

        self.items = self._index_sequences()

        self.target_positive_ratio = target_positive_ratio
        self.max_resample_attempts = max_resample_attempts

        if len(self.items) == 0:
            raise RuntimeError(f"No VOT frames found in: {self.root_dirs}")

    def _index_sequences(self):
        items = []

        for root_dir in self.root_dirs:
            if not root_dir.exists():
                raise FileNotFoundError(root_dir)

            for sequence_dir in sorted(root_dir.iterdir()):
                if not sequence_dir.is_dir():
                    continue

                gt_path = sequence_dir / "groundtruth.txt"
                if not gt_path.exists():
                    continue

                if self.sequence_filter is not None and sequence_dir.name not in self.sequence_filter:
                    continue

                frames = find_frames(sequence_dir)
                gt_boxes = read_groundtruth(gt_path)

                n = min(len(frames), len(gt_boxes))

                if n == 0:
                    continue

                for frame_idx in range(n):
                    for sample_idx in range(self.samples_per_frame):
                        items.append(
                            {
                                "frame_path": frames[frame_idx],
                                "gt_box": gt_boxes[frame_idx],
                                "sequence": sequence_dir.name,
                                "frame_idx": frame_idx,
                                "sample_idx": sample_idx,
                            }
                        )

        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        image = cv2.imread(str(item["frame_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {item['frame_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_height, img_width = image.shape[:2]

        gt_box = clip_box(item["gt_box"], img_width, img_height)
        target_positive = np.random.rand() < self.target_positive_ratio

        sample_box = sample_box_with_target_class(
            gt_box=gt_box,
            img_width=img_width,
            img_height=img_height,
            positive_iou_threshold=self.positive_iou_threshold,
            target_positive=target_positive,
            max_attempts=self.max_resample_attempts,
)

        patch = crop_patch(image, sample_box)

        if patch is None or patch.size == 0:
            patch = np.zeros(
                (self.input_size[1], self.input_size[0], 3),
                dtype=np.uint8,
            )
        else:
            patch = resize_patch(patch, self.input_size)

        if self.transform is not None:
            patch = self.transform(patch)
        else:
            patch = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0

        action_label = assign_action_label(
            sample_box=sample_box,
            gt_box=gt_box,
            action_set=self.action_set,
            img_width=img_width,
            img_height=img_height,
            positive_iou_threshold=self.positive_iou_threshold,
        )

        class_label = int(iou(sample_box, gt_box) > self.positive_iou_threshold)

        history = torch.zeros(self.history_dim, dtype=torch.float32)

        return {
            "patch": patch,
            "history": history,
            "action_label": torch.tensor(action_label, dtype=torch.long),
            "class_label": torch.tensor(class_label, dtype=torch.long),
            "sample_box": torch.tensor(sample_box, dtype=torch.float32),
            "gt_box": torch.tensor(gt_box, dtype=torch.float32),
            "frame_path": str(item["frame_path"]),
            "sequence": item["sequence"],
            "frame_idx": item["frame_idx"],
        }

