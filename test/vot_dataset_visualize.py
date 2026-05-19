from pathlib import Path
import cv2
import numpy as np


def polygon_to_bbox(poly):
    """
    poly: [x1,y1,x2,y2,x3,y3,x4,y4]
    returns: [x_center, y_center, width, height]
    """
    xs = poly[0::2]
    ys = poly[1::2]

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    w = x2 - x1
    h = y2 - y1
    x_center = x1 + w / 2
    y_center = y1 + h / 2

    return [x_center, y_center, w, h]


def bbox_to_corners(box):
    x, y, w, h = box
    return [
        int(round(x - w / 2)),
        int(round(y - h / 2)),
        int(round(x + w / 2)),
        int(round(y + h / 2)),
    ]


def read_groundtruth(path):
    boxes = []

    with open(path, "r") as f:
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
                raise ValueError(f"Unexpected GT format: {line}")

            boxes.append(box)

    return boxes


def find_frames(sequence_dir):
    image_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    frames = []

    for ext in image_exts:
        frames.extend(sequence_dir.glob(ext))

    return sorted(frames)


def draw_example(frame, box, label=None):
    vis = frame.copy()

    x1, y1, x2, y2 = bbox_to_corners(box)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if label is not None:
        cv2.putText(
            vis,
            str(label),
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return vis


def visualize_sequence(sequence_dir, output_dir=None, stride=20):
    sequence_dir = Path(sequence_dir)

    gt_path = sequence_dir / "groundtruth.txt"
    boxes = read_groundtruth(gt_path)
    frames = find_frames(sequence_dir)

    if len(frames) == 0:
        raise RuntimeError(f"No image frames found in {sequence_dir}")

    if len(frames) != len(boxes):
        print(f"Warning: {len(frames)} frames, {len(boxes)} ground-truth boxes")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(frames), len(boxes))

    for i in range(0, n, stride):
        frame = cv2.imread(str(frames[i]))
        print(f"Visualizing {frames[i]} with GT box {boxes[i]}")
        if frame is None:
            print(f"Could not read {frames[i]}")
            continue

        vis = draw_example(frame, boxes[i], label=f"frame {i}")

        if output_dir is not None:
            out_path = output_dir / f"{sequence_dir.name}_{i:04d}.jpg"
            cv2.imwrite(str(out_path), vis)
        else:
            cv2.imshow(sequence_dir.name, vis)
            key = cv2.waitKey(0)
            if key == 27:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_sequence(
        sequence_dir="data/VOT2013/bicycle",
        output_dir="debug_vot_examples",
        stride=20,
    )

    # Output images will be saved in debug_vot_examples/ with names like bicycle_0000.jpg, bicycle_0020.jpg, etc.