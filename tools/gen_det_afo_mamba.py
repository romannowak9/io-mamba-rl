# Example usage:
# python tools/gen_det_afo_mamba.py \
#     --split train \
#     --data_root ./data/afo \
#     --model_path output_dir/afo_project/mambayolo_afo_two/weights/best.pt \
#     --save_dir out/det_results_mamba/afo/{split} \
#     --generate_meta_data \
#     --vis \
#     --imgsz 960 \
#     --conf 0.05

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger

from ultralytics import YOLO


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "valid", "test"],
    )

    parser.add_argument(
        "--data_root",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--save_dir",
        default="out/det_results_mamba/afo/{split}",
    )

    parser.add_argument(
        "--imgsz",
        default=960,
        type=int,
    )

    parser.add_argument(
        "--conf",
        default=0.05,
        type=float,
    )

    parser.add_argument(
        "--high_thresh",
        default=0.5,
        type=float,
    )

    parser.add_argument(
        "--device",
        default="0",
        type=str,
    )

    parser.add_argument(
        "--vis",
        action="store_true",
    )

    parser.add_argument(
        "--generate_meta_data",
        action="store_true",
    )

    return parser.parse_args()


def save_results(folder_name, seq_name, result_dict):

    os.makedirs(folder_name, exist_ok=True)

    out_path = os.path.join(folder_name, f"{seq_name}.txt")

    with open(out_path, "w") as f:

        for frame_id, output in result_dict.items():

            for det in output:

                x, y, w, h, conf = det

                f.write(
                    f"{frame_id},-1,"
                    f"{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                    f"{conf:.6f},-1,-1,-1\n"
                )

    logger.info(f"Saved {out_path}")


def save_meta_data(folder_name, meta_data):

    os.makedirs(folder_name, exist_ok=True)

    path = os.path.join(folder_name, "meta_data.txt")

    with open(path, "w") as f:

        for k, v in meta_data.items():

            f.write(
                f"{k},{v[0]},{v[1]}\n"
            )

    logger.info(f"Saved {path}")


def plot_img(img, seq_name, frame_id, detections, save_dir):

    vis_dir = os.path.join(
        save_dir,
        "vis_results",
        seq_name,
    )

    os.makedirs(vis_dir, exist_ok=True)

    img_vis = img.copy()

    for det in detections:

        x, y, w, h, conf = det

        x1 = int(x)
        y1 = int(y)

        x2 = int(x + w)
        y2 = int(y + h)

        cv2.rectangle(
            img_vis,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            img_vis,
            f"{conf:.2f}",
            (x1, max(10, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    cv2.imwrite(
        os.path.join(
            vis_dir,
            f"{frame_id:05d}.jpg",
        ),
        img_vis,
    )


def main(args):

    logger.info("Loading Mamba-YOLO model")

    model = YOLO(args.model_path)

    split_map = {
        "train": "train",
        "val": "validation",
        "valid": "validation",
        "test": "test",
    }

    target_split = split_map[args.split]

    img_dir = (
        Path(args.data_root)
        / target_split
        / "img"
    )

    if not img_dir.exists():

        raise FileNotFoundError(img_dir)

    save_dir = args.save_dir.format(
        split=args.split
    )

    all_images = (
        list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.png"))
    )

    seq_groups = {}

    for img_path in all_images:

        stem = img_path.stem

        parts = stem.split("_")

        if len(parts) < 2:
            continue

        seq_id = "_".join(parts[:-1])

        try:
            frame_idx = int(parts[-1])
        except ValueError:
            continue

        seq_groups.setdefault(
            seq_id,
            []
        )

        seq_groups[seq_id].append(
            (frame_idx, img_path)
        )

    meta_data = {}

    for seq_id, frame_list in seq_groups.items():

        frame_list.sort(
            key=lambda x: x[0]
        )

        logger.info(
            f"Processing sequence {seq_id}"
        )

        det_result_dict = {}

        first_img = cv2.imread(
            str(frame_list[0][1])
        )

        meta_data[seq_id] = [
            first_img.shape[0],
            first_img.shape[1],
        ]

        virtual_frame_id = 1

        for _, img_path in tqdm(
            frame_list,
            desc=seq_id,
        ):

            img = cv2.imread(
                str(img_path)
            )

            results = model.predict(
                source=img,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False,
                device=args.device,
            )

            result = results[0]

            dets = []

            if len(result.boxes) > 0:

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                confs = (
                    result.boxes.conf
                    .cpu()
                    .numpy()
                )

                for box, conf in zip(
                    boxes,
                    confs,
                ):

                    x1, y1, x2, y2 = box

                    w = x2 - x1
                    h = y2 - y1

                    dets.append([
                        x1,
                        y1,
                        w,
                        h,
                        conf,
                    ])

            dets = np.array(
                dets,
                dtype=np.float32,
            )

            if len(dets) == 0:

                dets = np.empty(
                    (0, 5),
                    dtype=np.float32,
                )

            det_result_dict[
                virtual_frame_id
            ] = dets

            if args.vis:

                vis_dets = dets[
                    dets[:, 4]
                    > args.high_thresh
                ]

                if len(vis_dets):

                    plot_img(
                        img,
                        seq_id,
                        virtual_frame_id,
                        vis_dets,
                        save_dir,
                    )

            virtual_frame_id += 1

        save_results(
            save_dir,
            seq_id,
            det_result_dict,
        )

    if args.generate_meta_data:

        save_meta_data(
            save_dir,
            meta_data,
        )


if __name__ == "__main__":
    args = get_args()
    main(args)