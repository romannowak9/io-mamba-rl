from pathlib import Path
import shutil
import json
import cv2 
import numpy as np


def split_afo_into_sequences(
    image_dir,
    annotation_dir,
    output_dir,
):
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    processed = 0
    missing_annotations = []

    for image_path in image_dir.iterdir():

        if image_path.suffix.lower() not in image_extensions:
            continue

        # Example:
        # a_027.jpg -> a_027
        stem = image_path.stem

        if "_" not in stem:
            print(f"Skipping malformed filename: {image_path.name}")
            continue

        # a_027 -> a
        sequence_id = stem.rsplit("_", 1)[0]

        sequence_dir = output_dir / sequence_id
        sequence_dir.mkdir(parents=True, exist_ok=True)

        annotation_name = image_path.name + ".json"
        annotation_path = annotation_dir / annotation_name

        shutil.copy2(
            image_path,
            sequence_dir / image_path.name,
        )

        if annotation_path.exists():
            shutil.copy2(
                annotation_path,
                sequence_dir / annotation_path.name,
            )
        else:
            missing_annotations.append(image_path.name)

        processed += 1

    print(f"Processed {processed} images.")
    print(f"Created {len(list(output_dir.iterdir()))} sequence folders.")

    if missing_annotations:
        print(
            f"Warning: {len(missing_annotations)} files missing annotations."
        )
        for name in missing_annotations[:20]:
            print("  ", name)

    return missing_annotations

def afo_rectangle_to_box(exterior):
    (x1, y1), (x2, y2) = exterior

    w = x2 - x1
    h = y2 - y1

    return [
        x1 + w / 2,
        y1 + h / 2,
        w,
        h,
    ]


def rect_to_xywh(rect):
    """
    Supervisely rectangle:
    [[x1,y1],[x2,y2]]

    ->
    [x_center, y_center, width, height]
    """
    (x1, y1), (x2, y2) = rect

    width = x2 - x1
    height = y2 - y1

    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0

    return [x_center, y_center, width, height]


def crop_object_patch(
    image,
    image_name,
    frame_record,
    output_dir,
    crop_sizes=(336, 448, 560),
    offset_fraction=0.2,
):
    """
    Generate VOT-style crops from a frame_record.

    Parameters
    ----------
    image : np.ndarray
        Full-resolution image.

    image_name : str
        Original image filename.

    frame_record : dict
        Frame record produced by process_afo_annotations().

    output_dir : str | Path

    Returns
    -------
    list[dict]
        New frame records in cropped coordinates.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_h, img_w = image.shape[:2]

    generated_records = []

    for obj in frame_record["objects"]:

        obj_id = obj["id"]
        cls = obj["class"]

        x_center, y_center, box_w, box_h = obj["bbox"]

        valid_sizes = [
            s
            for s in crop_sizes
            if box_w < s / 2 and box_h < s / 2
        ]

        if len(valid_sizes) == 0:
            continue

        crop_size = np.random.choice(valid_sizes)

        max_dx = crop_size * offset_fraction
        max_dy = crop_size * offset_fraction

        dx = np.random.uniform(-max_dx, max_dx)
        dy = np.random.uniform(-max_dy, max_dy)

        crop_center_x = x_center + dx
        crop_center_y = y_center + dy

        x1 = int(round(crop_center_x - crop_size / 2))
        y1 = int(round(crop_center_y - crop_size / 2))
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - img_w)
        pad_bottom = max(0, y2 - img_h)

        padded_image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

        crop = padded_image[y1:y2, x1:x2]

        crop_filename = (
            f"{Path(image_name).stem}_obj_{obj_id}.jpg"
        )

        crop_path = output_dir / crop_filename

        cv2.imwrite(str(crop_path), crop)

        new_x = x_center - (x1 - pad_left)
        new_y = y_center - (y1 - pad_top)

        generated_records.append(
            {
                "image": crop_filename,
                "source_image": image_name,
                "object_id": obj_id,
                "class": cls,
                "bbox": [
                    float(new_x),
                    float(new_y),
                    float(box_w),
                    float(box_h),
                ],
            }
        )

    return generated_records

def process_afo_annotations(
    source_root,
    output_root,
    ignored_classes=None,
    max_objects=None,
    min_box_width=25,
):
    """
    Parameters
    ----------
    source_root : str | Path
        Root directory containing sequence subdirectories.

    output_root : str | Path
        Where processed annotation files will be written.

    ignored_classes : list[str] | None
        Classes to remove.

    max_objects : int | None
        Skip frames containing more than this many objects
        after filtering.

    min_box_width : int | float
        Ignore boxes whose width is below this threshold.
    """

    source_root = Path(source_root)
    output_root = Path(output_root)

    ignored_classes = set(ignored_classes or [])

    output_root.mkdir(parents=True, exist_ok=True)

    filtered_classes = 0
    filtered_small_boxes = 0
    filtered_crowded_frames = 0

    total_frames = 0
    kept_frames = 0

    width_values = []

    total_saved_obj = 0


    for sequence_dir in sorted(source_root.iterdir()):

        if not sequence_dir.is_dir():
            continue

        print(f"Processing {sequence_dir.name}")

        sequence_records = []

        all_crop_records = []

        json_files = sorted(sequence_dir.glob("*.json"))

        for json_file in json_files:

            total_frames += 1

            with open(json_file, "r") as f:
                ann = json.load(f)

            boxes = []
            object_ids = []
            classes = []

            next_object_id = 0

            for obj in ann.get("objects", []):

                class_name = obj["classTitle"]

                if class_name in ignored_classes:
                    filtered_classes += 1
                    continue

                if obj["geometryType"] != "rectangle":
                    continue

                bbox = rect_to_xywh(
                    obj["points"]["exterior"]
                )

                x_center, y_center, width, height = bbox

                width_values.append(width)

                if width < min_box_width:
                    filtered_small_boxes += 1
                    continue

                boxes.append(bbox)
                object_ids.append(next_object_id)
                classes.append(class_name)

                next_object_id += 1

            num_objects = len(boxes)

            if num_objects == 0:
                continue

            if (
                max_objects is not None
                and num_objects > max_objects
            ):
                filtered_crowded_frames += 1
                continue

            frame_record = {
                "image": json_file.stem.replace(".jpg", "") + ".jpg",
                "objects": [
                    {
                        "id": obj_id,
                        "class": cls,
                        "bbox": bbox,
                    }
                    for obj_id, cls, bbox in zip(
                        object_ids,
                        classes,
                        boxes,
                    )
                ],
            }

            total_saved_obj += num_objects

            sequence_records.append(frame_record)
            kept_frames += 1


        

        output_sequence_dir = output_root / sequence_dir.name
        output_sequence_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for frame_record in sequence_records:

            image_name = frame_record["image"]
            image_jpg = cv2.imread(str(sequence_dir / image_name))

            crops = crop_object_patch(image_jpg, image_name, frame_record, output_sequence_dir)

            all_crop_records.extend(crops)



        output_file = output_sequence_dir / "annotations.json"

        output_file_crops = output_sequence_dir / "crop_annotations.json"

        with open(output_file, "w") as f:
            json.dump(
                sequence_records,
                f,
                indent=2,
            )
        
        with open(output_file_crops, "w") as f:
            json.dump(
                all_crop_records,
                f,
                indent=2,
            )

        print(
            f"  kept {len(sequence_records)} frames"
        )

    print("\n========== AFO PROCESSING SUMMARY ==========")
    print(f"Total frames scanned:      {total_frames}")
    print(f"Frames kept:              {kept_frames}")
    print(f"Filtered by class:        {filtered_classes}")
    print(f"Filtered width < {min_box_width}: {filtered_small_boxes}")
    print(f"Filtered crowded frames:  {filtered_crowded_frames}")
    print(f"Total objects saved:      {total_saved_obj}")

    if len(width_values) > 0:
        width_values.sort()

        print("\nBounding box width statistics:")
        print(f"Min:    {width_values[0]:.1f}")
        print(f"Median: {width_values[len(width_values)//2]:.1f}")
        print(f"Mean:   {sum(width_values)/len(width_values):.1f}")
        print(f"Max:    {width_values[-1]:.1f}")

    print("============================================")



def generate_groundtruth_files(
    root_dir,
    annotation_filename="crop_annotations.json",
):
    """
    Generates VOT-style groundtruth.txt files from crop_annotations.json.

    Expected bbox format:
        [x_center, y_center, width, height]

    One line is written for every image in crop_annotations.json.
    """

    root_dir = Path(root_dir)
    errors = 0

    for sequence_dir in sorted(root_dir.iterdir()):

        if not sequence_dir.is_dir():
            continue

        annotation_path = sequence_dir / annotation_filename

        with open(annotation_path, "r") as f:
            records = json.load(f)

        # print(f"\n{annotation_path}")

        # print("records type:", type(records))

        if len(records) > 0:
            print("first element type:", type(records[0]))
            print("first element:", records[0])

        if not annotation_path.exists():
            print(
                f"Skipping {sequence_dir.name}: "
                f"{annotation_filename} not found."
            )
            continue

        with open(annotation_path, "r") as f:
            records = json.load(f)

        if len(records) == 0:
            print(
                f"Skipping {sequence_dir.name}: "
                f"empty annotation file."
            )
            continue

        # Match image order used by VOT loader
        try:
            records.sort(key=lambda r: r["image"])
        except TypeError:
            print(
                f"Skipping {sequence_dir.name}: "
                f"invalid 'image' key in annotation file."
            )
            continue

        gt_path = sequence_dir / "groundtruth.txt"

        with open(gt_path, "w") as gt_file:

            for record in records:

                x, y, w, h = record["bbox"]

                gt_file.write(
                    f"{x:.2f},{y:.2f},{w:.2f},{h:.2f}\n"
                )

        print(
            f"{sequence_dir.name}: "
            f"wrote {len(records)} GT entries."
        )
    
        num_images = len(list(sequence_dir.glob("*.jpg")))

        if num_images != len(records):
            print(
                f"WARNING: {sequence_dir.name} "
                f"contains {num_images} images but "
                f"{len(records)} annotations."
            )
    
    print(f"Total errors: {errors}")

if __name__ == "__main__":
    pass
    # img_dir_test = "data/afo/test/img"
    # ann_dir_test = "data/afo/test/ann"
    # out_dir_test = "data/afo/test/sequences"
    # split_afo_into_sequences(
    #     image_dir=img_dir_test,
    #     annotation_dir=ann_dir_test,
    #     output_dir=out_dir_test
    # )

    # img_dir_train = "data/afo/train/img"
    # ann_dir_train = "data/afo/train/ann"
    # out_dir_train = "data/afo/train/sequences"
    # split_afo_into_sequences(
    #     image_dir=img_dir_train,
    #     annotation_dir=ann_dir_train,
    #     output_dir=out_dir_train
    # )

    # img_dir_validation = "data/afo/validation/img"
    # ann_dir_validation = "data/afo/validation/ann"
    # out_dir_validation = "data/afo/validation/sequences"
    # split_afo_into_sequences(
    #     image_dir=img_dir_validation,
    #     annotation_dir=ann_dir_validation,
    #     output_dir=out_dir_validation
    # )

    # process_afo_annotations(
    #     source_root="data/afo/train/sequences",
    #     output_root="data/afo/train/sequences_processed",
    #     ignored_classes=["small_obj", "object"],
    #     max_objects=25,
    # )

    # process_afo_annotations(
    #     source_root="data/afo/validation/sequences",
    #     output_root="data/afo/validation/sequences_processed",
    #     ignored_classes=["small_obj", "object"],
    #     max_objects=25,
    # )

    # process_afo_annotations(
    #     source_root="data/afo/test/sequences",
    #     output_root="data/afo/test/sequences_processed",
    #     ignored_classes=["small_obj", "object"],
    #     max_objects=25,
    # )

    generate_groundtruth_files("data/afo/train/sequences_processed")
    generate_groundtruth_files("data/afo/validation/sequences_processed")
    generate_groundtruth_files("data/afo/test/sequences_processed")