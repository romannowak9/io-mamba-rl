from pathlib import Path
import shutil
import math


def flatten_otb_dataset(root_dir: str) -> None:
    """
    Convert OTB-style structure:

        OTB50/
            Basketball/
                groundtruth_rect.txt
                img/
                    0001.jpg
                    0002.jpg

    into:

        OTB50/
            Basketball/
                groundtruth.txt
                0001.jpg
                0002.jpg
    """
    root = Path(root_dir)

    for sequence_dir in root.iterdir():
        if not sequence_dir.is_dir():
            continue

        # Rename groundtruth file
        old_gt = sequence_dir / "groundtruth_rect.txt"
        new_gt = sequence_dir / "groundtruth.txt"

        if old_gt.exists():
            old_gt.rename(new_gt)
            print(f"Renamed: {old_gt} -> {new_gt}")

        # Move images from img/
        img_dir = sequence_dir / "img"

        if img_dir.exists() and img_dir.is_dir():
            for image_file in img_dir.iterdir():
                destination = sequence_dir / image_file.name

                if destination.exists():
                    raise FileExistsError(
                        f"Cannot move {image_file}: "
                        f"{destination} already exists."
                    )

                shutil.move(str(image_file), str(destination))

            # Remove img directory if empty
            try:
                img_dir.rmdir()
                print(f"Removed empty directory: {img_dir}")
            except OSError:
                print(f"Directory not empty: {img_dir}")

def cleanse_nan_groundtruths(dataset_root: str, dry_run: bool = False) -> None:
    """
    Remove NaN groundtruth rows and their corresponding images.

    Expected structure:
        dataset_root/
            sequence_1/
                groundtruth.txt
                0001.jpg
                0002.jpg
                ...
            sequence_2/
                groundtruth.txt
                ...

    Args:
        dataset_root: Path to the main dataset folder.
        dry_run: If True, only prints what would be removed.
    """

    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_root}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for seq_dir in sorted(dataset_root.iterdir()):
        if not seq_dir.is_dir():
            continue

        gt_path = seq_dir / "groundtruth.txt"
        if not gt_path.exists():
            print(f"[SKIP] No groundtruth.txt in {seq_dir}")
            continue

        images = sorted(
            [
                p for p in seq_dir.iterdir()
                if p.is_file() and p.suffix.lower() in image_extensions
            ]
        )

        lines = gt_path.read_text().splitlines()

        if len(images) != len(lines):
            print(
                f"[WARN] {seq_dir.name}: {len(images)} images, "
                f"{len(lines)} groundtruth lines"
            )

        kept_lines = []
        removed_count = 0

        for idx, line in enumerate(lines):
            has_nan = "nan" in line.lower()

            if has_nan:
                removed_count += 1

                if idx < len(images):
                    if dry_run:
                        print(f"[DRY RUN] Would delete: {images[idx]}")
                    else:
                        images[idx].unlink()
                        print(f"[DELETE] {images[idx]}")
                else:
                    print(f"[WARN] No image found for groundtruth line {idx + 1}")

            else:
                kept_lines.append(line)

        if removed_count > 0:
            if dry_run:
                print(
                    f"[DRY RUN] Would update {gt_path}: "
                    f"remove {removed_count} lines"
                )
            else:
                gt_path.write_text("\n".join(kept_lines) + "\n")
                print(
                    f"[OK] {seq_dir.name}: removed "
                    f"{removed_count} NaN samples"
                )
        else:
            print(f"[OK] {seq_dir.name}: no NaN values found")

if __name__ == "__main__":
    #flatten_otb_dataset("data/OTB/OTB50")
    #flatten_otb_dataset("data/OTB/OTB100")
    cleanse_nan_groundtruths("data/TrackingDataset", dry_run=False)
