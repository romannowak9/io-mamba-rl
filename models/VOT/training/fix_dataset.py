from pathlib import Path
import shutil


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

if __name__ == "__main__":
    flatten_otb_dataset("data/OTB/OTB50")
    flatten_otb_dataset("data/OTB/OTB100")