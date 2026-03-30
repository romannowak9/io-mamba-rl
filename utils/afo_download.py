from pathlib import Path
import zipfile
import shutil
import kagglehub


AFO_KAGGLE_NAME = "jangsienicajzkowy/afo-aerial-dataset-of-floating-objects"


def download_afo_dataset(data_dir="data/afo", force=False):

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_ready = (data_dir / "train").exists()
    if dataset_ready and not force:
        print("Dataset already exists. Skipping download.")
        return

    print(f"Downloading AFO dataset to {data_dir}...")
    zip_path = Path(kagglehub.dataset_download(AFO_KAGGLE_NAME))
    print(f"Downloaded ZIP to: {zip_path}")

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    # Sprawdzenie, który katalog został rozpakowany
    extracted_dirs = [p for p in data_dir.iterdir() if p.is_dir() and p.name != "__MACOSX"]
    if not extracted_dirs:
        raise RuntimeError("Extracted directory not found!")
    extracted = extracted_dirs[0]

    splits = ["train", "valid", "test"]
    for split in splits:
        src = extracted / split
        dst = data_dir / split

        if dst.exists():
            shutil.rmtree(dst)

        shutil.copytree(src, dst)

    # Usuwamy ZIP i tymczasowy folder
    zip_path.unlink()
    if extracted.exists():
        shutil.rmtree(extracted)

    print("Dataset ready!")
