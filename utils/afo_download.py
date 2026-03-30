import subprocess
import zipfile
from pathlib import Path
import shutil

AFO_KAGGLE_NAME = "jangsienicajzkowy/afo-aerial-dataset-of-floating-objects"


def download_afo_dataset(data_dir="data/afo", force=False):
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"

    dataset_ready = (data_dir / "train").exists()

    if dataset_ready and not force:
        return

    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "afo.zip"

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            AFO_KAGGLE_NAME,
            "-p",
            str(raw_dir),
            "--force",
        ],
        check=True,
    )

    downloaded_zip = list(raw_dir.glob("*.zip"))[0]

    if downloaded_zip != zip_path:
        downloaded_zip.rename(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw_dir)

    extracted = [p for p in raw_dir.iterdir() if p.is_dir()][0]

    splits = {
        "train": "train",
        "valid": "valid",
        "test": "test",
    }

    for split_src, split_dst in splits.items():
        src = extracted / split_src
        dst = data_dir / split_dst

        if dst.exists():
            shutil.rmtree(dst)

        shutil.copytree(src, dst)

    shutil.rmtree(raw_dir)