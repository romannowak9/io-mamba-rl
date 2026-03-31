from pathlib import Path
import shutil
import dataset_tools as dtools


def download_afo_dataset(data_dir="data", force=False):
    data_dir = Path(data_dir)

    if data_dir.exists() and not force:
        print(f"AFO dataset already exists at {data_dir}. Skipping download.")
        return data_dir
    elif force and data_dir.exists():
        print("Removing existing dataset...")
        shutil.rmtree(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)

    dtools.download(dataset="AFO", dst_dir=str(data_dir))

    # Delete .tar archive
    for tar_file in data_dir.glob("*.tar"):
        print(f"Removing archive: {tar_file}")
        tar_file.unlink()

    print(f"AFO dataset ready at: {data_dir}")


if __name__ == "__main__":
    download_afo_dataset()