import os
import tqdm
import requests
import re

def unzip_all_from_path(base_path, delete_zip=True):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".zip"):
                print(f"Unzipping {file}...")
                os.system(f"unzip {os.path.join(root, file)} -d {root}")
                if delete_zip:
                    print(f"Removing {file}...")
                    os.system(f"rm {os.path.join(root, file)}")

if __name__ == "__main__":
    unzip_all_from_path("./OTB/OTB50", delete_zip=False)  
    unzip_all_from_path("./OTB/OTB100", delete_zip=False)