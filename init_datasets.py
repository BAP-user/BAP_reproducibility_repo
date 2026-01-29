import os
import requests
import tarfile

import kagglehub
import shutil
from wilds import get_dataset
import subprocess


dataset = get_dataset(
    dataset="waterbirds",
    root_dir="/CLIP/data/wilds_data",
    download=True
)

def extract_tgz(tgz_path, extract_dir):
    """
    Extract a .tgz or .tar.gz file into a destination directory.

    Parameters
    ----------
    tgz_path : str
        Path to the .tgz/.tar.gz archive.
    extract_dir : str
        Directory where contents will be extracted.

    Returns
    -------
    extract_dir : str
        The directory where the files were extracted.
    """

    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    return extract_dir

def download_file(url, save_dir, filename=None):
    """
    Download a file from a URL and save it into save_dir.

    Parameters
    ----------
    url : str
        The download URL.
    save_dir : str
        Destination directory to save the file.
    filename : str or None
        Optional custom filename. If None, it is inferred from URL.

    Returns
    -------
    full_path : str
        Path to the downloaded file.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Infer filename from URL if not provided
    if filename is None:
        filename = url.split("/")[-1].split("?")[0]

    full_path = os.path.join(save_dir, filename)

    # Stream download to avoid large memory usage
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(full_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return full_path



def download_kagglehub_to_dir(dataset_name, target_dir):
    """
    Download a KaggleHub dataset and copy it to a specified directory.

    Parameters
    ----------
    dataset_name : str
        Name used in kagglehub.dataset_download(), e.g. "mittalshubham/images256"
    target_dir : str
        Directory where you want the final dataset to reside.

    Returns
    -------
    target_dir : str
        Path to your dataset in its final location.
    """

    # Step 1: Download to KaggleHub cache
    cache_path = kagglehub.dataset_download(dataset_name)

    # Step 2: Ensure your target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Step 3: Copy everything from cache -> target directory
    shutil.copytree(cache_path, target_dir, dirs_exist_ok=True)

    return target_dir


url = "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1"

save_dir = '/CLIP/data/waterbirds'

file_path = download_file(url, save_dir)
print("Downloaded to:", file_path)

url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"

save_dir = '/CLIP/data/waterbirds'

file_path = download_file(url, save_dir)
print("Downloaded to:", file_path)

extract_tgz('/CLIP/data/waterbirds/CUB_200_2011.tgz', '/CLIP/data/waterbirds')
extract_tgz('/CLIP/data/waterbirds/segmentations.tgz', '/CLIP/data/waterbirds/CUB_200_2011')

dataset = "mittalshubham/images256"
save_dir = '/CLIP/data/places'

final_path = download_kagglehub_to_dir(dataset, save_dir)

print("Dataset stored at:", final_path)

dataset = "awsaf49/coco-2017-dataset"
save_dir = '/CLIP/data/COCO'
final_path = download_kagglehub_to_dir(dataset, save_dir)


url = "https://www.dropbox.com/scl/fo/ix8u21atdwrstmgjkz2rx/AGivum8GpYvGJyi7R0UoHnQ/DG_Benchmark/NICO_Unique.zip?rlkey=kcbecly7tetqu57v4tsmo095m&dl=1"
output = "nico_benchmark.zip"

subprocess.run(["wget", "-O", output, url], check=True)
