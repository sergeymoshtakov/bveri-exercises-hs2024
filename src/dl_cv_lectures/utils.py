"""Different Util Functions"""
import os
from pathlib import Path

from sklearn.model_selection import train_test_split


def download_and_extract_zip(url: str, save_path: Path, extract_path: Path):
    """
    Downloads a ZIP file from a given URL and extracts its contents to a specified directory.

    Args:
        url (str): The URL of the ZIP file to download.
        save_path (Path): The path where the downloaded ZIP file will be saved.
        extract_path (Path): The directory where the ZIP file will be extracted.
    """
    import os
    import zipfile

    import requests

    # Make sure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not save_path.exists():
        print(f"Starting download from {url}")
        # Download the file
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                _ = file.write(chunk)

        print(f"File downloaded and saved to {save_path}")
    else:
        print(f"File {save_path} exists already - not overwriting")

    if not extract_path.exists():
        print(f"Starting extracting {extract_path}")
        # Unzip the file
        with zipfile.ZipFile(save_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        print(f"File extracted to {extract_path}")
    else:
        print(f"File {extract_path} exists already - not overwriting")


def download_from_gdrive_and_extract_zip(file_id: str, save_path: Path, extract_path: Path):
    """
    Downloads a ZIP file from Google Drive using its file ID and extracts its contents to a specified directory.

    Args:
        file_id (str): The Google Drive file ID of the ZIP file to download.
        save_path (Path): The path where the downloaded ZIP file will be saved.
        extract_path (Path): The directory where the ZIP file will be extracted.
    """
    import os
    import zipfile

    import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    if not save_path.exists():
        gdown.download(url, str(save_path), quiet=False)
        print(f"File downloaded and saved to {save_path}")

    if not extract_path.exists():
        print(f"Starting to extract... {extract_path}")
        # Unzip the file
        with zipfile.ZipFile(save_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        print(f"File extracted to {extract_path}")


def download_file_from_gdrive(file_id: str, download_path: str | Path):
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    if download_path.exists():
        print(f"File / path {download_path} already exists, not overwriting")
    else:
        gdown.download(url, str(download_path), quiet=False)   
        

def delete_bad_file(file_path: Path):
    """
    Deletes a specified file if it exists.

    Args:
        file_path (Path): The path of the file to be deleted.
    """
    import os

    # Check if file exists before trying to delete it
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} has been deleted")
    else:
        print(f"{file_path} does not exist")


def find_all_imges_and_their_labels(image_dir: str) -> list[dict]:
    """
    Load image paths and corresponding labels.

    Args:
        image_dir: Directory with all the images.

    Returns:
        A list of dicts, one for each obsevation
    """
    observations = list()
    classes = os.listdir(image_dir)
    image_extensions = {".jpg", ".jpeg", ".png"}

    for label in classes:
        class_dir = os.path.join(image_dir, label)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if any(img_path.lower().endswith(ext) for ext in image_extensions):
                observation = {"image_path": img_path, "label": label}
                observations.append(observation)
    return observations


def create_train_test_split(
    ids: list[int],
    labels: list[int | str],
    random_state: int = 123,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> tuple[list[dict], list[dict], list[dict]]:

    train_ids, test_ids = train_test_split(
        ids,
        stratify=labels,
        test_size=test_size,
        random_state=random_state,
    )

    train_ids, val_ids = train_test_split(
        train_ids,
        stratify=[labels[i] for i in train_ids],
        test_size=val_size,
        random_state=random_state,
    )

    return train_ids, val_ids, test_ids
