from pathlib import Path

from .. import utils


def download(download_dir: Path):
    utils.download_from_gdrive_and_extract_zip(
        file_id="1Bx3R56VBONS-x91wCDU6KX3xqPoJoH9P",
        save_path=download_dir.joinpath("scene_classification.zip"),
        extract_path=download_dir.joinpath("scene_classification/"),
    )
