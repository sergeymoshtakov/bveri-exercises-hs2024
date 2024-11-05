from pathlib import Path

from .. import utils


def download(download_dir: Path):
    utils.download_from_gdrive_and_extract_zip(
        file_id="1Q-qLQ2RTbpBExsPI4v4B-pBq2K3OrwLb",
        save_path=download_dir.joinpath("concrete_data.zip"),
        extract_path=download_dir.joinpath("concrete/"),
    )
