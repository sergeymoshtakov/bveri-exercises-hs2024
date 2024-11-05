from pathlib import Path

from .. import utils


def download(download_dir: Path):
    utils.download_from_gdrive_and_extract_zip(
        file_id="1bXWW8v-vASZ6dUv2CchhrbvyQU4uE2dk",
        save_path=download_dir.joinpath("stanford_background_dataset.zip"),
        extract_path=download_dir.joinpath("stanford_background_dataset/"),
    )
