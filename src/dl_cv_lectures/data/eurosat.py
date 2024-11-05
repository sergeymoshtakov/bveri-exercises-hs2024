from pathlib import Path

from .. import utils


def download(download_dir: Path):
    utils.download_and_extract_zip(
        url="https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1",
        save_path=download_dir.joinpath("EuroSAT_RGB.zip"),
        extract_path=download_dir.joinpath("EuroSAT_RGB/"),
    )
