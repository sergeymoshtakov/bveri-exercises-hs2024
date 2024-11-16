from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .. import utils


def download(download_dir: Path):
    utils.download_from_gdrive_and_extract_zip(
        file_id="1bXWW8v-vASZ6dUv2CchhrbvyQU4uE2dk",
        save_path=download_dir.joinpath("stanford_background_dataset.zip"),
        extract_path=download_dir.joinpath("stanford_background_dataset/"),
    )


class StanfordBackgroundDataset(Dataset):
    def __init__(
        self,
        root_path: Path,
        transform_images: Callable = None,
        transform_labels: Callable = None,
    ):
        """
        Initializes the dataset.

        Args:
            root_path (Path): Path to the dataset directory.
            transform_images (callable, optional): Transformation function for images.
            transform_labels (callable, optional): Transformation function for labels.
        """
        self.root_path = root_path
        self.transform_images = transform_images
        self.transform_labels = transform_labels
        self.image_paths = list((root_path / "images").glob("*.jpg"))
        self.classes = [
            "sky",
            "tree",
            "road",
            "grass",
            "water",
            "building",
            "mountain",
            "foreground object",
        ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor | Image.Image, torch.Tensor, torch.Tensor]:
        """
        Retrieves the image and corresponding label masks for a given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor | Image.Image): The transformed image or original image.
                - label_masks (torch.Tensor): A binary mask tensor of shape (K, H, W) where K is the number of classes.
                  Each channel represents the binary mask for a specific class.
                - labels_tensor (torch.Tensor): A segmentation map tensor of shape (1, H, W) indicating class indices
                for each pixel.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        label_path = self.root_path / f"labels/{image_path.stem}.regions.txt"
        labels = self._parse_regions(label_path)

        labels_tensor = torch.tensor(labels).unsqueeze(0).clamp(0, len(self.classes) - 1)
        label_masks = torch.zeros(len(self.classes), *labels.shape).scatter_(0, labels_tensor, 1)

        if self.transform_images:
            image = self.transform_images(image)
        if self.transform_labels:
            label_masks = self.transform_labels(label_masks)
            labels_tensor = self.transform_labels(labels_tensor)

        return image, label_masks, labels_tensor

    def _parse_regions(self, path: Path) -> np.ndarray:
        with open(path) as file:
            return np.array([list(map(int, line.split())) for line in file])
