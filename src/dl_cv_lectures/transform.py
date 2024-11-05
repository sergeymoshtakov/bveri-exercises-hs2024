"""Transformations."""
import random

import torch
import torch.nn.functional as F
import torchvision.transforms as T


class RandomQuadrantPad(torch.nn.Module):
    """Randomly place the input image in one of the four quadrants and pad the rest based on image size."""

    def __init__(self, choices=["top_left", "top_right", "bottom_left", "bottom_right"]):
        super().__init__()
        self.choices = choices

    def forward(self, image):
        """
        Args:
            image (PIL Image or Tensor): Input image to transform.

        Returns:
            Tensor: Transformed image with padding to place it in a random quadrant.
        """
        # If the input is a PIL Image, convert it to a Tensor
        if isinstance(image, torch.Tensor):
            height, width = image.shape[1], image.shape[2]  # Tensor shape: [C, H, W]
        else:
            image = T.ToTensor()(image)  # Convert PIL to Tensor
            height, width = image.shape[1], image.shape[2]  # Get dimensions after conversion

        # Randomly select a quadrant position
        position = random.choice(self.choices)

        # Calculate padding based on the selected position
        if position == "top_left":
            output_tensor = F.pad(image, (0, width, 0, height), "constant", 0)
        elif position == "top_right":
            output_tensor = F.pad(image, (width, 0, 0, height), "constant", 0)
        elif position == "bottom_left":
            output_tensor = F.pad(image, (0, width, height, 0), "constant", 0)
        elif position == "bottom_right":
            output_tensor = F.pad(image, (width, 0, height, 0), "constant", 0)

        return output_tensor
