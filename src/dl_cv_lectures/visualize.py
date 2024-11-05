"""Visualization Functions."""
import math
from textwrap import wrap

import torch
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms.v2 import functional as TF


def plot_square_collage_with_captions(
    images: list[torch.Tensor],
    captions: list[str],
    caption_width: int = 30,
    global_normalize: bool = False,
):
    """Plot a square collage of images with captions on top of each image."""

    num_images = len(images)
    num_images_per_axis = math.ceil(math.sqrt(num_images))  # Define the collage grid size

    pil_images = [TF.to_pil_image(img) for img in images]  # Convert tensors to PIL images

    if global_normalize:
        # Find global min and max across all images for normalization
        all_images = torch.cat([img.flatten() for img in images])
        vmin, vmax = all_images.min().item(), all_images.max().item()
    else:
        vmin, vmax = 0.0, 255.0

    # Create the figure for the collage
    fig, axes = plt.subplots(
        figsize=(num_images_per_axis * 2, num_images_per_axis * 2),
        nrows=num_images_per_axis,
        ncols=num_images_per_axis,
    )

    for i in range(num_images):
        ax = axes.flat[i]

        # Wrap caption text to fit within the caption width
        caption = captions[i]
        caption = "\n".join(wrap(caption, caption_width))

        # Plot image and set the title to the caption
        ax.imshow(pil_images[i], vmin=vmin * 255.0, vmax=vmax * 255.0)
        ax.set_title(caption, fontsize=10)
        ax.axis("off")  # Remove axes for cleaner display
    return fig, axes


def plot_square_collage(
    images: list[torch.Tensor] | list[Image.Image],
    global_normalize: bool = False,
):
    """Plot a square collage of images."""

    num_images = len(images)
    num_images_per_axis = math.ceil(math.sqrt(num_images))  # Define the collage grid size

    if isinstance(images[0], torch.Tensor):
        pil_images = [TF.to_pil_image(img) for img in images]  # Convert tensors to PIL images
    else:
        pil_images = images

    if global_normalize:
        # Find global min and max across all images for normalization
        all_images = np.concatenate([np.array(img).flatten() for img in pil_images])
        vmin, vmax = all_images.min().item(), all_images.max().item()
    else:
        vmin, vmax = 0.0, 255.0

    # Create the figure for the collage
    fig, axes = plt.subplots(
        figsize=(num_images_per_axis * 2, num_images_per_axis * 2),
        nrows=num_images_per_axis,
        ncols=num_images_per_axis,
    )

    for i in range(num_images):
        ax = axes.flat[i]
        # Plot image and set the title to the caption
        ax.imshow(pil_images[i], vmin=vmin * 255.0, vmax=vmax * 255.0)
        ax.axis("off")  # Remove axes for cleaner display
    return fig, axes
