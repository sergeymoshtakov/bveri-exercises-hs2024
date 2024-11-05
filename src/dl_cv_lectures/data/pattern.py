"""Experimental Random Dataset."""
from typing import Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PatternDataset(Dataset):
    """Creates a Dataset which displays a specific pattern at random locations.

    The pattern might be correct (label=0) or contain errors (label=1)

    max_errors: specifies the max  number of errors in the pattern

    max_x_y_shift: max shift of the pattern in y and/or x direction
    """

    def __init__(
        self,
        num_samples: int = 1000,
        image_side_length: int = 16,
        max_errors: int = 3,
        max_x_y_shift: int = 0,
        transform: Callable | None = None,
        seed: int = 123,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.image_side_length = image_side_length
        self.max_errors = max_errors
        self.max_x_y_shift = max_x_y_shift
        self.transform = transform

        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return self.num_samples

    def _create_binary_pattern(self, shift_x, shift_y):
        """
        Create a simple 'T' pattern in the center of a 16x16 image, then shift it by randomly
        removing rows or columns from the top/bottom or left/right based on shift_x and shift_y.
        """
        pattern = np.zeros((self.image_side_length, self.image_side_length), dtype=np.uint8)
        mid = self.image_side_length // 2

        # Create a centered 'T' pattern

        # Horizontal Bar (high, width)
        pattern[mid, self.max_x_y_shift : (self.image_side_length - self.max_x_y_shift)] = 255

        # Vertical Bar
        pattern[mid : (self.image_side_length - self.max_x_y_shift), mid] = 255

        # Randomly remove rows from the top and bottom
        if shift_y > 0:
            pattern = pattern[shift_y:, :]  # Remove shift_y rows from the top
            pattern = np.pad(pattern, ((0, shift_y), (0, 0)), mode="constant")  # Pad bottom
        elif shift_y < 0:
            pattern = pattern[:shift_y, :]  # Remove shift_y rows from the bottom
            pattern = np.pad(pattern, ((-shift_y, 0), (0, 0)), mode="constant")  # Pad top

        # Randomly remove columns from the left and right
        if shift_x > 0:
            pattern = pattern[:, shift_x:]  # Remove shift_x columns from the left
            pattern = np.pad(pattern, ((0, 0), (0, shift_x)), mode="constant")  # Pad right
        elif shift_x < 0:
            pattern = pattern[:, :shift_x]  # Remove shift_x columns from the right
            pattern = np.pad(pattern, ((0, 0), (-shift_x, 0)), mode="constant")  # Pad left

        return pattern

    def _get_adjacent_positions(self, pattern):
        """Get positions adjacent to the pattern for contamination."""
        adj_positions = set()
        pattern_positions = np.argwhere(pattern == 255)  # Find all pattern pixels

        for x, y in pattern_positions:
            # Check above, below, left, and right neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                # Ensure the new position is within image bounds
                if 0 <= nx < self.image_side_length and 0 <= ny < self.image_side_length:
                    # Add the position if it's not part of the pattern
                    if pattern[nx, ny] == 0:
                        adj_positions.add((nx, ny))

        return list(adj_positions)

    def _add_contamination(self, image, num_pixels):
        """Add contamination pixels only to adjacent positions around the pattern."""
        adj_positions = self._get_adjacent_positions(image)
        contamination_pixels = self._rng.choice(adj_positions, size=num_pixels, replace=False)
        for x, y in contamination_pixels:
            image[x, y] = 255  # Add contamination pixel
        return image

    def __getitem__(self, idx):
        local_rng = np.random.default_rng(idx)  # Seed the generator with the index

        # Create an empty 16x16 image with 3 channels
        image = np.zeros((self.image_side_length, self.image_side_length), dtype=np.uint8)

        # Random shift for the pattern
        shift_x = local_rng.integers(-self.max_x_y_shift, self.max_x_y_shift + 1)
        shift_y = local_rng.integers(-self.max_x_y_shift, self.max_x_y_shift + 1)

        # Generate the 'T' pattern with random shift
        pattern = self._create_binary_pattern(shift_x, shift_y)

        # Decide if this sample should be contaminated (class 1) or not (class 0)
        label = local_rng.choice([0, 1])

        if (label == 1) & self.max_errors > 0:
            # Add contamination to the pattern for class 1
            num_contaminations = local_rng.integers(self.max_errors, self.max_errors + 1)
            pattern = self._add_contamination(pattern, num_contaminations)

        image = pattern

        # vary brightness
        image = (image.astype(float) * self._rng.uniform(10 / 255.0, 1)).astype(np.uint8)

        # Convert image to PIL format for visualization (if needed)
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# # Example usage with seed:
# ds_train = PatternDataset(num_samples=100, seed=123, max_errors=3, max_x_y_shift=3)
# image, label = ds_train[0]
# print(f"Image shape: {image.size}, Label: {label}")
