"""Shapes Dataset for Object Localization and Detection.

This dataset generates synthetic images containing random shapes (circles, rectangles, and triangles)
for object localization and detection tasks. Each image has a specified number of randomly positioned 
and colored shapes on a customizable background. Each shape's class and bounding box coordinates 
are stored as annotations.
"""
from typing import Callable

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2.functional as F

from torch.utils.data import Dataset


class ShapeDataset(Dataset):
    """
        Initializes the ShapeDataset with the specified parameters.

        Args:
            num_samples (int): Number of images to generate.
            img_size (int): The height and width of each generated image in pixels.
            seed (int): Random seed for reproducibility.
            max_number_of_shapes_per_image (int): Maximum number of shapes in each image.
            background (str): Background color, either "random" or "white".
            transforms (callable, optional): Optional transforms to apply to the images.
    """
    def __init__(
        self,
        num_samples: int=1000,
        img_size: int=256,
        seed: int=123,
        max_number_of_shapes_per_image: int=5,
        background: str="random",
        transforms: Callable | None=None,
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.max_number_of_shapes_per_image = max_number_of_shapes_per_image
        self.classes = ["circle", "rectangle", "triangle"]
        self.class_map = {k: i for i, k in enumerate(self.classes)}
        self.background = background
        self.transforms = transforms

    def __len__(self):
        return self.num_samples

    def draw_random_circle(self, img, annotations):
        # Generate a random color for the circle
        color = tuple(int(value) for value in self.rng.integers(50, 255, size=3))

        # Generate random coordinates for the center of the circle
        center = (self.rng.integers(0, self.img_size), self.rng.integers(0, self.img_size))

        # Generate a random radius for the circle (between 5% and 20% of img_size)
        radius = int(self.rng.uniform(0.05 * self.img_size, 0.2 * self.img_size))

        # Draw the filled circle on the image
        cv2.circle(img, center, radius, color, -1)

        # Calculate the bounding box for the circle (xmin, ymin, xmax, ymax)
        bbox = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)

        # Append the annotation to the list with the class "circle"
        annotations.append({"class": self.class_map["circle"], "box": bbox})

    def draw_random_rectangle(self, img, annotations):
        # Generate a random color
        color = tuple(int(value) for value in self.rng.integers(50, 255, size=3))

        # Generate random coordinates for the top-left corner of the rectangle (xmin, ymin)
        xmin = self.rng.integers(0, int(self.img_size * 0.8))
        ymin = self.rng.integers(0, int(self.img_size * 0.8))

        # Generate random width and height for the rectangle
        width = int(self.rng.uniform(0.05 * self.img_size, 0.2 * self.img_size))
        height = int(self.rng.uniform(0.05 * self.img_size, 0.2 * self.img_size))

        # Calculate the coordinates for the bottom-right corner of the rectangle (xmax, ymax)
        xmax = xmin + width
        ymax = ymin + height

        # Draw the filled rectangle on the image
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, -1)

        # Define the bounding box (xmin, ymin, xmax, ymax)
        bbox = (xmin, ymin, xmax, ymax)

        # Append the annotation to the list with the class "rectangle"
        annotations.append({"class": self.class_map["rectangle"], "box": bbox})

    def draw_random_triangle(self, img, annotations):
        # Generate a random color
        color = tuple(int(value) for value in self.rng.integers(50, 255, size=3))

        # Generate a random point for the top-left vertex of the triangle
        pt1 = (
            self.rng.integers(0, int(self.img_size * 0.8)),
            self.rng.integers(0, int(self.img_size * 0.8)),
        )

        # Generate random width and height for the triangle
        width = int(self.rng.uniform(0.05 * self.img_size, 0.2 * self.img_size))
        height = int(self.rng.uniform(0.05 * self.img_size, 0.2 * self.img_size))

        # Calculate the coordinates for the other two points of the triangle
        pt2 = (pt1[0] + width, pt1[1])
        pt3 = (int((pt1[0] + pt2[0]) / 2), pt1[1] - height)

        # Define the triangle points as a numpy array
        triangle_points = np.array([pt1, pt2, pt3])

        # Draw the filled triangle on the image
        cv2.drawContours(img, [triangle_points], 0, color, -1)

        # Calculate the bounding box (xmin, ymin, xmax, ymax)
        x_coords = [pt[0] for pt in [pt1, pt2, pt3]]
        y_coords = [pt[1] for pt in [pt1, pt2, pt3]]
        bounding_box = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

        # Append the annotation to the list with the class "triangle"
        annotations.append({"class": self.class_map["triangle"], "box": bounding_box})

    def _clip_bounding_box_to_image(self, annotations):
        # Clip boxes to image size
        box_unclipped = torch.tensor(annotations[-1]["box"]).reshape(1, -1)
        box_clipped = torchvision.ops.clip_boxes_to_image(
            box_unclipped, (self.img_size - 1, self.img_size - 1)
        ).squeeze(0)
        annotations[-1]["box"] = [int(x) for x in box_clipped]

    def __getitem__(self, idx):
        # Seed the random number generator with a combination of the initial seed and the index
        self.rng = np.random.default_rng(seed=self.seed + idx)  # Seed with a base seed plus the index

        if self.background == "random":
            img = self.rng.integers(0, 256, (self.img_size, self.img_size, 3), dtype=np.uint8)
        elif self.background == "white":
            img = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255  # White background

        annotations = []

        # Generate a deterministic number of shapes per image for this index
        num_shapes = self.rng.integers(
            1, self.max_number_of_shapes_per_image + 1
        )
        
        # Draw shapes based on the seeded random generator
        for _ in range(num_shapes):
            choice = self.rng.choice(["circle", "rectangle", "triangle"])
            if choice == "circle":
                self.draw_random_circle(img, annotations)
            elif choice == "rectangle":
                self.draw_random_rectangle(img, annotations)
            else:
                self.draw_random_triangle(img, annotations)
            self._clip_bounding_box_to_image(annotations)

        # Convert image to PyTorch tensor if transforms are provided
        if self.transforms:
            img = self.transforms(img)

        labels = {
            "class": [x["class"] for x in annotations],
            "box": [x["box"] for x in annotations],
        }

        return img, labels
