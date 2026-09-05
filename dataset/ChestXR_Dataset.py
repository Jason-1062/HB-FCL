import os
import pickle
import re
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import ImageFilter, Image
import cv2
import torch.nn.functional as F


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ChestXR_Dataset(data.Dataset):
    def __init__(self, data_path, split="train", crop_size=(224, 224)):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")

        # Augmentation and preprocessing pipeline.
        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(
                        num_output_channels=3
                    ),  # Keep a consistent channel count.
                    transforms.Resize((224, 224)),  # Resize to a fixed shape.
                    transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Optional random crop.
                    transforms.RandomRotation(
                        10
                    ),  # Mild rotation without changing structure.
                    # transforms.RandomHorizontalFlip(p=0.5),  # Optional horizontal flip.
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        # Initialize dataset parameters.
        self.data_path = data_path
        self.classes = {"normal": 0, "pneumonia": 1, "covid": 2}  # Three classes.
        self.num_classes = len(self.classes)
        self.image_paths = []  # All sample paths.
        self.labels = []  # Single-label indices (0/1/2).

        # Split the dataset and populate image paths and labels.
        if split == "train":
            train_path = os.path.join(data_path, "train")
            for class_name, index in self.classes.items():
                train_sub_class_path = os.path.join(train_path, class_name)
                class_path_list = sorted(os.listdir(train_sub_class_path))
                class_path_list = [
                    os.path.join(train_sub_class_path, name) for name in class_path_list
                ]
                class_len = len(class_path_list)
                # Use the first 80% for training.
                train_sub_samples = class_path_list[: int(class_len * 0.8)]
                self.image_paths.extend(train_sub_samples)
                self.labels.extend([index] * len(train_sub_samples))
                print(f"Train Split - {class_name}: {len(train_sub_samples)} samples")

        elif split == "val":
            train_path = os.path.join(data_path, "train")
            for class_name, index in self.classes.items():
                train_sub_class_path = os.path.join(train_path, class_name)
                class_path_list = sorted(os.listdir(train_sub_class_path))
                class_path_list = [
                    os.path.join(train_sub_class_path, name) for name in class_path_list
                ]
                class_len = len(class_path_list)
                # Use the remaining 20% for validation.
                val_sub_samples = class_path_list[int(class_len * 0.8) :]
                self.image_paths.extend(val_sub_samples)
                self.labels.extend([index] * len(val_sub_samples))
                print(f"Val Split - {class_name}: {len(val_sub_samples)} samples")

        elif split == "test":
            test_path = os.path.join(
                data_path, "validation"
            )  # The test split uses validation images.
            for class_name, index in self.classes.items():
                test_sub_class_path = os.path.join(test_path, class_name)
                class_path_list = sorted(os.listdir(test_sub_class_path))
                class_path_list = [
                    os.path.join(test_sub_class_path, name) for name in class_path_list
                ]
                self.image_paths.extend(class_path_list)
                self.labels.extend([index] * len(class_path_list))
                print(f"Test Split - {class_name}: {len(class_path_list)} samples")

        labels_array = np.array(self.labels, dtype=np.int64)
        # Build the one-hot multi-label matrix.
        self.muti_labels = np.eye(self.num_classes)[labels_array].astype(np.float32)

        # print(f"\n{split} split - samples: {len(self.train_samples)}, classes: {self.num_classes}")
        # print(f"self.muti_labels shape: {self.muti_labels.shape}")  # Validate shape.

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # 1. Read the image.
        img_path = self.image_paths[index]
        # Read grayscale images and convert them to RGB for three-channel transforms.
        image2D = cv2.imread(img_path, 0)  # 0 selects grayscale mode.
        image2D = Image.fromarray(image2D).convert(
            "RGB"
        )  # Convert to three-channel RGB.
        image2D = self.transform(image2D)  # Apply preprocessing.

        # 2. Read multi-label targets already stored in self.muti_labels.
        muti_labels = torch.from_numpy(self.muti_labels[index])

        return image2D, muti_labels
