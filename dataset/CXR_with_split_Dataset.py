import os
import torch
import json
import cv2
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F


class CXR_with_split(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        Args:
            root_dir (str): Root directory containing MedicalExpert-I and MedicalExpert-II.
            split (str): Dataset split ('train' or 'test').
            test_size (float): Test-set fraction.
            random_state (int): Random seed.
        """
        self.root_dir = root_dir
        self.split = split
        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(
                        num_output_channels=3
                    ),  # Keep a consistent channel count.
                    transforms.Resize((224, 224)),  # Resize to a fixed shape.
                    transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1
                    ),  # Mild color augmentation.
                    # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Optional random crop.
                    transforms.RandomRotation(
                        10
                    ),  # Mild rotation without changing structure.
                    # transforms.RandomHorizontalFlip(p=0.5),  # Optional horizontal flip.
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],  # Neutral normalization.
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            self.dataset = ImageFolder(
                os.path.join(self.root_dir, "train"), transform=self.transform
            )
            print(f"CXR Train samples: {len(self.dataset)}")
        elif split == "val":
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((224, 224)),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            self.dataset = ImageFolder(
                os.path.join(self.root_dir, "val"), transform=self.transform
            )
            # print(self.dataset.class_to_idx)
            print(f"CXR Val samples: {len(self.dataset)}")
        elif split == "test":
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((224, 224)),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            self.dataset = ImageFolder(
                os.path.join(self.root_dir, "test"), transform=self.transform
            )
            # print(self.dataset.class_to_idx)
            print(f"CXR Test samples: {len(self.dataset)}")

        self.labels = self.dataset.targets
        self.num_classes = len(self.dataset.class_to_idx)

        self.muti_labels = F.one_hot(
            torch.tensor(self.labels), num_classes=self.num_classes
        ).float()

        self.class_to_idx = self.dataset.class_to_idx
        self.image_paths = [s[0] for s in self.dataset.samples]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        # Convert a single label to one-hot multi-label form.
        label = self.muti_labels[idx]
        return img, label
