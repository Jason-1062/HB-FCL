import os
import json
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CheXpertDataset(Dataset):
    def __init__(self, root_dir, split, policy="zeros", use_transform=True):
        """
        Args:
            root_dir (string): Dataset root directory.
            split (string): Dataset split ('train', 'val', or 'test').
            policy (string): Policy for uncertain labels ('zeros', 'ones', or 'mean').
        """
        # Select the metadata file for the requested split.
        if split == "train":
            csv_path = os.path.join(root_dir, "train.csv")
        elif split == "val":
            csv_path = os.path.join(root_dir, "valid.csv")
        elif split == "test":
            csv_path = os.path.join(root_dir, "test_labels.csv")
        else:
            raise ValueError("Invalid split. Choose 'train', 'val', or 'test'.")

        self.df = pd.read_csv(csv_path)
        self.img_dir = root_dir
        self.policy = policy

        # Define the 14 classes.
        self.classes = [
            "normal",
            "pneumonia",
            "covid",
            "enlarged cardiomediastinum",
            "cardiomegaly",
            "lung opacity",
            "lung lesion",
            "edema",
            "consolidation",
            "atelectasis",
            "pneumothorax",
            "pleural effusion",
            "pleural other",
            "fracture",
        ]

        # Map class names to indices.
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}

        # Original CheXpert labels.
        self.chexpert_labels = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]

        # Handle uncertain labels (-1).
        for label in self.chexpert_labels:
            if self.policy == "zeros":  # Treat uncertain as absent.
                self.df[label] = self.df[label].fillna(0.0)
                self.df[label] = self.df[label].replace(-1.0, 0.0)
            elif self.policy == "ones":  # Treat uncertain as present.
                self.df[label] = self.df[label].fillna(1.0)
                self.df[label] = self.df[label].replace(-1.0, 1.0)
            elif self.policy == "mean":
                # Fill NaN and -1 values with the label mean.
                mean_val = self.df[self.df[label] != -1][label].mean(skipna=True)
                self.df[label] = self.df[label].fillna(mean_val)
                self.df[label] = self.df[label].replace(-1.0, mean_val)
            else:
                raise ValueError("Invalid policy. Choose 'zeros', 'ones', or 'mean'.")

        # Filter samples containing excluded labels.
        filtered_labels = [
            "Pneumonia",
            "Enlarged Cardiomediastinum",
            "Lung Lesion",
            "Atelectasis",
            "Pleural Other",
            "Fracture",
            "Cardiomegaly",
        ]
        mask = pd.Series([False] * len(self.df))
        for label in filtered_labels:
            mask |= self.df[label] == 1.0

        # Keep rows that do not contain excluded labels.
        self.df = self.df[~mask].reset_index(drop=True)
        print(
            f"Filtered dataset: kept {len(self.df)} samples out of original {len(mask)}"
        )

        # Data augmentation.
        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((224, 224)),
                    transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomRotation(10),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((224, 224)),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        # Build the self.muti_labels attribute.
        self.muti_labels = np.zeros((len(self.df), len(self.classes)), dtype=np.float32)
        label_mapping = {
            "No Finding": "normal",
            "Enlarged Cardiomediastinum": "enlarged cardiomediastinum",
            "Cardiomegaly": "cardiomegaly",
            "Lung Opacity": "lung opacity",
            "Lung Lesion": "lung lesion",
            "Edema": "edema",
            "Consolidation": "consolidation",
            "Pneumonia": "pneumonia",
            "Atelectasis": "atelectasis",
            "Pneumothorax": "pneumothorax",
            "Pleural Effusion": "pleural effusion",
            "Pleural Other": "pleural other",
            "Fracture": "fracture",
        }

        for i, orig_label in enumerate(self.chexpert_labels):
            if orig_label in label_mapping:
                new_label_name = label_mapping[orig_label]
                if new_label_name in self.class_to_index:
                    col_idx = self.class_to_index[new_label_name]
                    self.muti_labels[:, col_idx] = self.df[orig_label].values.astype(
                        np.float32
                    )

        # Treat all-zero label samples as normal.
        zero_label_mask = self.muti_labels.sum(axis=1) == 0
        self.muti_labels[zero_label_mask, self.class_to_index["normal"]] = 1.0

        # Build absolute image paths.
        self.image_paths = [
            os.path.join(self.img_dir, row["Path"].replace("CheXpert-v1.0-small/", ""))
            for _, row in self.df.iterrows()
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        # Read images with PIL.
        image = Image.open(img_path).convert("L")

        # Apply image transforms.
        image = self.transform(image)

        # Retrieve labels.
        label = self.muti_labels[idx]

        return image, torch.from_numpy(label)
