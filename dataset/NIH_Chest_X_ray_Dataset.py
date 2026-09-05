import os
import json

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image


class NIH_Chest_X_Ray_Dataset(Dataset):
    def __init__(self, root_dir, split='train'):
        """
        Args:
            root_dir (str): Dataset root directory.
            split (str): 'train', 'val', or 'test'.
        """
        self.root_dir = root_dir
        self.split = split
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
            "infiltration",
            "effusion",
            "nodule",
            "mass",
            "pleural_thickening",
            "emphysema",
            "fibrosis",
            "hernia"
        ]
        self.class_to_index = {
            "normal": 0,
            "pneumonia": 1,
            "covid": 2,
            "enlarged cardiomediastinum": 3,
            "cardiomegaly": 4,
            "lung opacity": 5,
            "lung lesion": 6,
            "edema": 7,
            "consolidation": 8,
            "atelectasis": 9,
            "pneumothorax": 10,
            "pleural effusion": 11,
            "pleural other": 12,
            "fracture": 13,
            "infiltration": 14,
            "effusion": 15,
            "nodule": 16,
            "mass": 17,
            "pleural_thickening": 18,
            "emphysema": 19,
            "fibrosis": 20,
            "hernia": 21
        }

        # Configure transforms.
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomRotation(10),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Load metadata and samples.
        self._prepare_data()

    def _prepare_data(self):
        # Read the metadata file.
        df = pd.read_csv(os.path.join(self.root_dir, 'Data_Entry_2017.csv'))

        # Map image identifiers to labels.
        image_to_labels = {}
        for _, row in df.iterrows():
            image_index = row['Image Index']
            finding_labels = row['Finding Labels']
            image_to_labels[image_index] = finding_labels

        # Read the file list for the requested split.
        if self.split == 'train':
            list_file = os.path.join(self.root_dir, 'train.txt')
        elif self.split == 'val':
            list_file = os.path.join(self.root_dir, 'valid.txt')
        elif self.split == 'test':
            list_file = os.path.join(self.root_dir, 'test_list_NIH.txt')
        else:
            raise ValueError("split must be 'train', 'val' or 'test'")

        with open(list_file, 'r') as f:
            image_files = [line.strip() for line in f.readlines()]

        self.image_paths = []
        self.labels = []

        for image_file in image_files:
            if image_file in image_to_labels:
                # Build the complete image path.
                image_path = os.path.join(self.root_dir, 'images-224/images-224', image_file)
                if os.path.exists(image_path):  # Keep only existing files.
                    # Support multi-label targets.
                    label_str = image_to_labels[image_file]
                    label_vector = torch.zeros(len(self.classes))

                    if label_str == 'No Finding':
                        # Normal samples have only the normal label active.
                        label_vector[self.class_to_index["normal"]] = 1
                    else:
                        # Multi-label case.
                        labels = label_str.split('|')
                        for label in labels:
                            # Normalize names to match class_to_index.
                            label_key = label.lower().replace('_', ' ')
                            if label_key in self.class_to_index:
                                label_vector[self.class_to_index[label_key]] = 1

                    # Filter samples containing excluded labels.
                    filtered_labels = ["pneumonia", "consolidation", "pneumothorax", "emphysema", "fibrosis", "hernia"]
                    should_filter = False
                    for label in filtered_labels:
                        if label_vector[self.class_to_index[label]] == 1:
                            should_filter = True
                            break

                    # Keep samples that contain none of the excluded labels.
                    if not should_filter:
                        self.image_paths.append(image_path)
                        self.labels.append(label_vector)

        # Store multi-label targets for client partitioning.
        self.muti_labels = torch.stack(self.labels)
        print(f"NIH Chest X-ray {self.split} samples: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        image = cv2.imread(img_path, 0)  # Read as grayscale.
        image = Image.fromarray(image)
        image = self.transform(image)

        label = self.muti_labels[idx]
        return image, label
