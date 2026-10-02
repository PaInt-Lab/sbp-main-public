import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Dict, List, Optional
import random
import os
from PIL import Image

class TaskSubset(Dataset):
    def __init__(self, dataset, classes: List[int]):
        self.dataset = dataset
        self.classes = set(classes)

        self.indices = []
        if hasattr(dataset, 'targets'):
            for i, label in enumerate(dataset.targets):
                if isinstance(label, torch.Tensor):
                    label = label.item()
                if label in self.classes:
                    self.indices.append(i)
        else:
            for i in range(len(dataset)):
                _, label = dataset[i]
                label = label.item() if isinstance(label, torch.Tensor) else label
                if label in self.classes:
                    self.indices.append(i)

    def __getitem__(self, idx):
        data, label = self.dataset[self.indices[idx]]
        label = label.item() if isinstance(label, torch.Tensor) else label
        return data, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.indices)

class MiniImageNet(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.image_dir = os.path.join(root, 'images')

        # Load ALL CSVs to get images for all classes
        csv_files = ['train.csv', 'val.csv', 'test.csv']
        data_frames = []
        for f in csv_files:
            file_path = os.path.join(root, f)
            if os.path.exists(file_path):
                data_frames.append(pd.read_csv(file_path))

        if not data_frames:
            raise RuntimeError(f"No CSV files found in {root}")

        full_df = pd.concat(data_frames, ignore_index=True)

        # Map labels to integers
        self.classes = sorted(full_df['label'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Deterministic Train/Test Split per Class
        full_df = full_df.sort_values(by=['label', 'filename'])

        indices_to_keep = []

        for label, group in full_df.groupby('label'):
            imgs = group.index.tolist()
            num_imgs = len(imgs)

            # Standard MiniImageNet has 600 images per class
            split_point = 500 if num_imgs >= 600 else int(num_imgs * 0.833)

            if self.train:
                indices_to_keep.extend(imgs[:split_point])
            else:
                indices_to_keep.extend(imgs[split_point:])

        self.df = full_df.loc[indices_to_keep].reset_index(drop=True)

        self.data = self.df['filename'].values
        self.targets = [self.class_to_idx[x] for x in self.df['label'].values]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        label = self.targets[idx]

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_miniimagenet_datasets(data_dir: str = './data/miniimagenet/split', seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.Resize(92),
        transforms.RandomResizedCrop(84),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
        transforms.RandomErasing(p=0.5),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(92),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    train_dataset = MiniImageNet(root=data_dir, transform=transform_train, train=True)
    test_dataset = MiniImageNet(root=data_dir, transform=transform_test, train=False)

    return train_dataset, test_dataset