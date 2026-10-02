import numpy as np
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Dict, List, Optional
import random

def get_cifar100_datasets(data_dir: str = './data', seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.5), 
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    return train_dataset, test_dataset


class TaskSubset(Dataset):
    def __init__(self, dataset, classes: List[int]):
        self.dataset = dataset
        self.classes = set(classes)
        
        self.indices = []
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


class FewShotSubset(Dataset):
    def __init__(self, dataset, classes: List[int], k_shot: int = 5, seed: int = 42):
        self.dataset = dataset
        self.classes = list(classes)
        self.k_shot = k_shot
        
        random.seed(seed)
        
        class_indices = {c: [] for c in classes}
        for i in range(len(dataset)):
            _, label = dataset[i]
            label = label.item() if isinstance(label, torch.Tensor) else label
            if label in classes:
                class_indices[label].append(i)
        
        self.indices = []
        for c in classes:
            if len(class_indices[c]) >= k_shot:
                sampled = random.sample(class_indices[c], k_shot)
            else:
                sampled = class_indices[c]
            self.indices.extend(sampled)
    
    def __getitem__(self, idx):
        data, label = self.dataset[self.indices[idx]]
        label = label.item() if isinstance(label, torch.Tensor) else label
        return data, torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.indices)