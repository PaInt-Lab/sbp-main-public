import os
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import random
import torchvision
import math
from torchvision import transforms
from sbp import SBP
import copy
from PIL import Image
import pandas as pd

ICARL_CIFAR100_ORDER = [
    68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
    28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
    98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
    36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
]


class DiagnosticLogger:
    """Unified diagnostic logging for FSCIL training."""

    def __init__(self):
        self.all_stats = []

    @torch.no_grad()
    def audit_classifiers(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        seen_classes: List[int],
        base_classes: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare metrics and RETURN stats for history tracking.
        """
        model.eval()

        # 1. Collect Data
        all_feats, all_labels = [], []
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            f = model.extract_features(inputs)
            all_feats.append(f)
            all_labels.append(targets)

        X = torch.cat(all_feats)
        Y = torch.cat(all_labels)

        P = model.fc_out.weight.data[seen_classes]
        InvCov = model.fc_out.precision_matrix
        local_to_global = torch.tensor(seen_classes, device=device)

        # Pre-calculate masks for Base classes (for Forgetting metric)
        is_base = torch.tensor([y.item() in base_classes for y in Y], device=device)
        is_new = ~is_base
        results = {}

        def compute_stats(dists, name):
            preds_local = torch.argmin(dists, dim=1)
            preds_global = local_to_global[preds_local]

            # 1. Overall Accuracy
            correct = (preds_global == Y).float()
            overall_acc = correct.mean().item() * 100.0

            # 2. Base Class Accuracy
            if is_base.sum() > 0:
                base_acc = correct[is_base].mean().item() * 100.0
            else:
                base_acc = 0.0

            # 3. New Class Accuracy
            if is_new.sum() > 0:
                new_acc = correct[is_new].mean().item() * 100.0
            else:
                new_acc = 0.0

            results[name] = {
                'overall': overall_acc,
                'base': base_acc,
                'new': new_acc
            }

        # --- A. Euclidean ---
        d_euc = torch.cdist(X, P).pow(2)
        compute_stats(d_euc, "Euclidean")

        # --- B. Cosine ---
        X_n = F.normalize(X, p=2, dim=1)
        P_n = F.normalize(P, p=2, dim=1)
        d_cos = 1.0 - torch.matmul(X_n, P_n.T)
        compute_stats(d_cos, "Cosine")

        # --- C. Mahalanobis ---
        d_maha = torch.zeros((X.shape[0], P.shape[0]), device=device)
        for i in range(P.shape[0]):
            delta = X - P[i]
            m = torch.matmul(delta, InvCov)
            d_maha[:, i] = (m * delta).sum(dim=1)
        compute_stats(d_maha, "MahaNorm")

        return results


    def print_final_summary(self, tasks: List[List[int]]):
        """Print summary across all sessions."""
        print(f"\n{'='*70}")
        print("FINAL DIAGNOSTIC SUMMARY")
        print(f"{'='*70}")

        base_classes = tasks[0]

        # Track logit evolution for base classes
        print(f"\n--- Base Class Logit Evolution ---")
        print(f"{'Session':<10}", end="")
        for c in base_classes[:10]:  # Show first 10 base classes
            print(f"C{c:<7}", end="")
        print()

        for stats in self.all_stats:
            print(f"{stats['session']:<10}", end="")
            for c in base_classes[:10]:
                mean_logit = stats['logit_stats']['mean_logit_per_class'].get(c, 0)
                print(f"{mean_logit:<8.3f}", end="")
            print()

        # Track overall metrics
        print(f"\n--- Session-wise Metrics ---")
        print(f"{'Session':<10} {'Global Mean':<12} {'Std Across':<12} {'Range':<10}")
        print("-" * 50)

        for stats in self.all_stats:
            ls = stats['logit_stats']
            print(f"{stats['session']:<10} {ls['global_mean_logit']:<12.4f} "
                  f"{ls['global_std_across_classes']:<12.4f} {ls['logit_range']:<10.4f}")


# =============================================================================
# DIAGNOSTICS
# =============================================================================

class TrackTime:
    """Track compute times for all phases of training."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.times = {
            'total': 0.0,
            'base_training': 0.0,
            'incremental_training': [],
            'evaluation': [],
            'per_epoch': [],
            'data_loading': 0.0,
            'forward_pass': 0.0,
            'backward_pass': 0.0,
            'budgeter_update': 0.0,
            'prototype_computation': 0.0,
        }
        self._start_time = None
        self._phase = None

    def start(self, phase: str = None):
        self._start_time = time.time()
        self._phase = phase

    def stop(self, phase: str = None) -> float:
        if self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        phase = phase or self._phase
        if phase:
            if phase in ['incremental_training', 'evaluation', 'per_epoch']:
                self.times[phase].append(elapsed)
            else:
                self.times[phase] += elapsed
        self._start_time = None
        return elapsed

    def add(self, phase: str, elapsed: float):
        if phase in ['incremental_training', 'evaluation', 'per_epoch']:
            self.times[phase].append(elapsed)
        else:
            self.times[phase] += elapsed

    def get_summary(self) -> Dict:
        return {
            'total_sec': self.times['total'],
            'total_min': self.times['total'] / 60,
            'base_training_sec': self.times['base_training'],
            'incremental_total_sec': sum(self.times['incremental_training']),
            'incremental_avg_sec': np.mean(self.times['incremental_training']) if self.times['incremental_training'] else 0,
            'incremental_per_session': self.times['incremental_training'],
            'evaluation_total_sec': sum(self.times['evaluation']),
            'evaluation_avg_sec': np.mean(self.times['evaluation']) if self.times['evaluation'] else 0,
            'epoch_avg_sec': np.mean(self.times['per_epoch']) if self.times['per_epoch'] else 0,
        }

    def print_summary(self):
        s = self.get_summary()
        print(f"\n{'='*60}")
        print("COMPUTE TIME SUMMARY")
        print(f"{'='*60}")
        print(f"Total Time: {s['total_sec']:.2f}s ({s['total_min']:.2f} min)")
        print(f"  Base Training: {s['base_training_sec']:.2f}s")
        print(f"  Incremental Training (total): {s['incremental_total_sec']:.2f}s")
        print(f"  Incremental Training (avg/session): {s['incremental_avg_sec']:.2f}s")
        print(f"  Evaluation (total): {s['evaluation_total_sec']:.2f}s")
        print(f"  Avg Epoch Time: {s['epoch_avg_sec']:.2f}s")


class BaseExemplarDataset(Dataset):
    """Store exemplars from base classes (optional, if Softnet also does it)"""
    def __init__(self, dataset, base_classes: List[int], exemplars_per_class: int = 20, seed: int = 42):
        self.dataset = dataset
        self.base_classes = base_classes

        random.seed(seed)

        class_indices = {c: [] for c in base_classes}
        for i in range(len(dataset)):
            _, label = dataset[i]
            label = label.item() if isinstance(label, torch.Tensor) else label
            if label in base_classes:
                class_indices[label].append(i)

        self.indices = []
        for c in base_classes:
            if len(class_indices[c]) > exemplars_per_class:
                sampled = random.sample(class_indices[c], exemplars_per_class)
            else:
                sampled = class_indices[c]
            self.indices.extend(sampled)

        print(f"[Base Exemplars] Stored {len(self.indices)} exemplars from {len(base_classes)} base classes")

    def __getitem__(self, idx):
        data, label = self.dataset[self.indices[idx]]
        label = label.item() if isinstance(label, torch.Tensor) else label
        return data, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.indices)


class MahalanobisClassifier(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # The Precision Matrix (Inverse Covariance).
        self.register_buffer("precision_matrix", torch.eye(in_features))

        # Scaling factor (optional, helps with Softmax gradients)
        self.scale = 1.0

    def forward(self, x):
        """
        Calculates negative Mahalanobis distance.
        d(x, mu) = (x - mu)^T * Sigma^-1 * (x - mu)
        """
        # x: [batch, feature_dim]
        # weight (means): [num_classes, feature_dim]

        # 1. Expand dimensions for broadcasting
        batch_x = x.unsqueeze(1)
        batch_means = self.weight.unsqueeze(0)

        # 2. Difference vector (x - mu)
        diff = batch_x - batch_means

        # 3. Apply Precision Matrix (Sigma^-1)
        transformed_diff = torch.matmul(diff, self.precision_matrix)

        # 4. Compute dot product (Mahalanobis distance squared)
        dist_sq = torch.sum(diff * transformed_diff, dim=2)

        # We return negative distance because CrossEntropy minimizes loss
        return -self.scale * dist_sq

    def set_covariance(self, covariance: torch.Tensor):
        """Invert and store the covariance matrix."""
        # Add slight jitter (regularization) to ensure invertibility
        reg = 1e-5 * torch.eye(covariance.size(0), device=covariance.device)
        inv_cov = torch.inverse(covariance + reg)
        self.precision_matrix.copy_(inv_cov)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # MaxPool to reduce spatial dimensions early (required for 84x84 MiniImageNet images)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512

        self.fc_out = MahalanobisClassifier(self.feature_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def extract_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        features = self.extract_features(x)
        return self.fc_out(features)


class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_logits, teacher_logits, old_classes_indices):
        if len(old_classes_indices) == 0:
            return torch.tensor(0.0, device=student_logits.device)

        student_old = student_logits[:, old_classes_indices]
        teacher_old = teacher_logits[:, old_classes_indices]

        prob_student = F.log_softmax(student_old / self.T, dim=1)
        prob_teacher = F.softmax(teacher_old / self.T, dim=1)

        loss = F.kl_div(prob_student, prob_teacher, reduction='batchmean') * (self.T ** 2)
        return loss


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


class FewShotSubset(Dataset):
    def __init__(self, dataset, classes: List[int], k_shot: int = 5, seed: int = 42):
        self.dataset = dataset
        self.classes = list(classes)
        self.k_shot = k_shot

        random.seed(seed)

        class_indices = {c: [] for c in classes}
        if hasattr(dataset, 'targets'):
            for i, label in enumerate(dataset.targets):
                if isinstance(label, torch.Tensor):
                    label = label.item()
                if label in classes:
                    class_indices[label].append(i)
        else:
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


def create_fscil_task_schedule(num_classes: int = 100, base_classes: int = 60,
                                n_way: int = 5, num_sessions: int = 8, seed: int = 42,
                                use_icarl_order: bool = True):
    if use_icarl_order:
        print(">>>> USING ICARL/ADAGAUSS CLASS ORDER <<<<")
        class_order = list(ICARL_CIFAR100_ORDER)
    else:
        print(f">>>> USING RANDOM CLASS PERMUTATION (SEED {seed}) <<<<")
        random.seed(seed)
        class_order = list(range(num_classes))
        random.shuffle(class_order)

    # Session 0: Base Classes
    tasks = [class_order[:base_classes]]

    # Sessions 1-N: Incremental batches
    remaining = class_order[base_classes:]
    for i in range(0, len(remaining), n_way):
        if len(tasks) - 1 < num_sessions:
            tasks.append(remaining[i:i + n_way])

    return tasks


def compute_budget_per_task(num_tasks: int, initial_budget: float = 0.90):
    budgets = [initial_budget]
    remaining = 1.0 - initial_budget
    if num_tasks > 1:
        incremental = remaining / (num_tasks - 1)
        budgets.extend([incremental] * (num_tasks - 1))
    return budgets


def make_classifier_hook(old_classes: List[int]):
    def hook(grad):
        grad = grad.clone()
        grad[old_classes] = 0
        return grad
    return hook


@torch.no_grad()
def imprint_classifier_weights(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    new_classes: List[int],
    time_tracker: Optional[TrackTime] = None,
    augment_rounds: int = 1
):
    t0 = time.time()
    model.eval()

    feature_dim = model.feature_dim
    new_classes = list(new_classes)

    class_sums = {c: torch.zeros(feature_dim, device=device) for c in new_classes}
    class_counts = {c: 0 for c in new_classes}

    # Accumulate features
    for _ in range(augment_rounds):
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            feats = model.extract_features(inputs)

            for c in new_classes:
                mask = (labels == c)
                if mask.any():
                    selected = feats[mask]
                    class_sums[c] += selected.sum(dim=0)
                    class_counts[c] += mask.sum().item()

    # Average to get Centroids
    for c in new_classes:
        if class_counts[c] > 0:
            # Mahalanobis needs the actual position in feature space
            proto = class_sums[c] / class_counts[c]

            model.fc_out.weight.data[c].copy_(proto)
        else:
            print(f"  [WARNING] Class {c}: no samples found")

    if time_tracker is not None:
        time_tracker.add('prototype_computation', time.time() - t0)


@torch.no_grad()
def set_bn_hard_freeze(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False

@torch.no_grad()
def compute_logit_statistics(model, dataloader, device):
    """
    Computes the average logit magnitude for the *correct* class
    on the provided training data.
    """
    model.eval()
    total_logit = 0.0
    count = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)

        # Get the logit corresponding to the true label
        correct_logits = logits.gather(1, targets.unsqueeze(1))

        total_logit += correct_logits.sum().item()
        count += targets.size(0)

    avg_logit = total_logit / max(1, count)
    return avg_logit

def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """Returns cutmix inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    W = x.size(2)
    H = x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_base_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    budgeter: SBP,
    max_grad_norm: float = 1.0,
    time_tracker: Optional[TrackTime] = None,
    use_cutmix_mixup: bool = True,
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    budgeter.new_epoch_update()

    for inputs, targets in dataloader:
        t0 = time.time()
        inputs = inputs.to(device)
        targets = targets.to(device)
        if time_tracker:
            time_tracker.add('data_loading', time.time() - t0)

        # Apply Mixup / CutMix
        r = np.random.rand()
        if use_cutmix_mixup and r < 0.5:
            if r < 0.25:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0, device=device)
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha=1.0, device=device)
        else:
            targets_a, targets_b, lam = targets, targets, 1.0

        optimizer.zero_grad(set_to_none=True)

        t0 = time.time()
        logits = model(inputs)

        if lam < 1.0:
            loss = mixup_criterion(F.cross_entropy, logits, targets_a, targets_b, lam)
        else:
            loss = F.cross_entropy(logits, targets)

        if time_tracker:
            time_tracker.add('forward_pass', time.time() - t0)

        t0 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if time_tracker:
            time_tracker.add('backward_pass', time.time() - t0)

        _, preds = logits.max(1)
        total += targets.size(0)
        correct += (lam * preds.eq(targets_a).sum().float() + (1 - lam) * preds.eq(targets_b).sum().float()).item()
        total_loss += loss.item()

    return total_loss / len(dataloader), 100.0 * correct / total


def train_incremental_iteration(
    model: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    distill_criterion: DistillationLoss,
    seen_classes: List[int],
    current_task_classes: List[int],
    budgeter: SBP,
    num_iterations: int,
    kd_lambda: float = 1.0,
    max_grad_norm: float = 1.0,
    time_tracker: Optional[TrackTime] = None,
    use_cutmix_mixup: bool = True,
):
    model.eval()
    teacher.eval()

    old_classes = [c for c in seen_classes if c not in current_task_classes]

    iteration = 0
    total_loss = 0.0
    total_ce_loss = 0.0
    total_kd_loss = 0.0

    while iteration < num_iterations:
        if iteration % len(dataloader) == 0:
            print(f"\nStarting pseudo-epoch at iteration {iteration}")
            budgeter.new_epoch_update()

        for inputs, targets in dataloader:
            if iteration >= num_iterations:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Apply Mixup / CutMix
            r = np.random.rand()
            if use_cutmix_mixup and r < 0.5:
                if r < 0.25:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0, device=device)
                else:
                    inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha=1.0, device=device)
            else:
                targets_a, targets_b, lam = targets, targets, 1.0

            optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)

            if lam < 1.0:
                ce_loss = mixup_criterion(F.cross_entropy, logits, targets_a, targets_b, lam)
            else:
                ce_loss = F.cross_entropy(logits, targets)

            kd_loss = torch.tensor(0.0, device=device)
            if len(old_classes) > 0:
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
                kd_loss = distill_criterion(logits, teacher_logits, old_classes)

            loss = ce_loss + kd_lambda * kd_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_kd_loss += kd_loss.item()

            iteration += 1

            if iteration % 10 == 0:
                avg_loss = total_loss / iteration
                avg_ce = total_ce_loss / iteration
                avg_kd = total_kd_loss / iteration
                print(f"  Iter {iteration}/{num_iterations}: "
                      f"Loss={avg_loss:.4f} (CE={avg_ce:.4f}, KD={avg_kd:.4f})")

    return total_loss / num_iterations

@torch.no_grad()
def update_global_covariance(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    print("[Mahalanobis] Calculating Within-Class Covariance Matrix...")
    model.eval()

    all_features = []
    all_labels = []

    # 1. Collect features AND labels
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        feats = model.extract_features(inputs)
        all_features.append(feats)
        all_labels.append(targets.to(device))

    all_features = torch.cat(all_features, dim=0) # [N, 512]
    all_labels = torch.cat(all_labels, dim=0)     # [N]

    # 2. Compute Class Means
    unique_classes = torch.unique(all_labels)
    class_means = torch.zeros_like(all_features)

    # We map every sample to its specific class mean
    for c in unique_classes:
        mask = (all_labels == c)
        class_mean = all_features[mask].mean(dim=0)
        class_means[mask] = class_mean

    # 3. Center features by their class mean (removing inter-class variance)
    centered_features = all_features - class_means

    # 4. Calculate Covariance: (X^T X) / (N-1)
    # Shape: [512, 512]
    covariance = torch.matmul(centered_features.t(), centered_features) / (all_features.shape[0] - 1)

    # 5. Regularization (Shrinkage)
    reg_factor = 1e-4
    covariance = covariance + reg_factor * torch.eye(covariance.size(0), device=device)

    # 6. Update the classifier
    model.fc_out.set_covariance(covariance)
    print(f"[Mahalanobis] Covariance updated (Within-Class). Trace: {covariance.trace().item():.4f}")

@torch.no_grad()
def compensate_drift(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    old_classes: List[int],
    budgeter: Optional[SBP] = None
):
    print(f"\n[Drift Compensation] Calculating drift using current data...")
    student.eval()
    teacher.eval()

    total_drift = torch.zeros(student.feature_dim, device=device)
    count = 0

    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        feat_student = student.extract_features(inputs)
        feat_teacher = teacher.extract_features(inputs)
        batch_drift = feat_student - feat_teacher
        total_drift += batch_drift.sum(dim=0)
        count += inputs.size(0)

    mean_drift = total_drift / count

    if budgeter is not None:
        target_layer = 'layer4.1.conv2.weight'

        if target_layer in budgeter.frozen_mask and target_layer in budgeter.assigned_mask:
            frozen_mask = budgeter.frozen_mask[target_layer].float().to(device)
            assigned_mask = budgeter.assigned_mask[target_layer].float().to(device)
            bleed_factor = 1.0

            scaling_vector = frozen_mask + (assigned_mask * bleed_factor)

            mean_drift = mean_drift * scaling_vector

            print(f"  [Budget Mask] Applied SCALED compensation.")
            print(f"  [Budget Mask] Frozen Dims: 100% | Assigned Dims: {bleed_factor*100}%")
        else:
            print(f"  [Warning] Masks not found. Reverting to Global Drift.")

    weights = student.fc_out.weight.data

    for c in old_classes:
        weights[c] = weights[c] + mean_drift

    print(f"  Compensated weights for {len(old_classes)} old classes.")


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device,
             seen_classes: List[int], bias_map: Dict[int, float] = None) -> float:
    model.eval()
    correct, total = 0, 0

    # Pre-convert bias map to tensor for speed
    bias_tensor = None
    if bias_map:
        # Initialize with zeros
        bias_tensor = torch.zeros(model.fc_out.out_features, device=device)
        for c, b in bias_map.items():
            bias_tensor[c] = b

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)

        if bias_tensor is not None:
            logits = logits + bias_tensor

        mask = torch.zeros(logits.size(1), device=device)
        mask[seen_classes] = 1
        logits = logits * mask - 1e9 * (1 - mask)

        _, pred = logits.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

    return 100.0 * correct / max(1, total)


@torch.no_grad()
def evaluate_per_class(model: nn.Module, dataloader: DataLoader, device: torch.device,
                       seen_classes: List[int], bias_map: Dict[int, float] = None) -> Dict[int, float]:
    model.eval()
    correct = {c: 0 for c in seen_classes}
    total = {c: 0 for c in seen_classes}

    # Pre-convert bias map to tensor
    bias_tensor = None
    if bias_map:
        bias_tensor = torch.zeros(model.fc_out.out_features, device=device)
        for c, b in bias_map.items():
            bias_tensor[c] = b

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)

        if bias_tensor is not None:
            logits = logits + bias_tensor

        mask = torch.zeros(logits.size(1), device=device)
        mask[seen_classes] = 1
        logits = logits * mask - 1e9 * (1 - mask)

        _, preds = logits.max(1)

        for i in range(targets.size(0)):
            c = targets[i].item()
            if c in seen_classes:
                total[c] += 1
                if preds[i].item() == c:
                    correct[c] += 1

    return {c: 100.0 * correct[c] / total[c] if total[c] > 0 else 0.0 for c in seen_classes}


class ReplayDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


### MAIN TRAINING LOOP FOR FSCIL ###
def run_fscil_with_budgeting(
    # FSCIL setup
    base_classes: int = 60,
    n_way: int = 5,
    k_shot: int = 5,
    num_incremental_sessions: int = 8,

    # Training config
    base_epochs: int = 100,
    incremental_iterations: int = 100,

    # SBP config
    initial_budget: float = 0.90,
    gradient_replay_ratio: float = 0.0,
    replay_weight: float = 0.0,
    reset_free_weights: bool = True,

    # Replay config
    use_base_exemplars: bool = True,
    exemplar_budget: int = 2000,

    # KD config
    kd_temperature: float = 2.0,
    kd_lambda: float = 0.9,

    max_grad_norm: float = 1.0,
    seed: int = 1993,
    use_icarl_order: bool = True,
    enable_detailed_logging: bool = True,
    enable_diagnostics: bool = True,
):
    # Initialize timing
    time_tracker = TrackTime()
    total_start = time.time()

    diagnostics = DiagnosticLogger() if enable_diagnostics else None

    print(f">>>> INITIALIZING GLOBAL SEEDS TO: {seed} <<<<")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("FSCIL WITH BUDGETING + WEIGHT IMPRINTING + KD")
    print(f"Device: {device}")
    print(f"Setup: {base_classes} base + {num_incremental_sessions} sessions × {n_way}-way {k_shot}-shot")
    print(f"SBP: initial={initial_budget*100:.0f}%")
    print(f"KD: temperature={kd_temperature}, lambda={kd_lambda}")
    print(f"Base exemplars: {'ENABLED' if use_base_exemplars else 'DISABLED'} ({exemplar_budget} total)")
    print(f"Diagnostics: {'ENABLED' if enable_diagnostics else 'DISABLED'}")
    print(f"{'='*70}\n")

    num_classes = 100
    trainset, testset = get_miniimagenet_datasets(data_dir='./data/miniimagenet/split', seed=seed)

    # Create FSCIL task schedule using the new flag
    tasks = create_fscil_task_schedule(num_classes, base_classes, n_way,
                                        num_incremental_sessions, seed,
                                        use_icarl_order=use_icarl_order)
    num_tasks = len(tasks)

    print(f"Task schedule: {[len(t) for t in tasks]}")
    print(f"Total classes: {sum(len(t) for t in tasks)}")

    # Compute budgets
    budgets = compute_budget_per_task(num_tasks, initial_budget)
    print(f"Budgets: {[f'{b*100:.2f}%' for b in budgets]}")

    # Task to class mapping (for budgeter)
    task_class_map = {i + 1: task_classes for i, task_classes in enumerate(tasks)}

    # Initialize model
    model = ResNet18(num_classes=num_classes).to(device)
    teacher = None

    # Initialize budgeter
    classifier_params = {"fc_out.weight"}
    budgeter = SBP(
        model,
        budget_slice=budgets[0],
        replay_ratio=gradient_replay_ratio,
        replay_weight=replay_weight,
        reset_assigned_weights=True,
        reset_free_weights=reset_free_weights,
        device=device,
        classifier_params=classifier_params,
        task_to_class=task_class_map,
    )

    # Storage for incremental learning
    base_exemplar_ds = None
    incremental_memory_ds = None

    # Training state
    seen_classes: List[int] = []
    session_results: List[Dict] = []
    active_hooks = []

    base_session_mean_logit = 0.0
    class_bias_map = {}

    # Store base classes for reference
    base_classes_list = tasks[0]

    audit_history = {
        'Euclidean': [],
        'Cosine': [],
        'MahaNorm': []
    }

    ### MAIN TRAINING LOOP ###
    for task_idx, task_classes in enumerate(tasks):
        is_base = (task_idx == 0)
        current_budget = budgets[task_idx]

        print(f"\n{'='*70}")
        print(f"SESSION {task_idx}: {'BASE' if is_base else 'INCREMENTAL'}")
        print(f"Classes: {task_classes}")
        print(f"Budget slice: {current_budget*100:.2f}%")
        print(f"{'='*70}")

        # Remove old classifier hooks
        for hook in active_hooks:
            hook.remove()
        active_hooks = []

        # Update seen classes
        seen_classes.extend(task_classes)

        # Update budgeter
        t0 = time.time()
        budgeter.budget_slice = current_budget
        budgeter.new_task_update(task_id=task_idx + 1, task_classes=task_classes)
        time_tracker.add('budgeter_update', time.time() - t0)

        # Register classifier hook to freeze old class weights
        if task_idx > 0:
            old_classes = [c for c in seen_classes if c not in task_classes]
            if old_classes and model.fc_out.weight.requires_grad:
                hook = model.fc_out.weight.register_hook(make_classifier_hook(old_classes))
                active_hooks.append(hook)
                print(f"[Hook] Freezing gradients for {len(old_classes)} old classes")

        if is_base:
            train_ds = TaskSubset(trainset, task_classes)
            train_loader = DataLoader(
                train_ds,
                batch_size=128,
                shuffle=True,
                num_workers=2,
                drop_last=True
            )
            print(f"Training samples: {len(train_ds)}")

        else:
            current_fs_ds = FewShotSubset(
                trainset,
                task_classes,
                k_shot=k_shot,
                seed=seed + task_idx
            )
            print(f"Current few-shot samples: {len(current_fs_ds)} ({k_shot} per class)")

            datasets_for_training = [current_fs_ds]
            if base_exemplar_ds is not None:
                datasets_for_training.append(base_exemplar_ds)
            if incremental_memory_ds is not None:
                datasets_for_training.append(incremental_memory_ds)

            train_ds = ConcatDataset(datasets_for_training)

            train_loader = DataLoader(
                train_ds,
                batch_size=min(64, len(train_ds)),
                shuffle=True,
                num_workers=2,
                drop_last=False,
            )

            replay_samples_to_add = []
            found_labels = set()
            for i in range(len(current_fs_ds)):
                image, label = current_fs_ds[i]
                if label.item() not in found_labels:
                    replay_samples_to_add.append((image, label))
                    found_labels.add(label.item())
                if len(found_labels) == len(task_classes):
                    break

            print(f"[Memory] Will store {len(replay_samples_to_add)} samples for future replay")

        ### TRAINING LOOP ###
        if is_base:
            time_tracker.start('base_training')

            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=base_epochs)

            print(f"\nTraining for {base_epochs} epochs")
            for epoch in range(base_epochs):
                tr_loss, tr_acc = train_base_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    budgeter=budgeter,
                    max_grad_norm=max_grad_norm,
                    time_tracker=time_tracker,
                )
                scheduler.step()

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1:3d}/{base_epochs}: Loss={tr_loss:.4f}, Acc={tr_acc:.2f}%")

            update_global_covariance(model, train_loader, device)

            print("[Refinement] Re-aligning Base Class Prototypes with feature centroids...")
            imprint_classifier_weights(
                model=model,
                dataloader=train_loader, # Use the training loader to get centroids
                device=device,
                new_classes=task_classes, # In session 0, these are the base classes
                time_tracker=time_tracker,
                augment_rounds=1 # 1 round is sufficient for base data
            )

            print("\n[BN] Freezing BatchNorm statistics")
            set_bn_hard_freeze(model)

            if use_base_exemplars:
                # exemplars_per_class = exemplar_budget // len(task_classes)
                # for replay of 5 samples per class
                exemplars_per_class = 0
                base_exemplar_ds = BaseExemplarDataset(
                    trainset,
                    base_classes=task_classes,
                    exemplars_per_class=exemplars_per_class,
                    seed=seed
                )

            teacher = copy.deepcopy(model).eval()
            for param in teacher.parameters():
                param.requires_grad = False
            print("[KNOWLEDGE DISTILLATION] Created frozen teacher model")

            print("[Bias Correction] Calculating base class logit stats...")
            base_session_mean_logit = compute_logit_statistics(model, train_loader, device)
            print(f"  Base Mean Logit: {base_session_mean_logit:.4f}")

            time_tracker.stop('base_training')

        else:
            # INCREMENTAL SESSION
            time_tracker.start('incremental_training')

            # KD LOGIC
            # 1. Calculate ratios
            num_old_classes = len(seen_classes) - len(task_classes)
            adaptive_factor = math.sqrt(num_old_classes / base_classes) * 0.5

            # 2. Adapative Lambda: Increases as we have more history to protect
            current_kd_lambda = kd_lambda * adaptive_factor

            # 3. Adaptive Temperature: Decays slightly to sharpen logic as space gets crowded
            current_kd_temp = kd_temperature * (0.99 ** task_idx)

            # 4. Re-initialize criterion with new temperature
            distill_criterion = DistillationLoss(temperature=current_kd_temp)

            print(f"[Adaptive KD] Session {task_idx}: Lambda={current_kd_lambda:.3f}, Temp={current_kd_temp:.3f}")

            # Loader for imprinting (uses training transforms for augmentation)
            imprint_loader = DataLoader(
                current_fs_ds,
                batch_size=64,
                shuffle=True,
                num_workers=2,
            )

            print(f"\n[Imprinting] Initializing weights for new classes (10 rounds)...")
            imprint_classifier_weights(
                model=model,
                dataloader=imprint_loader,
                device=device,
                new_classes=task_classes,
                time_tracker=time_tracker,
                augment_rounds=60
            )

            # Optimizer: Using the "Goldilocks" LR (0.0025)
            optimizer = optim.SGD(model.parameters(), lr=0.0025, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=incremental_iterations)

            print(f"\nTraining for {incremental_iterations} iterations")
            train_incremental_iteration(
                model=model,
                teacher=teacher,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                distill_criterion=distill_criterion, # Pass the new adaptive criterion
                seen_classes=seen_classes,
                current_task_classes=task_classes,
                budgeter=budgeter,
                num_iterations=incremental_iterations,
                kd_lambda=current_kd_lambda,         # Pass the new adaptive lambda
                max_grad_norm=max_grad_norm,
                time_tracker=time_tracker,
            )

            # We use the old 'teacher' (start of session) vs current 'model' (end of session)
            old_classes_indices = [c for c in seen_classes if c not in task_classes]

            if len(old_classes_indices) > 0:
                # 1. Combine our saved exemplars into a dedicated memory loader
                memory_datasets = []
                if base_exemplar_ds is not None:
                    memory_datasets.append(base_exemplar_ds)
                if incremental_memory_ds is not None:
                    memory_datasets.append(incremental_memory_ds)

                if memory_datasets:
                    full_memory_ds = ConcatDataset(memory_datasets)
                    memory_loader = DataLoader(full_memory_ds, batch_size=64, shuffle=False)

                    # 2. Measure drift using the ACTUAL old images, not the new ones
                    print(f"\n[Drift Compensation] Measuring exact feature drift using {len(full_memory_ds)} replay exemplars...")
                    compensate_drift(
                        student=model,
                        teacher=teacher,
                        dataloader=memory_loader,
                        device=device,
                        old_classes=old_classes_indices,
                        budgeter=budgeter
                    )
                else:
                    compensate_drift(
                        student=model,
                        teacher=teacher,
                        dataloader=imprint_loader,
                        device=device,
                        old_classes=old_classes_indices,
                        budgeter=budgeter
                    )

            # We re-calculate the prototypes for the new classes
            print(f"\n[Refinement] Re-imprinting new classes after training (50 rounds)...")
            imprint_classifier_weights(
                model=model,
                dataloader=imprint_loader,
                device=device,
                new_classes=task_classes,
                time_tracker=time_tracker,
                augment_rounds=140
            )

            print(f"\n[Bias Correction] Calculating Task {task_idx} logit stats...")
            # Use imprint_loader (training data for this task)
            current_task_mean_logit = compute_logit_statistics(model, imprint_loader, device)

            bias_val = base_session_mean_logit - current_task_mean_logit

            print(f"  Task Mean: {current_task_mean_logit:.4f} | Target (Base): {base_session_mean_logit:.4f}")
            print(f"  Applying Bias: {bias_val:+.4f} to classes {task_classes}")

            # Apply bias specifically to the classes of this task
            for c in task_classes:
                class_bias_map[c] = bias_val * 0.8

            # 3. Update the teacher for the next round
            teacher = copy.deepcopy(model).eval()
            for param in teacher.parameters():
                param.requires_grad = False

            time_tracker.stop('incremental_training')

            if not is_base:
                new_replay_ds = ReplayDataset(replay_samples_to_add)
                if incremental_memory_ds is None:
                    incremental_memory_ds = new_replay_ds
                else:
                    incremental_memory_ds = ConcatDataset([incremental_memory_ds, new_replay_ds])
                print(f"[Memory] Total replay samples: {len(incremental_memory_ds)}")

        ### EVALUATION ###
        time_tracker.start('evaluation')

        eval_ds = TaskSubset(testset, seen_classes)
        eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False, num_workers=2)

        # Perform the evaluation with the bias map.
        overall_acc = evaluate(model, eval_loader, device, seen_classes, bias_map=class_bias_map)
        per_class_acc = evaluate_per_class(model, eval_loader, device, seen_classes, bias_map=class_bias_map)

        time_tracker.stop('evaluation')

        # Compute session metrics
        base_acc = np.mean([per_class_acc.get(c, 0) for c in base_classes_list])
        new_acc = np.mean([per_class_acc.get(c, 0) for c in task_classes])

        result = {
            'session': task_idx,
            'is_base': is_base,
            'overall_acc': overall_acc,
            'base_classes_acc': base_acc,
            'new_classes_acc': new_acc,
            'per_class_acc': per_class_acc,
            'num_classes': len(seen_classes),
        }
        session_results.append(result)

        print(f"\n--- Session {task_idx} Results ---")
        print(f"Overall: {overall_acc:.2f}% | Base: {base_acc:.2f}% | New: {new_acc:.2f}%")

        if enable_detailed_logging and not is_base:
            print(f"New class accuracies: {[f'{per_class_acc[c]:.1f}' for c in task_classes]}")

    time_tracker.times['total'] = time.time() - total_start

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    print("\nSession-wise Accuracy:")
    print("-" * 60)
    print(f"{'Session':<10} {'Type':<12} {'Overall':<10} {'Base':<10} {'New':<10}")
    print("-" * 60)
    for r in session_results:
        session_type = "BASE" if r['is_base'] else f"INC-{r['session']}"
        print(f"{r['session']:<10} {session_type:<12} {r['overall_acc']:<10.2f} "
              f"{r['base_classes_acc']:<10.2f} {r['new_classes_acc']:<10.2f}")

    final_acc = session_results[-1]['overall_acc']
    base_session_acc = session_results[0]['overall_acc']
    performance_drop = base_session_acc - final_acc
    avg_acc = np.mean([r['overall_acc'] for r in session_results])

    initial_base_acc = session_results[0]['base_classes_acc']
    final_base_acc = session_results[-1]['base_classes_acc']
    forgetting = initial_base_acc - final_base_acc

    print(f"\n--- Key Metrics ---")
    print(f"Final Accuracy (all {len(seen_classes)} classes): {final_acc:.2f}%")
    print(f"Base Session Accuracy: {base_session_acc:.2f}%")
    print(f"Performance Drop: {performance_drop:.2f}%")
    print(f"Average Session Accuracy: {avg_acc:.2f}%")
    print(f"Forgetting (Base Classes): {forgetting:.2f}%")

    print(f"\n{'='*70}")
    print("AUDIT METRICS: SESSION-WISE BREAKDOWN")
    print(f"{'='*70}")

    for metric_name in ['Euclidean', 'Cosine', 'MahaNorm']:
        history = audit_history[metric_name]

        print(f"\n>>> Metric: {metric_name}")
        print("-" * 60)
        print(f"{'Session':<10} {'Type':<12} {'Overall':<10} {'Base':<10} {'New':<10}")
        print("-" * 60)

        for i, stats in enumerate(history):
            session_type = "BASE" if i == 0 else f"INC-{i}"

            # Retrieve stats
            ov = stats.get('overall', 0.0)
            ba = stats.get('base', 0.0)
            nw = stats.get('new', 0.0)

            print(f"{i:<10} {session_type:<12} {ov:<10.2f} {ba:<10.2f} {nw:<10.2f}")

    for metric_name in ['Euclidean', 'Cosine', 'MahaNorm']:
        history = audit_history[metric_name]

        # Calculate the 5 Key Metrics
        final_acc = history[-1]['overall']
        base_sess_acc = history[0]['overall'] # Session 0 Overall
        perf_drop = base_sess_acc - final_acc
        avg_acc = np.mean([h['overall'] for h in history])

        initial_base_acc = history[0]['base']
        final_base_acc = history[-1]['base']
        audit_forgetting = initial_base_acc - final_base_acc

        print(f"\n--- Key Metrics ({metric_name}) ---")
        print(f"Final Accuracy (all {len(seen_classes)} classes): {final_acc:.2f}%")
        print(f"Base Session Accuracy: {base_sess_acc:.2f}%")
        print(f"Performance Drop: {perf_drop:.2f}%")
        print(f"Average Session Accuracy: {avg_acc:.2f}%")
        print(f"Forgetting (Base Classes): {audit_forgetting:.2f}%")

    #Print final diagnostic summary
    if diagnostics is not None:
        diagnostics.print_final_summary(tasks)

    time_tracker.print_summary()

    return {
        'session_results': session_results,
        'final_acc': final_acc,
        'base_session_acc': base_session_acc,
        'performance_drop': performance_drop,
        'avg_acc': avg_acc,
        'forgetting': forgetting,
        'timing': time_tracker.get_summary(),
        'diagnostics': diagnostics.all_stats if diagnostics else None,
    }


def main():
    """Run FSCIL experiment."""

    # Generate a random seed based on the current system time
    dynamic_seed = int(time.time()) % 100000
    print(f"Starting run with RANDOM SEED: {dynamic_seed}")

    results = run_fscil_with_budgeting(
        # FSCIL setup
        base_classes=60,
        n_way=5,
        k_shot=5,
        num_incremental_sessions=8,

        # Training
        base_epochs=200,
        incremental_iterations=120,

        # SBP
        initial_budget=0.85,
        seed=dynamic_seed,
        use_icarl_order=True,
        gradient_replay_ratio=0.0,
        replay_weight=0.0,
        reset_free_weights=True,

        # Base classes (60) samples replay
        use_base_exemplars=False,
        exemplar_budget=1000,

        # Knowledge distillation
        kd_temperature=5.0,
        kd_lambda=7.0,

        # Other
        max_grad_norm=1.0,
        enable_detailed_logging=True,
        enable_diagnostics=True,
    )

    return results


if __name__ == "__main__":
    main()
