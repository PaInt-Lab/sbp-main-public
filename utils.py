import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import time

def compute_budget_per_task(num_tasks: int, initial_budget: float = 0.85):
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

def print_accuracy_matrix(matrix: np.ndarray) -> None:
    """Print R[session][task]: diagonal is 'learned', last row is 'final'."""
    num_tasks = matrix.shape[0]
    header = "sess\\task" + "".join(f"{i:>8d}" for i in range(num_tasks))
    print("\nAccuracy matrix R[session][task] (accuracy on each task's classes after each session):")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for session_index in range(num_tasks):
        cells = "".join(
            "       -" if np.isnan(matrix[session_index, i]) else f"{matrix[session_index, i]:8.2f}"
            for i in range(num_tasks)
        )
        print(f"{session_index:>9d}{cells}")
    print("-" * len(header))
    print("diagonal = accuracy right after learning; last row = accuracy after the final session")


@torch.no_grad()
def bn_freeze(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False


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

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# time tracker for all sessions (train time only, not eval)
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
