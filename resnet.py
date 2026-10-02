import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class MahalanobisClassifier(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # inverse covariance
        self.register_buffer("precision_matrix", torch.eye(in_features))
        
        # Scaling factor: helps with softmax gradients
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)
    
    def forward(self, x):
        features = self.extract_features(x)
        return self.fc_out(features)
