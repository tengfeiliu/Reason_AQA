import torch
import torch.nn as nn
import torch.nn.functional as F


class PositiveAttributionHead(nn.Module):
    """
    Positive Quality Attribution Head

    Takes temporal features and outputs a scalar temporal attribution signal [B, T]
    that identifies quality-enhancing temporal segments.

    This is NOT attention - it's a temporal signal that models positive quality contribution.
    """
    def __init__(self, in_channels=64, hidden_dim=64, temporal_length=96):
        super(PositiveAttributionHead, self).__init__()

        self.temporal_length = temporal_length

        # Temporal feature processing
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        # Attribution head - outputs scalar per timestep
        self.attribution_head = nn.Conv1d(hidden_dim // 2, 1, kernel_size=1)

        # Initialize attribution head with small weights for stability
        nn.init.normal_(self.attribution_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.attribution_head.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x: Temporal features [B, T, C] or [B, C, T]

        Returns:
            attribution: Normalized temporal attribution signal [B, T]
            attribution_raw: Raw attribution before normalization [B, T]
        """
        # Ensure input is [B, C, T]
        if x.dim() == 3 and x.size(1) > x.size(2):
            # Likely [B, T, C], transpose to [B, C, T]
            x = x.transpose(1, 2)

        # Temporal convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Attribution signal [B, 1, T]
        attribution_raw = self.attribution_head(x)
        attribution_raw = attribution_raw.squeeze(1)  # [B, T]

        # Temporal normalization using softmax
        # This ensures the attribution sums to 1 over time
        attribution = F.softmax(attribution_raw, dim=1)  # [B, T]

        return attribution, attribution_raw


class PositiveQualityAggregator(nn.Module):
    """
    Aggregates positive quality contribution from temporal features
    using the attribution signal.
    """
    def __init__(self, feature_dim=64):
        super(PositiveQualityAggregator, self).__init__()

        # MLP to compute positive quality score from weighted features
        self.quality_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, features, attribution):
        """
        Args:
            features: Temporal features [B, C, T] or [B, T, C]
            attribution: Normalized attribution signal [B, T]

        Returns:
            positive_contribution: Scalar positive quality score [B, 1]
        """
        # Ensure features is [B, T, C]
        if features.dim() == 3 and features.size(1) < features.size(2):
            # [B, C, T] -> [B, T, C]
            features = features.transpose(1, 2)

        # Weighted aggregation: sum over time weighted by attribution
        # attribution: [B, T] -> [B, T, 1]
        # features: [B, T, C]
        weighted_features = features * attribution.unsqueeze(-1)  # [B, T, C]
        aggregated = weighted_features.sum(dim=1)  # [B, C]

        # Compute positive quality score
        positive_contribution = self.quality_mlp(aggregated)  # [B, 1]

        # Apply ReLU to ensure non-negative contribution
        positive_contribution = F.relu(positive_contribution)

        return positive_contribution


def compute_sparsity_loss(attribution, loss_type='l1'):
    """
    Compute sparsity loss on attribution signal.

    Args:
        attribution: Normalized attribution signal [B, T]
        loss_type: 'l1' or 'entropy'

    Returns:
        sparsity_loss: Scalar loss value
    """
    if loss_type == 'l1':
        # L1 sparsity: encourages few non-zero attributions
        return attribution.abs().mean()
    elif loss_type == 'entropy':
        # Entropy sparsity: encourages peaked distribution
        # H = -sum(p * log(p))
        eps = 1e-8
        entropy = -(attribution * torch.log(attribution + eps)).sum(dim=1).mean()
        # Normalize by max entropy (log(T))
        max_entropy = torch.log(torch.tensor(attribution.size(1), dtype=torch.float32))
        return entropy / max_entropy
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def compute_concentration_loss(attribution, k=5):
    """
    Concentration loss: encourages attribution to concentrate on few segments.

    Args:
        attribution: Normalized attribution signal [B, T]
        k: Number of top segments to concentrate on

    Returns:
        concentration_loss: Scalar loss value
    """
    # Get top-k attribution values
    topk_values, _ = torch.topk(attribution, k=k, dim=1)

    # Loss is negative sum of top-k (we want to maximize top-k)
    # Or equivalently, minimize the complement
    concentration_loss = 1.0 - topk_values.sum(dim=1).mean()

    return concentration_loss


def compute_smoothness_loss(attribution):
    """
    Smoothness loss: encourages temporal smoothness in attribution.

    Args:
        attribution: Normalized attribution signal [B, T]

    Returns:
        smoothness_loss: Scalar loss value
    """
    # Compute temporal differences
    diff = attribution[:, 1:] - attribution[:, :-1]

    # L2 smoothness
    smoothness_loss = (diff ** 2).mean()

    return smoothness_loss
