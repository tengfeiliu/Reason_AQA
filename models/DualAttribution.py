import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DualBranchAttribution(nn.Module):
    """
    Dual-Branch Quality Attribution Network for unified AQA framework.

    Combines positive and negative attribution branches to identify both
    quality-enhancing and quality-degrading temporal segments.

    This unified module works for all datasets (FineDiving, MTL-AQA, AQA-7, JIGSAWS)
    without requiring action segmentation labels.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        temporal_length: int = 96,
        dropout: float = 0.3,
        use_gating: bool = True
    ):
        """
        Args:
            input_dim: Dimension of input temporal features
            hidden_dim: Hidden dimension for attribution networks
            temporal_length: Number of temporal frames
            dropout: Dropout rate
            use_gating: Whether to use gating mechanism to balance pos/neg
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temporal_length = temporal_length
        self.use_gating = use_gating

        # Shared feature encoder
        self.shared_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Positive attribution branch (captures excellence)
        self.positive_branch = AttributionBranch(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            branch_type='positive'
        )

        # Negative attribution branch (captures errors/deficiencies)
        self.negative_branch = AttributionBranch(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            branch_type='negative'
        )

        # Gating mechanism to balance positive/negative contributions
        if use_gating:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 2),
                nn.Softmax(dim=-1)
            )
        else:
            self.register_buffer('fixed_gate', torch.tensor([0.5, 0.5]))

    def forward(
        self,
        features: torch.Tensor,
        return_attributions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Temporal features [B, T, C] or [B, C, T]
            return_attributions: Whether to return attribution weights

        Returns:
            Dictionary containing:
                - positive_score: Positive contribution [B, 1]
                - negative_score: Negative contribution [B, 1]
                - positive_attrs: Positive attribution weights [B, T]
                - negative_attrs: Negative attribution weights [B, T]
                - gate_weights: Gating weights [B, 2] if use_gating=True
        """
        # Ensure input is [B, C, T]
        # If input is [B, T, C] where T < C, transpose it
        # We check if the second dimension matches input_dim to determine format
        if features.dim() == 3:
            if features.size(2) == self.input_dim:
                # Input is [B, T, C], transpose to [B, C, T]
                features = features.transpose(1, 2)
            # else: already in [B, C, T] format

        B = features.size(0)

        # Shared encoding
        shared_features = self.shared_encoder(features)  # [B, hidden_dim, T]

        # Positive and negative attribution
        pos_output = self.positive_branch(shared_features)
        neg_output = self.negative_branch(shared_features)

        # Gating
        if self.use_gating:
            gate_weights = self.gate(shared_features)  # [B, 2]
        else:
            gate_weights = self.fixed_gate.unsqueeze(0).expand(B, -1)

        # Prepare output dictionary
        output = {
            'positive_score': pos_output['score'],
            'negative_score': neg_output['score'],
            'positive_attrs': pos_output['attribution'],
            'negative_attrs': neg_output['attribution'],
            'gate_weights': gate_weights,
            'positive_features': pos_output['features'],
            'negative_features': neg_output['features']
        }

        return output


class AttributionBranch(nn.Module):
    """
    Single attribution branch (positive or negative).

    Uses temporal convolutions to compute attribution weights over time.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        branch_type: str  # 'positive' or 'negative'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.branch_type = branch_type

        # Temporal feature processing
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 2)
        self.relu = nn.ReLU(inplace=True)
        

        # Attribution head - outputs scalar per timestep
        self.attribution_head = nn.Conv1d(hidden_dim // 2, 1, kernel_size=1)

        # Initialize attribution head with small weights for stability
        nn.init.normal_(self.attribution_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.attribution_head.bias, 0.0)

        # Quality aggregator
        self.quality_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Input features [B, C, T]

        Returns:
            Dictionary with:
                - features: Aggregated features [B, hidden_dim]
                - score: Branch-specific score [B, 1]
                - attribution: Attribution weights [B, T]
        """
        # Temporal convolutions
        x = self.conv1(features)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # Attribution signal [B, 1, T]
        attribution_raw = self.attribution_head(x)
        attribution_raw = attribution_raw.squeeze(1)  # [B, T]

        # Temporal normalization using softmax
        attribution = F.softmax(attribution_raw, dim=1)  # [B, T]

        # Weighted aggregation: sum over time weighted by attribution
        # features: [B, C, T], attribution: [B, T]
        weighted_features = features * attribution.unsqueeze(1)  # [B, C, T]
        aggregated = weighted_features.sum(dim=2)  # [B, C]

        # Compute branch score
        score = self.quality_mlp(aggregated)  # [B, 1]

        # Apply ReLU to ensure non-negative contribution
        score = F.relu(score)

        return {
            'features': aggregated,
            'score': score,
            'attribution': attribution
        }


class AttributionRegularizer(nn.Module):
    """
    Regularization losses for dual-branch attribution learning.

    Ensures:
    1. Orthogonality: Positive and negative branches focus on different aspects
    2. Sparsity: Encourage sparse attribution patterns
    3. Smoothness: Temporal smoothness in attribution
    """

    def __init__(
        self,
        orthogonality_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        smoothness_weight: float = 0.001
    ):
        super().__init__()
        self.orthogonality_weight = orthogonality_weight
        self.sparsity_weight = sparsity_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        pos_features: torch.Tensor,
        neg_features: torch.Tensor,
        pos_attrs: torch.Tensor,
        neg_attrs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute attribution regularization losses.

        Args:
            pos_features: Positive branch features [B, D]
            neg_features: Negative branch features [B, D]
            pos_attrs: Positive attribution weights [B, T]
            neg_attrs: Negative attribution weights [B, T]

        Returns:
            Dictionary of regularization losses
        """
        # Orthogonality loss: minimize correlation between pos and neg features
        pos_norm = F.normalize(pos_features, p=2, dim=1)
        neg_norm = F.normalize(neg_features, p=2, dim=1)
        orthogonality_loss = torch.mean(torch.abs(torch.sum(pos_norm * neg_norm, dim=1)))

        # Sparsity loss: encourage sparse attribution patterns (L1 regularization)
        sparsity_loss = (torch.mean(torch.abs(pos_attrs)) + torch.mean(torch.abs(neg_attrs))) / 2

        # Smoothness loss: encourage temporal smoothness
        def smoothness(attrs):
            diff = attrs[:, 1:] - attrs[:, :-1]
            return (diff ** 2).mean()

        smoothness_loss = (smoothness(pos_attrs) + smoothness(neg_attrs)) / 2

        # Total regularization loss
        total_loss = (
            self.orthogonality_weight * orthogonality_loss +
            self.sparsity_weight * sparsity_loss +
            self.smoothness_weight * smoothness_loss
        )

        return {
            'attribution_reg_loss': total_loss,
            'orthogonality_loss': orthogonality_loss,
            'sparsity_loss': sparsity_loss,
            'smoothness_loss': smoothness_loss
        }
