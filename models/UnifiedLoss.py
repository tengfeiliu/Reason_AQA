"""
Unified Loss Function for Action Quality Assessment

Combines multiple loss components:
1. Score regression (MSE)
2. Contrastive regression (pairwise)
3. Attribution regularization (orthogonality, sparsity, smoothness)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class UnifiedAQALoss(nn.Module):
    """
    Unified loss function for all AQA datasets.

    Total loss = λ_reg * L_reg + λ_attr * L_attr

    Where:
    - L_reg: Score regression loss (MSE)
    - L_attr: Attribution regularization loss
    """

    def __init__(
        self,
        weight_regression: float = 1.0,
        weight_attribution: float = 0.1,
        weight_orthogonality: float = 0.1,
        weight_sparsity: float = 0.01,
        weight_smoothness: float = 0.001,
        weight_global_contrastive: float = 0.1,
        weight_pairwise_ranking: float = 0.1,
        attribution_start_epoch: int = 20
    ):
        """
        Args:
            weight_regression: Weight for MSE regression loss
            weight_attribution: Weight for total attribution regularization
            weight_orthogonality: Weight for orthogonality loss
            weight_sparsity: Weight for sparsity loss
            weight_smoothness: Weight for smoothness loss
        """
        super().__init__()

        self.weight_regression = weight_regression
        self.weight_attribution = weight_attribution
        self.weight_orthogonality = weight_orthogonality
        self.weight_sparsity = weight_sparsity
        self.weight_smoothness = weight_smoothness
        self.weight_global_contrastive = weight_global_contrastive
        self.weight_pairwise_ranking = weight_pairwise_ranking
        self.attribution_start_epoch = attribution_start_epoch

        # Loss modules
        self.mse_loss = nn.MSELoss()
        self.globalContrastiveRegressionLoss = GlobalContrastiveRegressionLoss()
        self.PairwiseRankingLoss = PairwiseRankingLoss()

    def forward(
        self,
        predictions: torch.Tensor,  # [B]
        targets: torch.Tensor,  # [B]
        pos_features: Optional[torch.Tensor] = None,  # [B, D]
        neg_features: Optional[torch.Tensor] = None,  # [B, D]
        pos_attrs: Optional[torch.Tensor] = None,  # [B, T]
        neg_attrs: Optional[torch.Tensor] = None,  # [B, T]
        pos_scores: Optional[torch.Tensor] = None,  # [B]
        neg_scores: Optional[torch.Tensor] = None,  # [B]
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute unified total loss.

        Args:
            predictions: Predicted quality scores [B]
            targets: Ground truth quality scores [B]
            pos_features: Positive branch features [B, D]
            neg_features: Negative branch features [B, D]
            pos_attrs: Positive attribution weights [B, T]
            neg_attrs: Negative attribution weights [B, T]

        Returns:
            Dictionary containing:
                - total_loss: Weighted sum of all losses
                - regression_loss: Score regression component
                - attribution_loss: Attribution regularization component
                - orthogonality_loss: Orthogonality component
                - sparsity_loss: Sparsity component
                - smoothness_loss: Smoothness component
        """
        device = predictions.device

        # 1. Score regression loss
        loss_reg = self.mse_loss(predictions, targets)

        # 2. Attribution regularization losses
        if (pos_features is not None and neg_features is not None and
            pos_attrs is not None and neg_attrs is not None):

            # Orthogonality loss: minimize correlation between pos and neg features
            pos_norm = F.normalize(pos_scores, p=2, dim=1)
            neg_norm = F.normalize(neg_scores, p=2, dim=1)
            loss_orthogonality = torch.mean(torch.abs(torch.sum(pos_norm * neg_norm, dim=1)))

            # Sparsity loss: encourage sparse attribution patterns
            loss_sparsity = (torch.mean(torch.abs(pos_attrs)) + torch.mean(torch.abs(neg_attrs))) / 2

            # Smoothness loss: encourage temporal smoothness
            def smoothness(attrs):
                diff = attrs[:, 1:] - attrs[:, :-1]
                return (diff ** 2).mean()

            loss_smoothness = (smoothness(pos_attrs) + smoothness(neg_attrs)) / 2

            loss_constractive = self.globalContrastiveRegressionLoss(
                predictions, targets
            )

            loss_pairwiseranking = self.PairwiseRankingLoss(
                predictions, targets
            )

            # Total attribution loss
            loss_attr = (
                self.weight_orthogonality * loss_orthogonality +
                self.weight_sparsity * loss_sparsity +
                self.weight_smoothness * loss_smoothness + 
                self.weight_global_contrastive * loss_constractive +
                self.weight_pairwise_ranking * loss_pairwiseranking
            )
        else:
            loss_orthogonality = torch.tensor(0.0, device=device)
            loss_sparsity = torch.tensor(0.0, device=device)
            loss_smoothness = torch.tensor(0.0, device=device)
            loss_attr = torch.tensor(0.0, device=device)
            loss_constractive = torch.tensor(0.0, device=device)
            loss_pairwiseranking = torch.tensor(0.0, device=device)

        # 3. Compute weighted total loss
        if epoch < self.attribution_start_epoch:
            # Only regression loss in early epochs
            # print(self.weight_regression)
            total_loss = self.weight_regression * loss_reg
        else:
            total_loss = (
                self.weight_regression * loss_reg +
                self.weight_attribution * loss_attr
            )

        # Return dictionary with all components
        return {
            'total_loss': total_loss,
            'regression_loss': loss_reg,
            'attribution_loss': loss_attr,
            'orthogonality_loss': loss_orthogonality,
            'sparsity_loss': loss_sparsity,
            'smoothness_loss': loss_smoothness,
            'contrastive_loss': loss_constractive,
            'pairwise_ranking_loss': loss_pairwiseranking
        }


class ContrastiveRegressionLoss(nn.Module):
    """
    Contrastive regression loss for pairwise video comparison.

    Used in training with video pairs (query, exemplar).
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        delta_12: torch.Tensor,  # Predicted score difference (video1 - video2)
        delta_21: torch.Tensor,  # Predicted score difference (video2 - video1)
        score_1: torch.Tensor,  # Ground truth score for video1
        score_2: torch.Tensor,  # Ground truth score for video2
    ) -> torch.Tensor:
        """
        Compute contrastive regression loss.

        Args:
            delta_12: Predicted score difference [B]
            delta_21: Predicted score difference [B]
            score_1: Ground truth score for video1 [B]
            score_2: Ground truth score for video2 [B]

        Returns:
            Contrastive regression loss
        """
        # Ground truth score differences
        gt_delta_12 = score_1 - score_2
        gt_delta_21 = score_2 - score_1

        # MSE loss on both directions
        loss = self.mse_loss(delta_12, gt_delta_12) + self.mse_loss(delta_21, gt_delta_21)

        return loss


class GlobalContrastiveRegressionLoss(nn.Module):
    """
    Global contrastive regression loss using in-batch pairwise score differences.

    Enforces that predicted score differences match ground-truth differences.
    Unlike ranking loss (which only cares about ordering), this loss ensures
    the magnitude of differences is also preserved.

    For each pair (i, j):
        Loss = SmoothL1(pred_i - pred_j, target_i - target_j)
    """

    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = 'mean',
        normalize_by_batch: bool = False
    ):
        """
        Args:
            beta: Threshold for Smooth L1 loss
            reduction: 'mean' or 'sum'
            normalize_by_batch: If True, normalize by batch size instead of num_pairs
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.normalize_by_batch = normalize_by_batch

    def forward(
        self,
        predictions: torch.Tensor,  # [B]
        targets: torch.Tensor,       # [B]
    ) -> torch.Tensor:
        """
        Compute contrastive regression loss over all pairs in batch.

        Args:
            predictions: Predicted quality scores [B]
            targets: Ground truth quality scores [B]

        Returns:
            loss: Scalar contrastive regression loss
        """
        B = predictions.size(0)

        if B < 2:
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)

        # Compute pairwise differences
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # [B, B]
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # [B, B]

        # Smooth L1 loss between predicted and target differences
        contrastive_loss = F.smooth_l1_loss(
            pred_diff,
            target_diff,
            beta=self.beta,
            reduction='none'
        )  # [B, B]

        # Mask out diagonal (self-pairs)
        mask = ~torch.eye(B, dtype=torch.bool, device=predictions.device)
        contrastive_loss = contrastive_loss * mask.float()

        # Reduce
        if self.reduction == 'mean':
            if self.normalize_by_batch:
                loss = contrastive_loss.sum() / (B * B)
            else:
                num_pairs = B * (B - 1)
                loss = contrastive_loss.sum() / num_pairs
        elif self.reduction == 'sum':
            loss = contrastive_loss.sum()
        else:
            loss = contrastive_loss

        return loss


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss computed over all valid pairs in a batch.

    For each pair (i, j) where target_i > target_j, we encourage pred_i > pred_j.
    Uses margin-based ranking loss with numerical stability.

    Loss for pair (i, j): max(0, margin - (pred_i - pred_j)) if target_i > target_j

    This ensures that the model learns the relative ordering of quality scores,
    not just the absolute values.
    """

    def __init__(
        self,
        margin: float = 2.0,
        reduction: str = 'mean',
        min_score_diff: float = 1e-6
    ):
        """
        Args:
            margin: Margin for ranking loss (higher = stricter separation)
            reduction: 'mean' or 'sum'
            min_score_diff: Minimum score difference to consider valid pair
                           (avoids comparing near-identical scores)
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.min_score_diff = min_score_diff

    def forward(
        self,
        predictions: torch.Tensor,  # [B]
        targets: torch.Tensor,       # [B]
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss over all valid pairs in batch.

        Args:
            predictions: Predicted quality scores [B]
            targets: Ground truth quality scores [B]

        Returns:
            loss: Scalar ranking loss

        Algorithm:
            1. Create pairwise difference matrices for predictions and targets
            2. Identify valid pairs where |target_i - target_j| > threshold
            3. For each valid pair where target_i > target_j:
               loss += max(0, margin - (pred_i - pred_j))
            4. Average over all valid pairs
        """
        B = predictions.size(0)

        if B < 2:
            # Need at least 2 samples for pairwise comparison
            return torch.tensor(0.0, device=predictions.device)

        # Compute pairwise differences
        # pred_diff[i, j] = pred_i - pred_j
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # [B, B]

        # target_diff[i, j] = target_i - target_j
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # [B, B]

        # Create mask for valid pairs
        # Valid if |target_i - target_j| > min_score_diff
        valid_pairs = torch.abs(target_diff) > self.min_score_diff  # [B, B]

        # Create mask for pairs where target_i > target_j (should also have pred_i > pred_j)
        # sign[i, j] = 1 if target_i > target_j, -1 if target_i < target_j
        target_sign = torch.sign(target_diff)  # [B, B]

        # Ranking loss: max(0, margin - sign * pred_diff)
        # If target_i > target_j (sign=1), we want pred_i > pred_j (pred_diff > 0)
        # Loss = max(0, margin - pred_diff) encourages pred_diff > margin
        ranking_loss = F.relu(self.margin - target_sign * pred_diff)  # [B, B]

        # Apply valid pair mask and exclude diagonal (i=j)
        mask = valid_pairs & ~torch.eye(B, dtype=torch.bool, device=predictions.device)
        ranking_loss = ranking_loss * mask.float()

        # Reduce loss
        num_valid_pairs = mask.sum().float()

        if num_valid_pairs > 0:
            if self.reduction == 'mean':
                loss = ranking_loss.sum() / num_valid_pairs
            elif self.reduction == 'sum':
                loss = ranking_loss.sum()
            else:
                loss = ranking_loss
        else:
            # No valid pairs found
            loss = torch.tensor(0.0, device=predictions.device)

        return loss