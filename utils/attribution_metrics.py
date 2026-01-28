"""
Attribution Evaluation Metrics

Quantitative evaluation of attribution quality without frame-level ground truth.
"""

import torch
import numpy as np
from scipy import stats
from typing import Dict, Tuple


def compute_attribution_score_correlation(
    positive_attrs: torch.Tensor,  # [N, T]
    negative_attrs: torch.Tensor,  # [N, T]
    quality_scores: torch.Tensor  # [N]
) -> Dict[str, float]:
    """
    Compute correlation between attribution strength and quality scores.

    Hypothesis:
    - Higher quality videos should have stronger positive attribution
    - Lower quality videos should have stronger negative attribution

    Args:
        positive_attrs: Positive attribution weights [N, T]
        negative_attrs: Negative attribution weights [N, T]
        quality_scores: Ground truth quality scores [N]

    Returns:
        Dictionary with correlation metrics
    """
    # Aggregate attribution per video
    pos_strength = positive_attrs.mean(dim=1).cpu().numpy()  # [N]
    neg_strength = negative_attrs.mean(dim=1).cpu().numpy()  # [N]
    quality = quality_scores.cpu().numpy()  # [N]

    # Compute correlations
    # Positive attribution should correlate positively with quality
    pos_pearson, _ = stats.pearsonr(pos_strength, quality)
    pos_spearman, _ = stats.spearmanr(pos_strength, quality)

    # Negative attribution should correlate negatively with quality
    neg_pearson, _ = stats.pearsonr(neg_strength, quality)
    neg_spearman, _ = stats.spearmanr(neg_strength, quality)

    return {
        'pos_pearson': float(pos_pearson),
        'pos_spearman': float(pos_spearman),
        'neg_pearson': float(neg_pearson),
        'neg_spearman': float(neg_spearman),
    }


def compute_attribution_diversity(
    positive_attrs: torch.Tensor,  # [N, T]
    negative_attrs: torch.Tensor  # [N, T]
) -> Dict[str, float]:
    """
    Compute diversity of attribution patterns across videos.

    Good attribution should show diverse patterns for different videos.

    Args:
        positive_attrs: Positive attribution weights [N, T]
        negative_attrs: Negative attribution weights [N, T]

    Returns:
        Dictionary with diversity metrics
    """
    N, T = positive_attrs.shape

    # Compute pairwise cosine similarities
    pos_norm = torch.nn.functional.normalize(positive_attrs, p=2, dim=1)
    neg_norm = torch.nn.functional.normalize(negative_attrs, p=2, dim=1)

    pos_sim_matrix = torch.matmul(pos_norm, pos_norm.T)  # [N, N]
    neg_sim_matrix = torch.matmul(neg_norm, neg_norm.T)  # [N, N]

    # Exclude diagonal (self-similarity)
    mask = 1 - torch.eye(N, device=positive_attrs.device)

    pos_sim_mean = (pos_sim_matrix * mask).sum() / (N * (N - 1))
    neg_sim_mean = (neg_sim_matrix * mask).sum() / (N * (N - 1))

    # Lower similarity = higher diversity
    pos_diversity = 1.0 - pos_sim_mean.item()
    neg_diversity = 1.0 - neg_sim_mean.item()

    return {
        'pos_diversity': float(pos_diversity),
        'neg_diversity': float(neg_diversity),
        'pos_similarity': float(pos_sim_mean.item()),
        'neg_similarity': float(neg_sim_mean.item()),
    }


def compute_attribution_sparsity(
    positive_attrs: torch.Tensor,  # [N, T]
    negative_attrs: torch.Tensor  # [N, T]
) -> Dict[str, float]:
    """
    Compute sparsity of attribution patterns.

    Good attribution should be sparse (focused on few key moments).

    Args:
        positive_attrs: Positive attribution weights [N, T]
        negative_attrs: Negative attribution weights [N, T]

    Returns:
        Dictionary with sparsity metrics
    """
    # L1 norm (average absolute value)
    pos_l1 = positive_attrs.abs().mean().item()
    neg_l1 = negative_attrs.abs().mean().item()

    # Gini coefficient (measure of inequality/sparsity)
    def gini_coefficient(attrs):
        # attrs: [N, T]
        sorted_attrs, _ = torch.sort(attrs.flatten())
        n = sorted_attrs.size(0)
        index = torch.arange(1, n + 1, device=attrs.device).float()
        gini = (2 * (index * sorted_attrs).sum()) / (n * sorted_attrs.sum()) - (n + 1) / n
        return gini.item()

    pos_gini = gini_coefficient(positive_attrs)
    neg_gini = gini_coefficient(negative_attrs)

    return {
        'pos_l1': float(pos_l1),
        'neg_l1': float(neg_l1),
        'pos_gini': float(pos_gini),
        'neg_gini': float(neg_gini),
    }


def compute_all_attribution_metrics(
    positive_attrs: torch.Tensor,  # [N, T]
    negative_attrs: torch.Tensor,  # [N, T]
    quality_scores: torch.Tensor  # [N]
) -> Dict[str, float]:
    """
    Compute all attribution evaluation metrics.

    Args:
        positive_attrs: Positive attribution weights [N, T]
        negative_attrs: Negative attribution weights [N, T]
        quality_scores: Ground truth quality scores [N]

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # Correlation metrics
    corr_metrics = compute_attribution_score_correlation(
        positive_attrs, negative_attrs, quality_scores
    )
    metrics.update(corr_metrics)

    # Diversity metrics
    div_metrics = compute_attribution_diversity(positive_attrs, negative_attrs)
    metrics.update(div_metrics)

    # Sparsity metrics
    sparse_metrics = compute_attribution_sparsity(positive_attrs, negative_attrs)
    metrics.update(sparse_metrics)

    return metrics
