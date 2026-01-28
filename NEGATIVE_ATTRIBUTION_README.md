# Negative Quality Attribution Extension for FineDiving TSA

## Overview

This implementation extends the original FineDiving Temporal Segmentation Attention (TSA) method with an explicit **negative quality attribution branch** that models quality-degrading temporal segments in action quality assessment.

## Key Design Principles

1. **Non-invasive Extension**: The original TSA module remains completely intact and unmodified
2. **No Frame-level Supervision**: The attribution branch learns without frame-level or step-level labels
3. **Attribution vs Attention**: Attribution is implemented as a scalar temporal signal, NOT as attention weights
4. **Temporal Normalization**: Attribution signals are normalized using softmax over the temporal dimension

## Architecture Components

### 1. NegativeAttributionHead (`models/NegativeAttribution.py`)

**Purpose**: Identifies quality-degrading temporal segments in action videos

**Input**: Temporal features from PSNet `[B, T, C]` or `[B, C, T]`

**Output**:
- `attribution`: Normalized temporal attribution signal `[B, T]` (softmax over time)
- `attribution_raw`: Raw attribution before normalization `[B, T]`

**Architecture**:
```
Input [B, T, C] or [B, C, T]
  ↓
Conv1D(C → 64) + BatchNorm + ReLU
  ↓
Conv1D(64 → 32) + BatchNorm + ReLU
  ↓
Conv1D(32 → 1) [Attribution Head]
  ↓
Softmax over time dimension
  ↓
Output [B, T]
```

### 2. NegativeQualityAggregator (`models/NegativeAttribution.py`)

**Purpose**: Aggregates negative quality contribution from temporal features using attribution weights

**Input**:
- `features`: Temporal features `[B, T, C]`
- `attribution`: Normalized attribution signal `[B, T]`

**Output**:
- `negative_contribution`: Scalar negative quality score `[B, 1]`

**Process**:
1. Weighted aggregation: `weighted_features = features * attribution.unsqueeze(-1)`
2. Temporal pooling: `aggregated = weighted_features.sum(dim=1)`
3. MLP scoring: `negative_contribution = MLP(aggregated)`
4. ReLU activation to ensure non-negative contribution

## Modified Score Aggregation

### Original TSA Score:
```python
score = delta + label_2_score
```

### Modified Score with Negative Attribution:
```python
score = (delta + label_2_score) - lambda_neg * negative_contribution
```

Where:
- `delta`: Score difference predicted by TSA
- `label_2_score`: Exemplar video score
- `lambda_neg`: Weight for negative contribution (default: 0.1)
- `negative_contribution`: Negative quality score from attribution branch

## Loss Functions

### 1. Sparsity Loss (L1)
```python
loss_sparsity = attribution.abs().mean()
```
**Purpose**: Encourages sparse attribution (few non-zero values)

### 2. Concentration Loss
```python
loss_concentration = 1.0 - topk_values.sum(dim=1).mean()
```
**Purpose**: Encourages attribution to concentrate on top-k segments (default k=5)

### 3. Smoothness Loss (L2)
```python
loss_smoothness = (attribution[:, 1:] - attribution[:, :-1])^2.mean()
```
**Purpose**: Encourages temporal smoothness in attribution

### Total Training Loss:
```python
loss = loss_aqa + loss_tas +
       lambda_sparsity * loss_sparsity +
       lambda_concentration * loss_concentration +
       lambda_smoothness * loss_smoothness
```

## Hyperparameters

Added to `FineDiving_TSA.yaml`:

```yaml
# Negative attribution hyperparameters
lambda_neg: 0.1              # Weight for negative quality contribution
lambda_sparsity: 0.01        # Weight for sparsity loss
lambda_concentration: 0.01   # Weight for concentration loss
lambda_smoothness: 0.001     # Weight for smoothness loss
```

### Tuning Guidelines:

- **lambda_neg** (0.05 - 0.2): Controls impact of negative attribution on final score
  - Higher values: More aggressive penalty for negative segments
  - Lower values: More conservative, closer to original TSA

- **lambda_sparsity** (0.001 - 0.05): Controls sparsity of attribution
  - Higher values: Sparser attribution (fewer segments identified)
  - Lower values: More distributed attribution

- **lambda_concentration** (0.001 - 0.05): Controls concentration on top segments
  - Higher values: Forces attribution to concentrate on fewer segments
  - Lower values: Allows more distributed attribution

- **lambda_smoothness** (0.0001 - 0.01): Controls temporal smoothness
  - Higher values: Smoother temporal transitions
  - Lower values: Allows sharper temporal changes

## Integration Points

### 1. Model Builder (`tools/builder.py`)
- Added `NegativeAttributionHead` and `NegativeQualityAggregator` to model initialization
- Updated optimizer to include parameters from new modules
- Updated checkpoint loading/saving to handle new modules

### 2. Training Loop (`tools/runner.py`)
- Added new modules to CUDA device placement
- Added new modules to DataParallel wrapping
- Updated all function calls to pass new modules

### 3. Forward Pass (`tools/helper.py`)
- **Training**: Computes attribution, negative contribution, and all losses
- **Testing**: Computes attribution and negative contribution for score adjustment
- Modified score computation to subtract negative contribution

## Usage

### Training:
```bash
bash train.sh TSA FineDiving 0,1
```

### Testing:
```bash
bash test.sh TSA FineDiving 0,1 ./experiments/TSA/FineDiving/default/last.pth
```

### Resume Training:
```bash
bash train.sh TSA FineDiving 0,1 --resume
```

## Implementation Details

### Feature Extraction Point:
The negative attribution branch operates on `u_fea_96_1` and `u_fea_96_2`, which are:
- Temporal features from PSNet after procedure segmentation
- Shape: `[B, T, C]` where T=96 (temporal length), C=96 (feature dimension)
- These features capture temporal dynamics before cross-attention fusion

### Why This Integration Point?
1. **Rich Temporal Information**: Features after PSNet contain procedure-aware temporal information
2. **Before Score Regression**: Allows attribution to influence score before final prediction
3. **Parallel to TSA**: Operates independently without modifying TSA pipeline

### Backward Compatibility:
- Old checkpoints can be loaded (new modules initialized randomly)
- New checkpoints include all modules
- Can disable negative attribution by setting `lambda_neg=0`

## Expected Behavior

### During Training:
- Attribution should become increasingly sparse over epochs
- Negative contribution should correlate with quality-degrading segments
- Total loss should decrease while maintaining TSA performance

### During Testing:
- Attribution identifies temporal segments that degrade quality
- Final scores are adjusted by subtracting negative contributions
- Should improve correlation with ground truth scores

## Ablation Studies

To evaluate the contribution of each component:

1. **Disable negative attribution**: Set `lambda_neg=0`
2. **Disable sparsity loss**: Set `lambda_sparsity=0`
3. **Disable concentration loss**: Set `lambda_concentration=0`
4. **Disable smoothness loss**: Set `lambda_smoothness=0`

## Visualization (Future Work)

To visualize attribution:
```python
# In evaluation mode
attribution, _ = neg_attr_head(features)
# attribution[b, t] shows importance of timestep t for sample b
# Higher values indicate more negative contribution
```

## Citation

If you use this negative attribution extension, please cite both:

1. Original FineDiving paper (CVPR 2022)
2. Your work describing the negative attribution extension

## Contact

For questions about the negative attribution extension, please refer to the implementation details in:
- `models/NegativeAttribution.py`: Core attribution modules
- `tools/helper.py`: Integration into training/testing pipeline
- `FineDiving_TSA.yaml`: Hyperparameter configuration
