# 统一AQA训练框架

## 概述

本项目实现了统一的动作质量评估（AQA）训练框架，支持所有数据集使用相同的架构：
- **FineDiving**
- **MTL-AQA**
- **AQA-7**
- **JIGSAWS**

### 核心特性

1. **统一架构**：所有数据集使用相同的训练流程
2. **移除PSNet**：不再依赖子动作分割标签
3. **双分支归因**：同时使用正向和负向归因
4. **端到端训练**：简化的训练流程

## 架构设计

### 统一流程

```
Input Video
    ↓
I3D Backbone (Feature Extraction)
    ↓
Direct Cross-Attention (No PSNet)
    ↓
Score Regression
    ↓
Dual-Branch Attribution
    ├── Positive Branch (Quality-Enhancing Segments)
    └── Negative Branch (Quality-Degrading Segments)
    ↓
Final Score = Base Score + λ_pos * Pos_Score - λ_neg * Neg_Score
```

### 关键模块

1. **DualBranchAttribution** (`models/DualAttribution.py`)
   - 正向归因分支：识别高质量片段
   - 负向归因分支：识别低质量片段
   - 门控机制：平衡正负贡献

2. **UnifiedAQALoss** (`models/UnifiedLoss.py`)
   - 回归损失：MSE
   - 归因正则化：正交性、稀疏性、平滑性

3. **Attribution Metrics** (`utils/attribution_metrics.py`)
   - 归因-分数相关性
   - 归因多样性
   - 归因稀疏性

## 使用方法

### 训练

#### FineDiving
```bash
python unified_main.py \
    --config configs/FineDiving_Unified.yaml \
    --benchmark FineDiving \
    --experiment_path ./experiments/finediving \
    --gpu 0,1
```

#### MTL-AQA
```bash
python unified_main.py \
    --config configs/MTL_AQA_Unified.yaml \
    --benchmark MTL_AQA \
    --experiment_path ./experiments/mtl_aqa \
    --gpu 0,1
```

#### AQA-7
```bash
python unified_main.py \
    --config configs/AQA7_Unified.yaml \
    --benchmark AQA7 \
    --experiment_path ./experiments/aqa7 \
    --gpu 0,1
```

#### JIGSAWS
```bash
python unified_main.py \
    --config configs/JIGSAWS_Unified.yaml \
    --benchmark JIGSAWS \
    --experiment_path ./experiments/jigsaws \
    --gpu 0,1
```

### 恢复训练

```bash
python unified_main.py \
    --config configs/FineDiving_Unified.yaml \
    --benchmark FineDiving \
    --experiment_path ./experiments/finediving \
    --resume \
    --gpu 0,1
```

### 测试

```bash
python unified_main.py \
    --config configs/FineDiving_Unified.yaml \
    --benchmark FineDiving \
    --experiment_path ./experiments/finediving \
    --test \
    --ckpts ./experiments/finediving/best.pth \
    --gpu 0,1
```

## 配置参数

### 基础设置
- `bs_train`: 训练批次大小
- `bs_test`: 测试批次大小
- `workers`: 数据加载线程数
- `max_epoch`: 最大训练轮数

### 模型设置
- `pretrained_i3d_weight`: I3D预训练权重路径
- `frame_length`: 视频帧数
- `voter_number`: 测试时exemplar数量

### 优化器设置
- `optimizer`: 优化器类型（Adam）
- `base_lr`: 基础学习率
- `lr_factor`: backbone学习率因子
- `weight_decay`: 权重衰减

### 双分支归因超参数
- `lambda_pos`: 正向贡献权重（默认0.1）
- `lambda_neg`: 负向贡献权重（默认0.1）
- `weight_attribution`: 归因正则化总权重（默认0.1）
- `weight_orthogonality`: 正交性损失权重（默认0.1）
- `weight_sparsity`: 稀疏性损失权重（默认0.01）
- `weight_smoothness`: 平滑性损失权重（默认0.001）

## 文件结构

```
Reason_AQA/
├── unified_main.py                 # 统一训练入口
├── configs/                        # 配置文件
│   ├── FineDiving_Unified.yaml
│   ├── MTL_AQA_Unified.yaml
│   ├── AQA7_Unified.yaml
│   └── JIGSAWS_Unified.yaml
├── models/
│   ├── DualAttribution.py         # 双分支归因模块
│   └── UnifiedLoss.py             # 统一损失函数
├── tools/
│   ├── unified_builder.py         # 统一模型构建器
│   ├── unified_helper.py          # 统一训练辅助函数
│   └── unified_runner.py          # 统一训练运行器
├── utils/
│   └── attribution_metrics.py     # 归因评估指标
└── datasets/                       # 数据集加载器
    ├── FineDiving_Pair.py
    ├── MTL_AQA_Pair.py
    ├── AQA7_Pair.py
    └── JIGSAWS_Pair.py
```

## 与原框架的区别

### 原框架（分离式）
- FineDiving：使用PSNet + 负向归因
- 其他数据集：使用正向归因
- 需要子动作分割标签（仅FineDiving有）

### 新框架（统一式）
- 所有数据集：统一架构
- 移除PSNet：不需要分割标签
- 双分支归因：同时使用正向和负向
- 简化训练：单一训练流程

## 优势

1. **统一性**：所有数据集使用相同的代码和流程
2. **简洁性**：移除PSNet，减少复杂度
3. **可解释性**：双分支归因提供更丰富的解释
4. **灵活性**：易于扩展到新数据集
5. **性能**：端到端优化，更好的特征学习

## 评估指标

### 性能指标
- Spearman相关系数（rho）
- L2误差
- 相对L2误差（RL2）

### 归因指标
- 正向归因-分数相关性
- 负向归因-分数相关性
- 归因多样性
- 归因稀疏性（L1、Gini系数）

## 注意事项

1. **数据路径**：请根据实际情况修改配置文件中的数据路径
2. **GPU内存**：根据GPU内存调整批次大小
3. **超参数**：建议先使用默认超参数，再根据验证集表现调整
4. **预训练权重**：确保I3D预训练权重路径正确

## 引用

如果使用本框架，请引用相关论文。
