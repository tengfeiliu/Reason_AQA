"""
Unified Model Builder

Builds models for unified AQA framework without PSNet.
All datasets use the same architecture with dual-branch attribution.
"""

import os
import sys
import torch
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

from models import I3D_backbone
from models import decoder_fuser
from models import MLP_score
from models.DualAttribution import DualBranchAttribution
from models.UnifiedLoss import UnifiedAQALoss
from utils.misc import import_class
from torchvideotransforms import video_transforms, volume_transforms


def get_video_trans():
    """Get video transformations for training and testing."""
    train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((200, 112)),
        video_transforms.RandomCrop(112),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_trans = video_transforms.Compose([
        video_transforms.Resize((200, 112)),
        video_transforms.CenterCrop(112),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_trans, test_trans


def dataset_builder(args):
    """Build train and test datasets."""
    train_trans, test_trans = get_video_trans()
    Dataset = import_class("datasets." + args.benchmark)
    train_dataset = Dataset(args, transform=train_trans, subset='train')
    test_dataset = Dataset(args, transform=test_trans, subset='test')
    return train_dataset, test_dataset


def unified_model_builder(args):
    """
    Build unified model for all datasets.

    Returns:
        base_model: I3D backbone
        pos_decoder: Cross-attention decoder
        neg_decoder: Cross-attention decoder
        regressor_delta: Score regression head
        dual_attribution: Dual-branch attribution module
        criterion: Unified loss function
    """
    # I3D backbone
    base_model = I3D_backbone(I3D_class=400)
    base_model.load_pretrain(args.pretrained_i3d_weight)

    # Decoder
    pos_decoder = decoder_fuser(dim=1024, num_heads=8, num_layers=3)
    neg_decoder = decoder_fuser(dim=1024, num_heads=8, num_layers=3)

    # Regressor
    pos_regressor_delta = MLP_score(in_channel=1024, out_channel=1)
    neg_regressor_delta = MLP_score(in_channel=1024, out_channel=1)

    # Dual-branch attribution
    dual_attribution = DualBranchAttribution(
        input_dim=1024,
        hidden_dim=1024,
        temporal_length=getattr(args, 'frame_length', 96),
        dropout=0.3,
        use_gating=True
    )

    # Unified loss function
    criterion = UnifiedAQALoss(
        weight_regression=1.0,
        weight_attribution=getattr(args, 'weight_attribution', 0.1),
        weight_orthogonality=getattr(args, 'weight_orthogonality', 0.1),
        weight_sparsity=getattr(args, 'weight_sparsity', 0.01),
        weight_smoothness=getattr(args, 'weight_smoothness', 0.001),
        weight_global_contrastive=getattr(args, 'weight_global_contrastive', 0.1),
        weight_pairwise_ranking=getattr(args, 'weight_pairwise_ranking', 0.1),
        attribution_start_epoch=getattr(args, 'attribution_start_epoch', 20)
    )

    return base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution, criterion


def build_opti_sche(base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution, args):
    """Build optimizer and scheduler."""
    if args.optimizer == 'Adam':
        optimizer = optim.Adam([
            {'params': base_model.parameters(), 'lr': args.base_lr * args.lr_factor},
            {'params': pos_decoder.parameters()},
            {'params': neg_decoder.parameters()},
            {'params': pos_regressor_delta.parameters()},
            {'params': neg_regressor_delta.parameters()},
            {'params': dual_attribution.parameters()}
        ], lr=args.base_lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()

    scheduler = None
    return optimizer, scheduler


def resume_train(base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution, optimizer, args):
    """Resume training from checkpoint."""
    ckpt_path = os.path.join(args.experiment_path, 'last.pth')
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        return 0, 0, 0, 1000, 1000
    print('Loading weights from %s...' % ckpt_path)

    # Load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # Load model weights
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    pos_decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['pos_decoder'].items()}
    pos_decoder.load_state_dict(pos_decoder_ckpt)

    neg_decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['neg_decoder'].items()}
    neg_decoder.load_state_dict(neg_decoder_ckpt)

    pos_regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict['pos_regressor_delta'].items()}
    pos_regressor_delta.load_state_dict(pos_regressor_delta_ckpt)

    neg_regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict['neg_regressor_delta'].items()}
    neg_regressor_delta.load_state_dict(neg_regressor_delta_ckpt)

    dual_attribution_ckpt = {k.replace("module.", ""): v for k, v in state_dict['dual_attribution'].items()}
    dual_attribution.load_state_dict(dual_attribution_ckpt)

    # Load optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

    # Load training state
    start_epoch = state_dict['epoch'] + 1
    epoch_best_aqa = state_dict['epoch_best_aqa']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']

    return start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min


def load_model(base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution, args):
    """Load model from checkpoint for testing."""
    ckpt_path = args.ckpts
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print('Loading weights from %s...' % ckpt_path)

    # Load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # Load model weights
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    pos_decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['pos_decoder'].items()}
    pos_decoder.load_state_dict(pos_decoder_ckpt)

    neg_decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['neg_decoder'].items()}
    neg_decoder.load_state_dict(neg_decoder_ckpt)

    pos_regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict['pos_regressor_delta'].items()}
    pos_regressor_delta.load_state_dict(pos_regressor_delta_ckpt)

    neg_regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict['neg_regressor_delta'].items()}
    neg_regressor_delta.load_state_dict(neg_regressor_delta_ckpt)

    dual_attribution_ckpt = {k.replace("module.", ""): v for k, v in state_dict['dual_attribution'].items()}
    dual_attribution.load_state_dict(dual_attribution_ckpt)

    # Print checkpoint info
    epoch_best_aqa = state_dict['epoch_best_aqa']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']
    print('ckpts @ %d epoch(rho = %.4f, L2 = %.4f , RL2 = %.4f)' %
          (epoch_best_aqa - 1, rho_best, L2_min, RL2_min))
    return
