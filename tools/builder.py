import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import torch.optim as optim
from models import I3D_backbone
from models.PS import PSNet
from utils.misc import import_class
from torchvideotransforms import video_transforms, volume_transforms
from models import decoder_fuser
from models import MLP_score
from models.NegativeAttribution import NegativeAttributionHead, NegativeQualityAggregator
from models.PositiveAttribution import PositiveAttributionHead, PositiveQualityAggregator


def get_video_trans():
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
    train_trans, test_trans = get_video_trans()
    Dataset = import_class("datasets." + args.benchmark)
    train_dataset = Dataset(args, transform=train_trans, subset='train')
    test_dataset = Dataset(args, transform=test_trans, subset='test')
    return train_dataset, test_dataset

def model_builder(args):
    base_model = I3D_backbone(I3D_class=400)
    base_model.load_pretrain(args.pretrained_i3d_weight)
    PSNet_model = PSNet(n_channels=9)
    Decoder_vit = decoder_fuser(dim=64, num_heads=8, num_layers=3)
    posRegressor_delta = MLP_score(in_channel=64, out_channel=1)
    negRegressor_delta = MLP_score(in_channel=64, out_channel=1)

    # Check if dataset has action segmentation labels
    has_segmentation = getattr(args, 'has_segmentation', True)

    if has_segmentation:
        # Negative attribution branch for datasets with segmentation (FineDiving)
        neg_attr_head = NegativeAttributionHead(in_channels=64, hidden_dim=64, temporal_length=96)
        neg_quality_agg = NegativeQualityAggregator(feature_dim=64)
        pos_attr_head = None
        pos_quality_agg = None
    else:
        # Positive attribution branch for datasets without segmentation (MTL-AQA, AQA-7, JIGSAWS)
        pos_attr_head = PositiveAttributionHead(in_channels=64, hidden_dim=64, temporal_length=96)
        pos_quality_agg = PositiveQualityAggregator(feature_dim=64)
        neg_attr_head = None
        neg_quality_agg = None

    return base_model, PSNet_model, Decoder_vit, posRegressor_delta, negRegressor_delta, neg_attr_head, neg_quality_agg, pos_attr_head, pos_quality_agg

def build_opti_sche(base_model, psnet_model, decoder, pos_regressor_delta, neg_regressor_delta, neg_attr_head, neg_quality_agg, pos_attr_head, pos_quality_agg, args):
    param_groups = [
        {'params': base_model.parameters(), 'lr': args.base_lr * args.lr_factor},
        {'params': decoder.parameters()},
        {'params': pos_regressor_delta.parameters()},
        {'params': neg_regressor_delta.parameters()}
    ]

    # Add PSNet parameters only if using segmentation
    has_segmentation = getattr(args, 'has_segmentation', True)
    if has_segmentation:
        param_groups.append({'params': psnet_model.parameters()})

    # Add attribution module parameters based on dataset type
    if neg_attr_head is not None and neg_quality_agg is not None:
        param_groups.append({'params': neg_attr_head.parameters()})
        param_groups.append({'params': neg_quality_agg.parameters()})

    if pos_attr_head is not None and pos_quality_agg is not None:
        param_groups.append({'params': pos_attr_head.parameters()})
        param_groups.append({'params': pos_quality_agg.parameters()})

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(param_groups, lr=args.base_lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()

    scheduler = None
    return optimizer, scheduler


def resume_train(base_model, psnet_model, decoder, regressor_delta, neg_attr_head, neg_quality_agg, pos_attr_head, pos_quality_agg, optimizer, args):
    ckpt_path = os.path.join(args.experiment_path, 'last.pth')
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        return 0, 0, 0, 1000, 1000
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # parameter resume of base model
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    psnet_ckpt = {k.replace("module.", ""): v for k, v in state_dict['psnet_model'].items()}
    psnet_model.load_state_dict(psnet_ckpt)

    decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder'].items()}
    decoder.load_state_dict(decoder_ckpt)

    regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor_delta'].items()}
    regressor_delta.load_state_dict(regressor_delta_ckpt)

    # Load negative attribution modules if they exist in checkpoint
    if neg_attr_head is not None and 'neg_attr_head' in state_dict:
        neg_attr_head_ckpt = {k.replace("module.", ""): v for k, v in state_dict['neg_attr_head'].items()}
        neg_attr_head.load_state_dict(neg_attr_head_ckpt)

    if neg_quality_agg is not None and 'neg_quality_agg' in state_dict:
        neg_quality_agg_ckpt = {k.replace("module.", ""): v for k, v in state_dict['neg_quality_agg'].items()}
        neg_quality_agg.load_state_dict(neg_quality_agg_ckpt)

    # Load positive attribution modules if they exist in checkpoint
    if pos_attr_head is not None and 'pos_attr_head' in state_dict:
        pos_attr_head_ckpt = {k.replace("module.", ""): v for k, v in state_dict['pos_attr_head'].items()}
        pos_attr_head.load_state_dict(pos_attr_head_ckpt)

    if pos_quality_agg is not None and 'pos_quality_agg' in state_dict:
        pos_quality_agg_ckpt = {k.replace("module.", ""): v for k, v in state_dict['pos_quality_agg'].items()}
        pos_quality_agg.load_state_dict(pos_quality_agg_ckpt)

    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

    # parameter
    start_epoch = state_dict['epoch'] + 1
    epoch_best_aqa = state_dict['epoch_best_aqa']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']

    return start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min


def load_model(base_model, psnet_model, decoder, regressor_delta, neg_attr_head, neg_quality_agg, pos_attr_head, pos_quality_agg, args):
    ckpt_path = args.ckpts
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path,map_location='cpu')

    # parameter resume of base model
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)
    psnet_model_ckpt = {k.replace("module.", ""): v for k, v in state_dict['psnet_model'].items()}
    psnet_model.load_state_dict(psnet_model_ckpt)
    decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder'].items()}
    decoder.load_state_dict(decoder_ckpt)
    regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor_delta'].items()}
    regressor_delta.load_state_dict(regressor_delta_ckpt)

    # Load negative attribution modules if they exist in checkpoint
    if neg_attr_head is not None and 'neg_attr_head' in state_dict:
        neg_attr_head_ckpt = {k.replace("module.", ""): v for k, v in state_dict['neg_attr_head'].items()}
        neg_attr_head.load_state_dict(neg_attr_head_ckpt)

    if neg_quality_agg is not None and 'neg_quality_agg' in state_dict:
        neg_quality_agg_ckpt = {k.replace("module.", ""): v for k, v in state_dict['neg_quality_agg'].items()}
        neg_quality_agg.load_state_dict(neg_quality_agg_ckpt)

    # Load positive attribution modules if they exist in checkpoint
    if pos_attr_head is not None and 'pos_attr_head' in state_dict:
        pos_attr_head_ckpt = {k.replace("module.", ""): v for k, v in state_dict['pos_attr_head'].items()}
        pos_attr_head.load_state_dict(pos_attr_head_ckpt)

    if pos_quality_agg is not None and 'pos_quality_agg' in state_dict:
        pos_quality_agg_ckpt = {k.replace("module.", ""): v for k, v in state_dict['pos_quality_agg'].items()}
        pos_quality_agg.load_state_dict(pos_quality_agg_ckpt)

    epoch_best_aqa = state_dict['epoch_best_aqa']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']
    print('ckpts @ %d epoch(rho = %.4f, L2 = %.4f , RL2 = %.4f)' % (epoch_best_aqa - 1, rho_best,  L2_min, RL2_min))
    return