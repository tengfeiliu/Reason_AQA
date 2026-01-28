"""
Unified Training Helper Functions

Unified training/testing functions for all AQA datasets without PSNet.
Uses dual-branch attribution (positive + negative) for all datasets.
"""

import os
import sys
import time
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))


def unified_network_forward_train(
    base_model, pos_decoder, neg_decoder, regressor_delta, dual_attribution,
    pred_scores, video_1, label_1_score, video_2, label_2_score,
    criterion, optimizer, epoch, batch_idx, batch_num, args
):
    """
    Unified training forward pass for all datasets.

    Uses dual-branch attribution (positive + negative) without PSNet.

    Args:
        base_model: I3D backbone
        decoder: Cross-attention decoder
        regressor_delta: Score regression head
        dual_attribution: Dual-branch attribution module
        pred_scores: List to store predicted scores
        video_1: Query video [B, C, T, H, W]
        label_1_score: Query score [B, 1]
        video_2: Exemplar video [B, C, T, H, W]
        label_2_score: Exemplar score [B, 1]
        criterion: Loss function
        optimizer: Optimizer
        epoch: Current epoch
        batch_idx: Current batch index
        batch_num: Total number of batches
        args: Arguments
    """
    start = time.time()
    optimizer.zero_grad()

    ############# I3D feature extraction #############
    com_feature_12, com_feamap_12 = base_model(video_1, video_2)
    video_1_fea = com_feature_12[:, :, :com_feature_12.shape[2] // 2]  # [B, 1024, 9]
    video_2_fea = com_feature_12[:, :, com_feature_12.shape[2] // 2:]  # [B, 1024, 9]


    ############# Dual-Branch Attribution #############
    # Compute attribution on query video (video_1)
    # DualAttribution expects [B, C, T] format for Conv1d
    attr_output_1 = dual_attribution(video_1_fea, return_attributions=True)
    pos_score_1 = attr_output_1['positive_score']
    neg_score_1 = attr_output_1['negative_score']
    pos_attrs_1 = attr_output_1['positive_attrs']
    neg_attrs_1 = attr_output_1['negative_attrs']
    pos_features_1 = attr_output_1['positive_features']
    neg_features_1 = attr_output_1['negative_features']
    gate_weights_1 = attr_output_1['gate_weights']
    # feature_1 = torch.cat([pos_features_1.unsqueeze(1), neg_features_1.unsqueeze(1)], dim=1) 
    pos_feature_1 = pos_features_1.unsqueeze(1)
    neg_features_1 = neg_features_1.unsqueeze(1)

    # Compute attribution on exemplar video (video_2)
    # DualAttribution expects [B, C, T] format for Conv1d
    attr_output_2 = dual_attribution(video_2_fea, return_attributions=True)
    pos_score_2 = attr_output_2['positive_score']
    neg_score_2 = attr_output_2['negative_score']
    pos_attrs_2 = attr_output_2['positive_attrs']
    neg_attrs_2 = attr_output_2['negative_attrs']
    pos_features_2 = attr_output_2['positive_features']
    neg_features_2 = attr_output_2['negative_features']
    gate_weights_2 = attr_output_2['gate_weights']
    # feature_2 = torch.cat([pos_features_2.unsqueeze(1), neg_features_2.unsqueeze(1)], dim=1) 
    pos_feature_2 = pos_features_2.unsqueeze(1)
    neg_features_2 = neg_features_2.unsqueeze(1)

    # Transpose from [B, C, T] to [B, T, C] for decoder
    # video_1_fea_decoder = video_1_fea.transpose(1, 2)  # [B, 9, 1024]
    # video_2_fea_decoder = video_2_fea.transpose(1, 2)  # [B, 9, 1024]

    pos_video_1_fea_decoder = pos_feature_1.transpose(1, 2)  # [B, 9, 1024]
    pos_video_2_fea_decoder = pos_feature_2.transpose(1, 2)  # [B, 9, 1024]

    neg_video_1_fea_decoder = neg_features_1.transpose(1, 2)  # [B, 9, 1024]
    neg_video_2_fea_decoder = neg_features_2.transpose(1, 2)  # [B, 9, 1024]

    ############# Direct Cross-attention (no PSNet) #############
    pos_decoder_video_12 = pos_decoder(pos_video_1_fea_decoder, pos_video_2_fea_decoder)
    pos_decoder_video_21 = pos_decoder(pos_video_2_fea_decoder, pos_video_1_fea_decoder)

    ############# Fine-grained Contrastive Regression #############
    pos_decoder_12_21 = torch.cat((pos_decoder_video_12, pos_decoder_video_21), 0)
    pos_delta = regressor_delta(pos_decoder_12_21.transpose(1, 2))
    pos_delta = pos_delta.mean(1)


    ############# Direct Cross-attention (no PSNet) #############
    neg_decoder_video_12 = neg_decoder(neg_video_1_fea_decoder, neg_video_2_fea_decoder)
    neg_decoder_video_21 = neg_decoder(neg_video_2_fea_decoder, neg_video_1_fea_decoder)

    ############# Fine-grained Contrastive Regression #############
    neg_decoder_12_21 = torch.cat((neg_decoder_video_12, neg_decoder_video_21), 0)
    neg_delta = regressor_delta(neg_decoder_12_21.transpose(1, 2))
    neg_delta = neg_delta.mean(1)



    # Get hyperparameters
    lambda_pos = getattr(args, 'lambda_pos', 0.1)
    lambda_neg = getattr(args, 'lambda_neg', 0.1)

    # Modified score: base_score + lambda_pos * pos_contribution - lambda_neg * neg_contribution
    # For contrastive regression, we compute the delta
    score_1 = pos_delta[:pos_delta.shape[0]//2] \
               - lambda_neg * neg_delta[:neg_delta.shape[0]//2]
    score_2 = pos_delta[pos_delta.shape[0]//2:] \
               - lambda_neg * neg_delta[neg_delta.shape[0]//2:]

    # Compute loss
    # Regression loss
    loss_dict = criterion(
        predictions=score_1.squeeze(),
        targets=label_1_score.squeeze(),
        pos_features=torch.cat([pos_features_1, pos_features_2], 0),
        neg_features=torch.cat([neg_features_1, neg_features_2], 0),
        pos_attrs=torch.cat([pos_attrs_1, pos_attrs_2], 0),
        neg_attrs=torch.cat([neg_attrs_1, neg_attrs_2], 0),
        pos_scores=torch.cat([pos_score_1, pos_score_2], 0),
        neg_scores=torch.cat([neg_score_1, neg_score_2], 0),
        epoch=epoch
    )

    loss = loss_dict['total_loss']
    loss.backward()
    optimizer.step()

    end = time.time()
    batch_time = end - start

    # Store predictions
    pred_scores.extend([i.item() for i in score_1.detach()])

    if batch_idx % args.print_freq == 0:
        print('[Training][%d/%d][%d/%d] \t Batch_time: %.2f \t Loss: %.4f \t '
              'Reg: %.4f \t Attr: %.4f \t lr1: %0.5f \t lr2: %0.5f'
              % (epoch, args.max_epoch, batch_idx, batch_num, batch_time,
                 loss.item(), loss_dict['regression_loss'].item(),
                 loss_dict['attribution_loss'].item(),
                 optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))


def unified_network_forward_test(
    base_model, pos_decoder, neg_decoder, regressor_delta, dual_attribution,
    pred_scores, video_1, video_2_list, label_2_score_list, args
):
    """
    Unified testing forward pass for all datasets.

    Uses dual-branch attribution (positive + negative) without PSNet.

    Args:
        base_model: I3D backbone
        decoder: Cross-attention decoder
        regressor_delta: Score regression head
        dual_attribution: Dual-branch attribution module
        pred_scores: List to store predicted scores
        video_1: Query video [B, C, T, H, W]
        video_2_list: List of exemplar videos
        label_2_score_list: List of exemplar scores
        args: Arguments
    """
    score = 0
    for video_2, label_2_score in zip(video_2_list, label_2_score_list):

        ############# I3D feature extraction #############
        com_feature_12, com_feamap_12 = base_model(video_1, video_2)
        video_1_fea = com_feature_12[:, :, :com_feature_12.shape[2] // 2]  # [B, 1024, 9]
        video_2_fea = com_feature_12[:, :, com_feature_12.shape[2] // 2:]  # [B, 1024, 9]

        # Transpose from [B, C, T] to [B, T, C] for decoder
        video_1_fea_decoder = video_1_fea.transpose(1, 2)  # [B, 9, 1024]
        video_2_fea_decoder = video_2_fea.transpose(1, 2)  # [B, 9, 1024]

        ############# Dual-Branch Attribution #############
        # Compute attribution on query video (video_1)
        # DualAttribution expects [B, C, T] format for Conv1d
        attr_output_1 = dual_attribution(video_1_fea, return_attributions=True)
        pos_score_1 = attr_output_1['positive_score']
        neg_score_1 = attr_output_1['negative_score']
        pos_attrs_1 = attr_output_1['positive_attrs']
        neg_attrs_1 = attr_output_1['negative_attrs']
        pos_features_1 = attr_output_1['positive_features']
        neg_features_1 = attr_output_1['negative_features']
        gate_weights_1 = attr_output_1['gate_weights']
        # feature_1 = torch.cat([pos_features_1.unsqueeze(1), neg_features_1.unsqueeze(1)], dim=1) 
        pos_feature_1 = pos_features_1.unsqueeze(1)
        neg_features_1 = neg_features_1.unsqueeze(1)

        # Compute attribution on exemplar video (video_2)
        # DualAttribution expects [B, C, T] format for Conv1d
        attr_output_2 = dual_attribution(video_2_fea, return_attributions=True)
        pos_score_2 = attr_output_2['positive_score']
        neg_score_2 = attr_output_2['negative_score']
        pos_attrs_2 = attr_output_2['positive_attrs']
        neg_attrs_2 = attr_output_2['negative_attrs']
        pos_features_2 = attr_output_2['positive_features']
        neg_features_2 = attr_output_2['negative_features']
        gate_weights_2 = attr_output_2['gate_weights']
        # feature_2 = torch.cat([pos_features_2.unsqueeze(1), neg_features_2.unsqueeze(1)], dim=1) 
        pos_feature_2 = pos_features_2.unsqueeze(1)
        neg_features_2 = neg_features_2.unsqueeze(1)

        # Transpose from [B, C, T] to [B, T, C] for decoder
        # video_1_fea_decoder = video_1_fea.transpose(1, 2)  # [B, 9, 1024]
        # video_2_fea_decoder = video_2_fea.transpose(1, 2)  # [B, 9, 1024]

        pos_video_1_fea_decoder = pos_feature_1.transpose(1, 2)  # [B, 9, 1024]
        pos_video_2_fea_decoder = pos_feature_2.transpose(1, 2)  # [B, 9, 1024]

        neg_video_1_fea_decoder = neg_features_1.transpose(1, 2)  # [B, 9, 1024]
        neg_video_2_fea_decoder = neg_features_2.transpose(1, 2)  # [B, 9, 1024]

        ############# Direct Cross-attention (no PSNet) #############
        pos_decoder_video_12 = pos_decoder(pos_video_1_fea_decoder, pos_video_2_fea_decoder)
        pos_decoder_video_21 = pos_decoder(pos_video_2_fea_decoder, pos_video_1_fea_decoder)

        ############# Fine-grained Contrastive Regression #############
        pos_decoder_12_21 = torch.cat((pos_decoder_video_12, pos_decoder_video_21), 0)
        pos_delta = regressor_delta(pos_decoder_12_21.transpose(1, 2))
        pos_delta = pos_delta.mean(1)


        ############# Direct Cross-attention (no PSNet) #############
        neg_decoder_video_12 = neg_decoder(neg_video_1_fea_decoder, neg_video_2_fea_decoder)
        neg_decoder_video_21 = neg_decoder(neg_video_2_fea_decoder, neg_video_1_fea_decoder)

        ############# Fine-grained Contrastive Regression #############
        neg_decoder_12_21 = torch.cat((neg_decoder_video_12, neg_decoder_video_21), 0)
        neg_delta = regressor_delta(neg_decoder_12_21.transpose(1, 2))
        neg_delta = neg_delta.mean(1)

        # Get hyperparameters
        lambda_pos = getattr(args, 'lambda_pos', 0.1)
        lambda_neg = getattr(args, 'lambda_neg', 0.1)

        # Modified score
        score = pos_delta[:pos_delta.shape[0]//2] \
               - lambda_neg * neg_delta[:neg_delta.shape[0]//2]

    pred_scores.extend([i.item() / len(video_2_list) for i in score])


def save_checkpoint(base_model, pos_decoder, neg_decoder, regressor_delta, dual_attribution,
                    optimizer, epoch, epoch_best_aqa, rho_best, L2_min, RL2_min,
                    prefix, args):
    """
    Save checkpoint for unified model.

    Args:
        base_model: I3D backbone
        decoder: Cross-attention decoder
        regressor_delta: Score regression head
        dual_attribution: Dual-branch attribution module
        optimizer: Optimizer
        epoch: Current epoch
        epoch_best_aqa: Best epoch
        rho_best: Best correlation
        L2_min: Best L2 error
        RL2_min: Best relative L2 error
        prefix: Checkpoint prefix
        args: Arguments
    """
    checkpoint = {
        'base_model': base_model.state_dict(),
        'pos_decoder': pos_decoder.state_dict(),
        'neg_decoder': neg_decoder.state_dict(),
        'regressor_delta': regressor_delta.state_dict(),
        'dual_attribution': dual_attribution.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'epoch_best_aqa': epoch_best_aqa,
        'rho_best': rho_best,
        'L2_min': L2_min,
        'RL2_min': RL2_min,
    }

    torch.save(checkpoint, os.path.join(args.experiment_path, prefix + '.pth'))


def save_outputs(pred_scores, true_scores, args):
    """Save predictions and ground truth scores."""
    save_path_pred = os.path.join(args.experiment_path, 'pred.npy')
    save_path_true = os.path.join(args.experiment_path, 'true.npy')
    np.save(save_path_pred, pred_scores)
    np.save(save_path_true, true_scores)
