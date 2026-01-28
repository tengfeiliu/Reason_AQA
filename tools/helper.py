import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import torch.nn as nn
import time
import numpy as np
from utils.misc import segment_iou, cal_tiou, seg_pool_1d, seg_pool_3d
from models.NegativeAttribution import compute_sparsity_loss, compute_concentration_loss, compute_smoothness_loss


def network_forward_train(base_model, psnet_model, decoder, regressor_delta, neg_attr_head, neg_quality_agg, pred_scores,
                          video_1, label_1_score, video_2, label_2_score, mse, optimizer, opti_flag,
                          epoch, batch_idx, batch_num, args, label_1_tas, label_2_tas, bce,
                          pred_tious_5, pred_tious_75):

    start = time.time()
    optimizer.zero_grad()

    ############# I3D featrue #############
    com_feature_12, com_feamap_12 = base_model(video_1, video_2)
    video_1_fea = com_feature_12[:,:,:com_feature_12.shape[2] // 2]
    video_2_fea = com_feature_12[:,:,com_feature_12.shape[2] // 2:]
    video_1_feamap = com_feamap_12[:,:,:com_feature_12.shape[2] // 2]
    video_2_feamap = com_feamap_12[:,:,com_feature_12.shape[2] // 2:]

    N,T,C,T_t,H_t,W_t = video_1_feamap.size()
    video_1_feamap = video_1_feamap.mean(-3)
    video_2_feamap = video_2_feamap.mean(-3)
    video_1_feamap_re = video_1_feamap.reshape(-1, T, C)
    video_2_feamap_re = video_2_feamap.reshape(-1, T, C)

    ############# Procedure Segmentation #############
    com_feature_12_u = torch.cat((video_1_fea, video_2_fea), 0)
    com_feamap_12_u = torch.cat((video_1_feamap_re, video_2_feamap_re), 0)

    u_fea_96, transits_pred = psnet_model(com_feature_12_u)
    u_feamap_96, transits_pred_map = psnet_model(com_feamap_12_u)
    u_feamap_96 = u_feamap_96.reshape(2*N, u_feamap_96.shape[1], u_feamap_96.shape[2], H_t, W_t)

    label_12_tas = torch.cat((label_1_tas, label_2_tas), 0)
    label_12_pad = torch.zeros(transits_pred.size())
    for bs in range(transits_pred.shape[0]):
        label_12_pad[bs, int(label_12_tas[bs, 0]), 0] = 1
        label_12_pad[bs, int(label_12_tas[bs, -1]), -1] = 1

    loss_tas = bce(transits_pred, label_12_pad.cuda())

    num = round(transits_pred.shape[1] / transits_pred.shape[-1])
    transits_st_ed = torch.zeros(label_12_tas.size())
    for bs in range(transits_pred.shape[0]):
        for i in range(transits_pred.shape[-1]):
            transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).cpu().item() + i * num
    label_1_tas_pred = transits_st_ed[:transits_st_ed.shape[0] // 2]
    label_2_tas_pred = transits_st_ed[transits_st_ed.shape[0] // 2:]

    ############# Procedure-aware Cross-attention #############
    u_fea_96_1 = u_fea_96[:u_fea_96.shape[0] // 2].transpose(1, 2)
    u_fea_96_2 = u_fea_96[u_fea_96.shape[0] // 2:].transpose(1, 2)

    u_feamap_96_1 = u_feamap_96[:u_feamap_96.shape[0] // 2].transpose(1, 2)
    u_feamap_96_2 = u_feamap_96[u_feamap_96.shape[0] // 2:].transpose(1, 2)

    if epoch / args.max_epoch <= args.prob_tas_threshold:
        video_1_segs = []
        for bs_1 in range(u_fea_96_1.shape[0]):
            video_1_st = int(label_1_tas[bs_1][0].item())
            video_1_ed = int(label_1_tas[bs_1][1].item())
            video_1_segs.append(seg_pool_1d(u_fea_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs = torch.cat(video_1_segs, 0).transpose(1, 2)

        video_2_segs = []
        for bs_2 in range(u_fea_96_2.shape[0]):                 
            video_2_st = int(label_2_tas[bs_2][0].item())
            video_2_ed = int(label_2_tas[bs_2][1].item())
            video_2_segs.append(seg_pool_1d(u_fea_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs = torch.cat(video_2_segs, 0).transpose(1, 2)   

        video_1_segs_map = []
        for bs_1 in range(u_feamap_96_1.shape[0]):
            video_1_st = int(label_1_tas[bs_1][0].item())
            video_1_ed = int(label_1_tas[bs_1][1].item())
            video_1_segs_map.append(seg_pool_3d(u_feamap_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs_map = torch.cat(video_1_segs_map, 0)
        video_1_segs_map = video_1_segs_map.reshape(video_1_segs_map.shape[0], video_1_segs_map.shape[1], video_1_segs_map.shape[2], -1).transpose(2, 3)
        video_1_segs_map = torch.cat([video_1_segs_map[:,:,:,i] for i in range(video_1_segs_map.shape[-1])], 2).transpose(1, 2)

        video_2_segs_map = []
        for bs_2 in range(u_fea_96_2.shape[0]):
            video_2_st = int(label_2_tas[bs_2][0].item())
            video_2_ed = int(label_2_tas[bs_2][1].item())
            video_2_segs_map.append(seg_pool_3d(u_feamap_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs_map = torch.cat(video_2_segs_map, 0)
        video_2_segs_map = video_2_segs_map.reshape(video_2_segs_map.shape[0], video_2_segs_map.shape[1], video_2_segs_map.shape[2], -1).transpose(2, 3)
        video_2_segs_map = torch.cat([video_2_segs_map[:, :, :, i] for i in range(video_2_segs_map.shape[-1])], 2).transpose(1, 2)
    else:
        video_1_segs = []
        for bs_1 in range(u_fea_96_1.shape[0]):
            video_1_st = int(label_1_tas_pred[bs_1][0].item())
            video_1_ed = int(label_1_tas_pred[bs_1][1].item())
            if video_1_st == 0:
                video_1_st = 1
            if video_1_ed == 0:
                video_1_ed = 1
            video_1_segs.append(seg_pool_1d(u_fea_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs = torch.cat(video_1_segs, 0).transpose(1, 2)   

        video_2_segs = []
        for bs_2 in range(u_fea_96_2.shape[0]):                 
            video_2_st = int(label_2_tas_pred[bs_2][0].item())
            video_2_ed = int(label_2_tas_pred[bs_2][1].item())
            if video_2_st == 0:
                video_2_st = 1
            if video_2_ed == 0:
                video_2_ed = 1
            video_2_segs.append(seg_pool_1d(u_fea_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs = torch.cat(video_2_segs, 0).transpose(1, 2)   

        video_1_segs_map = []
        for bs_1 in range(u_feamap_96_1.shape[0]):
            video_1_st = int(label_1_tas_pred[bs_1][0].item())
            video_1_ed = int(label_1_tas_pred[bs_1][1].item())
            if video_1_st == 0:
                video_1_st = 1
            if video_1_ed == 0:
                video_1_ed = 1
            video_1_segs_map.append(seg_pool_3d(u_feamap_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs_map = torch.cat(video_1_segs_map, 0)
        video_1_segs_map = video_1_segs_map.reshape(video_1_segs_map.shape[0], video_1_segs_map.shape[1], video_1_segs_map.shape[2], -1).transpose(2, 3)
        video_1_segs_map = torch.cat([video_1_segs_map[:, :, :, i] for i in range(video_1_segs_map.shape[-1])], 2).transpose(1, 2)

        video_2_segs_map = []
        for bs_2 in range(u_fea_96_2.shape[0]):
            video_2_st = int(label_2_tas_pred[bs_2][0].item())
            video_2_ed = int(label_2_tas_pred[bs_2][1].item())
            if video_2_st == 0:
                video_2_st = 1
            if video_2_ed == 0:
                video_2_ed = 1
            video_2_segs_map.append(seg_pool_3d(u_feamap_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs_map = torch.cat(video_2_segs_map, 0)
        video_2_segs_map = video_2_segs_map.reshape(video_2_segs_map.shape[0], video_2_segs_map.shape[1], video_2_segs_map.shape[2], -1).transpose(2, 3)
        video_2_segs_map = torch.cat([video_2_segs_map[:, :, :, i] for i in range(video_2_segs_map.shape[-1])], 2).transpose(1, 2)

    decoder_video_12_map_list = []
    decoder_video_21_map_list = []
    for i in range(args.step_num):
        decoder_video_12_map = decoder(video_1_segs[:, i*args.fix_size:(i+1)*args.fix_size,:],
                                                      video_2_segs_map[:, i*args.fix_size*H_t*W_t:(i+1)*args.fix_size*H_t*W_t,:])     # N,15,256/64
        decoder_video_21_map = decoder(video_2_segs[:, i*args.fix_size:(i+1)*args.fix_size,:],
                                          video_1_segs_map[:, i*args.fix_size*H_t*W_t:(i+1)*args.fix_size*H_t*W_t,:])    # N,15,256/64
        decoder_video_12_map_list.append(decoder_video_12_map)
        decoder_video_21_map_list.append(decoder_video_21_map)

    decoder_video_12_map = torch.cat(decoder_video_12_map_list, 1)
    decoder_video_21_map = torch.cat(decoder_video_21_map_list, 1)

    ############# Fine-grained Contrastive Regression #############
    decoder_12_21 = torch.cat((decoder_video_12_map, decoder_video_21_map), 0)
    delta = regressor_delta(decoder_12_21)
    delta = delta.mean(1)

    ############# Negative Quality Attribution #############
    # Compute negative attribution on query video (video_1) features
    # Use u_fea_96_1 which has shape [B, C, T] or [B, T, C]
    attribution_1, attribution_1_raw = neg_attr_head(u_fea_96_1)  # [B, T]
    neg_contribution_1 = neg_quality_agg(u_fea_96_1, attribution_1)  # [B, 1]

    # Compute negative attribution on exemplar video (video_2) features
    attribution_2, attribution_2_raw = neg_attr_head(u_fea_96_2)  # [B, T]
    neg_contribution_2 = neg_quality_agg(u_fea_96_2, attribution_2)  # [B, 1]

    # Aggregate negative contributions
    neg_contribution = torch.cat((neg_contribution_1, neg_contribution_2), 0)  # [2B, 1]

    # Sparsity loss: encourage sparse attribution
    sparsity_loss_1 = compute_sparsity_loss(attribution_1, loss_type='l1')
    sparsity_loss_2 = compute_sparsity_loss(attribution_2, loss_type='l1')
    loss_sparsity = (sparsity_loss_1 + sparsity_loss_2) / 2.0

    # Concentration loss: encourage attribution to concentrate on few segments
    concentration_loss_1 = compute_concentration_loss(attribution_1, k=5)
    concentration_loss_2 = compute_concentration_loss(attribution_2, k=5)
    loss_concentration = (concentration_loss_1 + concentration_loss_2) / 2.0

    # Smoothness loss: encourage temporal smoothness
    smoothness_loss_1 = compute_smoothness_loss(attribution_1)
    smoothness_loss_2 = compute_smoothness_loss(attribution_2)
    loss_smoothness = (smoothness_loss_1 + smoothness_loss_2) / 2.0

    # Get hyperparameters from args (with defaults)
    lambda_neg = getattr(args, 'lambda_neg', 0.1)
    lambda_sparsity = getattr(args, 'lambda_sparsity', 0.01)
    lambda_concentration = getattr(args, 'lambda_concentration', 0.01)
    lambda_smoothness = getattr(args, 'lambda_smoothness', 0.001)

    # Modified score aggregation: score_TSA - lambda * negative_contribution
    # For contrastive regression, we need to adjust the delta computation
    # Original: delta predicts score difference
    # Modified: delta - lambda * (neg_contribution_1 - neg_contribution_2)
    neg_delta = neg_contribution[:neg_contribution.shape[0]//2] - neg_contribution[neg_contribution.shape[0]//2:]

    loss_aqa = mse(delta[:delta.shape[0]//2], (label_1_score - label_2_score)) \
               + mse(delta[delta.shape[0]//2:], (label_2_score - label_1_score))

    # Total loss
    loss = loss_aqa + loss_tas + \
           lambda_sparsity * loss_sparsity + \
           lambda_concentration * loss_concentration + \
           lambda_smoothness * loss_smoothness
    loss.backward()
    optimizer.step()

    end = time.time()
    batch_time = end - start

    # Modified score: score_TSA - lambda * negative_contribution
    score = (delta[:delta.shape[0]//2].detach() + label_2_score) - lambda_neg * neg_contribution_1.detach()
    pred_scores.extend([i.item() for i in score])

    tIoU_results = []
    for bs in range(transits_pred.shape[0] // 2):
        tIoU_results.append(segment_iou(np.array(label_12_tas.squeeze(-1).cpu())[bs],
                                        np.array(transits_st_ed.squeeze(-1).cpu())[bs],
                                        args))

    tiou_thresholds = np.array([0.5, 0.75])
    tIoU_correct_per_thr = cal_tiou(tIoU_results, tiou_thresholds)
    Batch_tIoU_5 = tIoU_correct_per_thr[0]
    Batch_tIoU_75 = tIoU_correct_per_thr[1]
    pred_tious_5.extend([Batch_tIoU_5])
    pred_tious_75.extend([Batch_tIoU_75])

    if batch_idx % args.print_freq == 0:
        print('[Training][%d/%d][%d/%d] \t Batch_time: %.2f \t Batch_loss: %.4f \t '
              'lr1 : %0.5f \t lr2 : %0.5f'
              % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, loss.item(), 
                 optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))


def network_forward_test(base_model, psnet_model, decoder, regressor_delta, neg_attr_head, neg_quality_agg, pred_scores,
                         video_1, video_2_list, label_2_score_list,
                         args, label_1_tas, label_2_tas_list,
                         pred_tious_test_5, pred_tious_test_75):
    score = 0
    tIoU_results = []
    for video_2, label_2_score, label_2_tas in zip(video_2_list, label_2_score_list, label_2_tas_list):

        ############# I3D featrue #############
        com_feature_12, com_feamap_12 = base_model(video_1, video_2) 
        video_1_fea = com_feature_12[:, :, :com_feature_12.shape[2] // 2]
        video_2_fea = com_feature_12[:, :, com_feature_12.shape[2] // 2:]
        video_1_feamap = com_feamap_12[:, :, :com_feature_12.shape[2] // 2]
        video_2_feamap = com_feamap_12[:, :, com_feature_12.shape[2] // 2:]

        N, T, C, T_t, H_t, W_t = video_1_feamap.size()
        video_1_feamap = video_1_feamap.mean(-3)
        video_2_feamap = video_2_feamap.mean(-3)
        video_1_feamap_re = video_1_feamap.reshape(-1, T, C)
        video_2_feamap_re = video_2_feamap.reshape(-1, T, C)

        ############# Procedure Segmentation #############
        com_feature_12_u = torch.cat((video_1_fea, video_2_fea), 0)
        com_feamap_12_u = torch.cat((video_1_feamap_re, video_2_feamap_re), 0)

        u_fea_96, transits_pred = psnet_model(com_feature_12_u)
        u_feamap_96, transits_pred_map = psnet_model(com_feamap_12_u)
        u_feamap_96 = u_feamap_96.reshape(2 * N, u_feamap_96.shape[1], u_feamap_96.shape[2], H_t, W_t)

        label_12_tas = torch.cat((label_1_tas, label_2_tas), 0)
        num = round(transits_pred.shape[1] / transits_pred.shape[-1])
        transits_st_ed = torch.zeros(label_12_tas.size())
        for bs in range(transits_pred.shape[0]):
            for i in range(transits_pred.shape[-1]):
                transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).cpu().item() + i * num
        label_1_tas_pred = transits_st_ed[:transits_st_ed.shape[0] // 2]
        label_2_tas_pred = transits_st_ed[transits_st_ed.shape[0] // 2:]

        ############# Procedure-aware Cross-attention #############
        u_fea_96_1 = u_fea_96[:u_fea_96.shape[0] // 2].transpose(1, 2)
        u_fea_96_2 = u_fea_96[u_fea_96.shape[0] // 2:].transpose(1, 2)
        u_feamap_96_1 = u_feamap_96[:u_feamap_96.shape[0] // 2].transpose(1, 2)
        u_feamap_96_2 = u_feamap_96[u_feamap_96.shape[0] // 2:].transpose(1, 2)

        video_1_segs = []
        for bs_1 in range(u_fea_96_1.shape[0]):
            video_1_st = int(label_1_tas_pred[bs_1][0].item())
            video_1_ed = int(label_1_tas_pred[bs_1][1].item())
            if video_1_st == 0:
                video_1_st = 1
            if video_1_ed == 0:
                video_1_ed = 1
            video_1_segs.append(seg_pool_1d(u_fea_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs = torch.cat(video_1_segs, 0).transpose(1, 2)   

        video_2_segs = []
        for bs_2 in range(u_fea_96_2.shape[0]):                 
            video_2_st = int(label_2_tas_pred[bs_2][0].item())
            video_2_ed = int(label_2_tas_pred[bs_2][1].item())
            if video_2_st == 0:
                video_2_st = 1
            if video_2_ed == 0:
                video_2_ed = 1
            video_2_segs.append(seg_pool_1d(u_fea_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs = torch.cat(video_2_segs, 0).transpose(1, 2)   

        video_1_segs_map = []
        for bs_1 in range(u_feamap_96_1.shape[0]):
            video_1_st = int(label_1_tas_pred[bs_1][0].item())
            video_1_ed = int(label_1_tas_pred[bs_1][1].item())
            if video_1_st == 0:
                video_1_st = 1
            if video_1_ed == 0:
                video_1_ed = 1
            video_1_segs_map.append(seg_pool_3d(u_feamap_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs_map = torch.cat(video_1_segs_map, 0)
        video_1_segs_map = video_1_segs_map.reshape(video_1_segs_map.shape[0], video_1_segs_map.shape[1], video_1_segs_map.shape[2], -1).transpose(2, 3)
        video_1_segs_map = torch.cat([video_1_segs_map[:, :, :, i] for i in range(video_1_segs_map.shape[-1])], 2).transpose(1, 2)

        video_2_segs_map = []
        for bs_2 in range(u_fea_96_2.shape[0]):
            video_2_st = int(label_2_tas_pred[bs_2][0].item())
            video_2_ed = int(label_2_tas_pred[bs_2][1].item())
            if video_2_st == 0:
                video_2_st = 1
            if video_2_ed == 0:
                video_2_ed = 1
            video_2_segs_map.append(seg_pool_3d(u_feamap_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs_map = torch.cat(video_2_segs_map, 0)
        video_2_segs_map = video_2_segs_map.reshape(video_2_segs_map.shape[0], video_2_segs_map.shape[1], video_2_segs_map.shape[2], -1).transpose(2, 3)
        video_2_segs_map = torch.cat([video_2_segs_map[:, :, :, i] for i in range(video_2_segs_map.shape[-1])], 2).transpose(1, 2)

        decoder_video_12_map_list = []
        decoder_video_21_map_list = []
        for i in range(args.step_num):
            decoder_video_12_map = decoder(video_1_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                                     video_2_segs_map[:,
                                                     i * args.fix_size * H_t * W_t:(i + 1) * args.fix_size * H_t * W_t,
                                                     :])
            decoder_video_21_map = decoder(video_2_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                              video_1_segs_map[:, i * args.fix_size * H_t * W_t:(i + 1) * args.fix_size * H_t * W_t,
                                              :])
            decoder_video_12_map_list.append(decoder_video_12_map)
            decoder_video_21_map_list.append(decoder_video_21_map)

        decoder_video_12_map = torch.cat(decoder_video_12_map_list, 1)
        decoder_video_21_map = torch.cat(decoder_video_21_map_list, 1)

        ############# Fine-grained Contrastive Regression #############
        decoder_12_21 = torch.cat((decoder_video_12_map, decoder_video_21_map), 0)
        delta = regressor_delta(decoder_12_21)
        delta = delta.mean(1)

        ############# Negative Quality Attribution #############
        # Compute negative attribution on query video (video_1) features
        attribution_1, _ = neg_attr_head(u_fea_96_1)  # [B, T]
        neg_contribution_1 = neg_quality_agg(u_fea_96_1, attribution_1)  # [B, 1]

        # Get lambda_neg from args
        lambda_neg = getattr(args, 'lambda_neg', 0.1)

        # Modified score: score_TSA - lambda * negative_contribution
        score += (delta[:delta.shape[0]//2].detach() + label_2_score) - lambda_neg * neg_contribution_1.detach()

        for bs in range(transits_pred.shape[0] // 2):
            tIoU_results.append(segment_iou(np.array(label_12_tas.squeeze(-1).cpu())[bs],
                                            np.array(transits_st_ed.squeeze(-1).cpu())[bs], args))

    pred_scores.extend([i.item() / len(video_2_list) for i in score])

    tIoU_results_mean = [sum(tIoU_results) / len(tIoU_results)]
    tiou_thresholds = np.array([0.5, 0.75])
    tIoU_correct_per_thr = cal_tiou(tIoU_results_mean, tiou_thresholds)
    pred_tious_test_5.extend([tIoU_correct_per_thr[0]])
    pred_tious_test_75.extend([tIoU_correct_per_thr[1]])


def save_checkpoint(base_model, psnet_model, decoder, regressor_delta, neg_attr_head, neg_quality_agg, pos_attr_head, pos_quality_agg, optimizer, epoch,
                    epoch_best_aqa, rho_best, L2_min, RL2_min, prefix, args):
    checkpoint = {
        'base_model': base_model.state_dict(),
        'psnet_model': psnet_model.state_dict(),
        'decoder': decoder.state_dict(),
        'regressor_delta': regressor_delta.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'epoch_best_aqa': epoch_best_aqa,
        'rho_best': rho_best,
        'L2_min': L2_min,
        'RL2_min': RL2_min,
    }

    # Save negative attribution modules if they exist
    if neg_attr_head is not None:
        checkpoint['neg_attr_head'] = neg_attr_head.state_dict()
    if neg_quality_agg is not None:
        checkpoint['neg_quality_agg'] = neg_quality_agg.state_dict()

    # Save positive attribution modules if they exist
    if pos_attr_head is not None:
        checkpoint['pos_attr_head'] = pos_attr_head.state_dict()
    if pos_quality_agg is not None:
        checkpoint['pos_quality_agg'] = pos_quality_agg.state_dict()

    torch.save(checkpoint, os.path.join(args.experiment_path, prefix + '.pth'))

def save_outputs(pred_scores, true_scores, args):
    save_path_pred = os.path.join(args.experiment_path, 'pred.npy')
    save_path_true = os.path.join(args.experiment_path, 'true.npy')
    np.save(save_path_pred, pred_scores)
    np.save(save_path_true, true_scores)


def network_forward_train_no_segmentation(base_model, decoder, regressor_delta, pos_attr_head, pos_quality_agg, pred_scores,
                                           video_1, label_1_score, video_2, label_2_score, mse, optimizer, opti_flag,
                                           epoch, batch_idx, batch_num, args):
    """
    Training forward pass for datasets without action segmentation labels.
    Uses positive attribution instead of procedure segmentation.
    """
    start = time.time()
    optimizer.zero_grad()

    ############# I3D feature extraction #############
    com_feature_12, com_feamap_12 = base_model(video_1, video_2)
    video_1_fea = com_feature_12[:,:,:com_feature_12.shape[2] // 2]
    video_2_fea = com_feature_12[:,:,com_feature_12.shape[2] // 2:]

    # Use I3D features directly without PSNet
    # video_1_fea and video_2_fea have shape [B, T, C]

    ############# Direct Cross-attention without Procedure Segmentation #############
    # Use full temporal features for decoder
    decoder_video_12 = decoder(video_1_fea, video_2_fea)
    decoder_video_21 = decoder(video_2_fea, video_1_fea)

    ############# Fine-grained Contrastive Regression #############
    decoder_12_21 = torch.cat((decoder_video_12, decoder_video_21), 0)
    delta = regressor_delta(decoder_12_21)
    delta = delta.mean(1)

    ############# Positive Quality Attribution #############
    # Compute positive attribution on query video (video_1) features
    attribution_1, attribution_1_raw = pos_attr_head(video_1_fea)  # [B, T]
    pos_contribution_1 = pos_quality_agg(video_1_fea, attribution_1)  # [B, 1]

    # Compute positive attribution on exemplar video (video_2) features
    attribution_2, attribution_2_raw = pos_attr_head(video_2_fea)  # [B, T]
    pos_contribution_2 = pos_quality_agg(video_2_fea, attribution_2)  # [B, 1]

    # Aggregate positive contributions
    pos_contribution = torch.cat((pos_contribution_1, pos_contribution_2), 0)  # [2B, 1]

    # Sparsity loss: encourage sparse attribution
    from models.PositiveAttribution import compute_sparsity_loss, compute_concentration_loss, compute_smoothness_loss
    sparsity_loss_1 = compute_sparsity_loss(attribution_1, loss_type='l1')
    sparsity_loss_2 = compute_sparsity_loss(attribution_2, loss_type='l1')
    loss_sparsity = (sparsity_loss_1 + sparsity_loss_2) / 2.0

    # Concentration loss: encourage attribution to concentrate on few segments
    concentration_loss_1 = compute_concentration_loss(attribution_1, k=5)
    concentration_loss_2 = compute_concentration_loss(attribution_2, k=5)
    loss_concentration = (concentration_loss_1 + concentration_loss_2) / 2.0

    # Smoothness loss: encourage temporal smoothness
    smoothness_loss_1 = compute_smoothness_loss(attribution_1)
    smoothness_loss_2 = compute_smoothness_loss(attribution_2)
    loss_smoothness = (smoothness_loss_1 + smoothness_loss_2) / 2.0

    # Get hyperparameters from args (with defaults)
    lambda_pos = getattr(args, 'lambda_pos', 0.1)
    lambda_sparsity = getattr(args, 'lambda_sparsity', 0.01)
    lambda_concentration = getattr(args, 'lambda_concentration', 0.01)
    lambda_smoothness = getattr(args, 'lambda_smoothness', 0.001)

    # Score regression loss
    loss_aqa = mse(delta[:delta.shape[0]//2], (label_1_score - label_2_score)) \
               + mse(delta[delta.shape[0]//2:], (label_2_score - label_1_score))

    # Total loss
    loss = loss_aqa + \
           lambda_sparsity * loss_sparsity + \
           lambda_concentration * loss_concentration + \
           lambda_smoothness * loss_smoothness
    loss.backward()
    optimizer.step()

    end = time.time()
    batch_time = end - start

    # Modified score: score_TSA + lambda * positive_contribution
    score = (delta[:delta.shape[0]//2].detach() + label_2_score) + lambda_pos * pos_contribution_1.detach()
    pred_scores.extend([i.item() for i in score])

    if batch_idx % args.print_freq == 0:
        print('[Training][%d/%d][%d/%d] \t Batch_time: %.2f \t Batch_loss: %.4f \t '
              'lr1 : %0.5f \t lr2 : %0.5f'
              % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, loss.item(),
                 optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))


def network_forward_test_no_segmentation(base_model, decoder, regressor_delta, pos_attr_head, pos_quality_agg, pred_scores,
                                          video_1, video_2_list, label_2_score_list, args):
    """
    Testing forward pass for datasets without action segmentation labels.
    Uses positive attribution instead of procedure segmentation.
    """
    score = 0
    for video_2, label_2_score in zip(video_2_list, label_2_score_list):

        ############# I3D feature extraction #############
        com_feature_12, com_feamap_12 = base_model(video_1, video_2)
        video_1_fea = com_feature_12[:, :, :com_feature_12.shape[2] // 2]
        video_2_fea = com_feature_12[:, :, com_feature_12.shape[2] // 2:]

        ############# Direct Cross-attention without Procedure Segmentation #############
        decoder_video_12 = decoder(video_1_fea, video_2_fea)
        decoder_video_21 = decoder(video_2_fea, video_1_fea)

        ############# Fine-grained Contrastive Regression #############
        decoder_12_21 = torch.cat((decoder_video_12, decoder_video_21), 0)
        delta = regressor_delta(decoder_12_21)
        delta = delta.mean(1)

        ############# Positive Quality Attribution #############
        attribution_1, attribution_1_raw = pos_attr_head(video_1_fea)
        pos_contribution_1 = pos_quality_agg(video_1_fea, attribution_1)

        # Get hyperparameters
        lambda_pos = getattr(args, 'lambda_pos', 0.1)

        # Modified score: score_TSA + lambda * positive_contribution
        score += (delta[:delta.shape[0]//2] + label_2_score) + lambda_pos * pos_contribution_1

    pred_scores.extend([i.item() / len(video_2_list) for i in score])
