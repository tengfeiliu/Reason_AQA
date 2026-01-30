"""
Unified Training Runner

Unified training/testing runner for all AQA datasets without PSNet.
All datasets use the same training framework with dual-branch attribution.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from tools import unified_builder, unified_helper
from utils import misc
import time


def unified_train_net(args):
    """
    Unified training function for all datasets.

    All datasets (FineDiving, MTL-AQA, AQA-7, JIGSAWS) use the same framework:
    - I3D backbone
    - Direct cross-attention (no PSNet)
    - Dual-branch attribution (positive + negative)
    """
    print('Unified Trainer start ... ')

    # Build dataset
    train_dataset, test_dataset = unified_builder.dataset_builder(args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs_train,
        shuffle=True, num_workers=int(args.workers),
        pin_memory=True, worker_init_fn=misc.worker_init_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.bs_test,
        shuffle=False, num_workers=int(args.workers),
        pin_memory=True
    )

    # Build model
    base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution, criterion = \
        unified_builder.unified_model_builder(args)

    # CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_model = base_model.cuda()
        pos_decoder = pos_decoder.cuda()
        neg_decoder = neg_decoder.cuda()
        pos_regressor_delta = pos_regressor_delta.cuda()
        neg_regressor_delta = neg_regressor_delta.cuda()
        dual_attribution = dual_attribution.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    # Build optimizer
    optimizer, scheduler = unified_builder.build_opti_sche(
        base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution, args
    )

    # Initialize training state
    start_epoch = 0
    global epoch_best_aqa, rho_best, L2_min, RL2_min
    epoch_best_aqa = 0
    rho_best = 0
    L2_min = 1000
    RL2_min = 1000

    # Resume checkpoint
    if args.resume:
        start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min = \
            unified_builder.resume_train(
                base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution,
                optimizer, args
            )
        print('resume ckpts @ %d epoch(rho = %.4f, L2 = %.4f , RL2 = %.4f)'
              % (start_epoch - 1, rho_best, L2_min, RL2_min))

    # DataParallel
    base_model = nn.DataParallel(base_model)
    pos_decoder = nn.DataParallel(pos_decoder)
    neg_decoder = nn.DataParallel(neg_decoder)
    pos_regressor_delta = nn.DataParallel(pos_regressor_delta)
    neg_regressor_delta = nn.DataParallel(neg_regressor_delta)
    dual_attribution = nn.DataParallel(dual_attribution)

    # Training loop
    for epoch in range(start_epoch, args.max_epoch):
        true_scores = []
        pred_scores = []

        # Set to training mode
        base_model.train()
        pos_decoder.train()
        neg_decoder.train()
        pos_regressor_delta.train()
        neg_regressor_delta.train()
        dual_attribution.train()

        if args.fix_bn:
            base_model.apply(misc.fix_bn)

        for idx, (data, target) in enumerate(train_dataloader):
            # video_1 is query and video_2 is exemplar
            video_1 = data['video'].float().cuda()
            video_2 = target['video'].float().cuda()
            label_1_score = data['final_score'].float().reshape(-1, 1).cuda()
            label_2_score = target['final_score'].float().reshape(-1, 1).cuda()

            # Forward pass
            unified_helper.unified_network_forward_train(
                base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution,
                pred_scores, video_1, label_1_score, video_2, label_2_score,
                criterion, optimizer, epoch, idx+1, len(train_dataloader), args
            )
            true_scores.extend(data['final_score'].numpy())

        # Evaluation results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]

        print('[Training] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f, lr1: %.4f, lr2: %.4f' %
              (epoch, rho, L2, RL2, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))

        # Validation
        unified_validate(
            base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution,
            test_dataloader, epoch, optimizer, args
        )

        # Save checkpoint
        unified_helper.save_checkpoint(
            base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution,
            optimizer, epoch, epoch_best_aqa, rho_best, L2_min, RL2_min,
            'last', args
        )

        print('[TEST] EPOCH: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f' %
              (epoch_best_aqa, rho_best, L2_min, RL2_min))

        # Scheduler step
        if scheduler is not None:
            scheduler.step()


def unified_validate(
    base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution,
    test_dataloader, epoch, optimizer, args
):
    """
    Unified validation function for all datasets.
    """
    print("Start validating epoch {}".format(epoch))
    global use_gpu
    global epoch_best_aqa, rho_best, L2_min, RL2_min

    true_scores = []
    pred_scores = []

    # Set to evaluation mode
    base_model.eval()
    pos_decoder.eval()
    neg_decoder.eval()
    pos_regressor_delta.eval()
    neg_regressor_delta.eval()
    dual_attribution.eval()

    batch_num = len(test_dataloader)
    with torch.no_grad():
        datatime_start = time.time()

        for batch_idx, (data, target) in enumerate(test_dataloader, 0):
            datatime = time.time() - datatime_start
            start = time.time()

            video_1 = data['video'].float().cuda()
            video_2_list = [item['video'].float().cuda() for item in target]
            label_2_score_list = [item['final_score'].float().reshape(-1, 1).cuda() for item in target]

            # Forward pass
            unified_helper.unified_network_forward_test(
                base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution,
                pred_scores, video_1, video_2_list, label_2_score_list, args
            )

            batch_time = time.time() - start
            if batch_idx % args.print_freq == 0:
                print('[TEST][%d/%d][%d/%d] \t Batch_time %.2f \t Data_time %.2f'
                      % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, datatime))
            datatime_start = time.time()
            true_scores.extend(data['final_score'].numpy())

        # Evaluation results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]

        if L2_min > L2:
            L2_min = L2
        if RL2_min > RL2:
            RL2_min = RL2
        if rho > rho_best:
            rho_best = rho
            epoch_best_aqa = epoch
            print('-----New best found!-----')
            unified_helper.save_outputs(pred_scores, true_scores, args)
            unified_helper.save_checkpoint(
                base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution,
                optimizer, epoch, epoch_best_aqa, rho_best, L2_min, RL2_min,
                'best', args
            )
        print('[TEST] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f' %
              (epoch, rho, L2, RL2))


def unified_test_net(args):
    """
    Unified testing function for all datasets.
    """
    print('Unified Tester start ... ')

    # Build dataset
    train_dataset, test_dataset = unified_builder.dataset_builder(args)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.bs_test,
        shuffle=False, num_workers=int(args.workers),
        pin_memory=True
    )

    # Build model
    base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution, criterion = \
        unified_builder.unified_model_builder(args)

    # Load checkpoint
    unified_builder.load_model(
        base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution, args
    )

    # CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_model = base_model.cuda()
        pos_decoder = pos_decoder.cuda()
        neg_decoder = neg_decoder.cuda()
        pos_regressor_delta = pos_regressor_delta.cuda()
        neg_regressor_delta = neg_regressor_delta.cuda()
        dual_attribution = dual_attribution.cuda()
        torch.backends.cudnn.benchmark = True

    # DataParallel
    base_model = nn.DataParallel(base_model)
    pos_decoder = nn.DataParallel(pos_decoder)
    neg_decoder = nn.DataParallel(neg_decoder)
    pos_regressor_delta = nn.DataParallel(pos_regressor_delta)
    neg_regressor_delta = nn.DataParallel(neg_regressor_delta)
    dual_attribution = nn.DataParallel(dual_attribution)

    # Test
    unified_test(
        base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution,
        test_dataloader, args
    )


def unified_test(
    base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution,
    test_dataloader, args
):
    """
    Unified test function for all datasets.
    """
    global use_gpu

    true_scores = []
    pred_scores = []

    # Set to evaluation mode
    base_model.eval()
    pos_decoder.eval()
    neg_decoder.eval()
    pos_regressor_delta.eval()
    neg_regressor_delta.eval()
    dual_attribution.eval()

    batch_num = len(test_dataloader)
    with torch.no_grad():
        datatime_start = time.time()

        for batch_idx, (data, target) in enumerate(test_dataloader, 0):
            datatime = time.time() - datatime_start
            start = time.time()

            video_1 = data['video'].float().cuda()
            video_2_list = [item['video'].float().cuda() for item in target]
            label_2_score_list = [item['final_score'].float().reshape(-1, 1).cuda() for item in target]

            # Forward pass
            unified_helper.unified_network_forward_test(
                base_model, pos_decoder, neg_decoder, pos_regressor_delta, neg_regressor_delta, dual_attribution,
                pred_scores, video_1, video_2_list, label_2_score_list, args
            )

            batch_time = time.time() - start
            if batch_idx % args.print_freq == 0:
                print('[TEST][%d/%d] \t Batch_time %.2f \t Data_time %.2f'
                      % (batch_idx, batch_num, batch_time, datatime))
            datatime_start = time.time()
            true_scores.extend(data['final_score'].numpy())

        # Evaluation results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]

        print('[TEST] correlation: %.6f, L2: %.6f, RL2: %.6f' % (rho, L2, RL2))
