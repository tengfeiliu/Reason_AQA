"""
Unified Main Entry Point for AQA Training

Uses unified framework for all datasets (FineDiving, MTL-AQA, AQA-7, JIGSAWS).
No PSNet, uses dual-branch attribution (positive + negative) for all datasets.
"""

import os
import sys
import argparse
import yaml
import torch
from utils import parser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from tools import unified_runner


# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description='Unified AQA Training')

#     # Basic settings
#     parser.add_argument('--config', type=str, required=True, default='configs/FineDiving_Unified.yaml',
#                         help='Path to config file')
#     parser.add_argument('--benchmark', type=str, required=True, choices=['FineDiving', 'MTL_AQA', 'AQA7', 'JIGSAWS'], default='FineDiving',
#                         help='Dataset name (e.g., FineDiving, MTL_AQA, AQA7, JIGSAWS)')
#     parser.add_argument('--experiment_path', type=str, default='./experiments',
#                         help='Path to save experiments')

#     # Training settings
#     parser.add_argument('--resume', action='store_true', default=False,
#                         help='Resume from checkpoint')
#     parser.add_argument('--test', action='store_true', default=False,
#                         help='Test mode')
#     parser.add_argument('--ckpts', type=str, default='',
#                         help='Path to checkpoint for testing')

#     # GPU settings
#     parser.add_argument('--gpu', type=str, default='0,1',
#                         help='GPU ids')

#     args = parser.parse_args()

#     # Load config file
#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)

#     # Merge config into args
#     for key, value in config.items():
#         setattr(args, key, value)

#     return args


def main():
    """Main function."""
    # args = parse_args()
    args = parser.get_args()
    parser.setup(args)   
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Create experiment directory
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)

    # Print configuration
    print('='*80)
    print('Unified AQA Framework')
    print('='*80)
    print(f'Dataset: {args.benchmark}')
    print(f'Config: {args.config}')
    print(f'Experiment path: {args.experiment_path}')
    print(f'GPU: {args.gpu}')
    print('='*80)

    # Train or test
    if args.test:
        print('Testing mode...')
        unified_runner.unified_test_net(args)
    else:
        print('Training mode...')
        unified_runner.unified_train_net(args)


if __name__ == '__main__':
    main()
