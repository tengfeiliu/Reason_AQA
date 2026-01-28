import torch
import scipy.io
import os
import random
import numpy as np
from PIL import Image


class AQA7_Pair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        self.subset = subset
        self.transforms = transform

        # sport class configuration
        classes_name = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
        self.sport_class = classes_name[args.class_idx - 1]
        self.class_idx = args.class_idx  # sport class index (from 1 begin)
        self.score_range = args.score_range

        # file path
        self.data_root = args.data_root
        self.data_path = os.path.join(self.data_root, '{}-out'.format(self.sport_class))

        # load split files
        self.split_path = os.path.join(self.data_root, 'Split_4', 'split_4_train_list.mat')
        self.split = scipy.io.loadmat(self.split_path)['consolidated_train_list']
        self.split = self.split[self.split[:, 0] == self.class_idx].tolist()

        if self.subset == 'test':
            self.split_path_test = os.path.join(self.data_root, 'Split_4', 'split_4_test_list.mat')
            self.split_test = scipy.io.loadmat(self.split_path_test)['consolidated_test_list']
            self.split_test = self.split_test[self.split_test[:, 0] == self.class_idx].tolist()

        # setting
        self.length = args.frame_length
        self.voter_number = args.voter_number

        if self.subset == 'test':
            self.dataset = self.split_test.copy()
        else:
            self.dataset = self.split.copy()

    def load_video(self, idx):
        """Load video frames for AQA-7 dataset

        Args:
            idx: video index

        Returns:
            transformed video tensor
        """
        video_path = os.path.join(self.data_path, '%03d' % idx)
        video = [Image.open(os.path.join(video_path, 'img_%05d.jpg' % (i + 1))) for i in range(self.length)]
        return self.transforms(video)

    def normalize_score(self, score):
        """Normalize score based on sport class and score range"""
        # score normalization based on class_idx and score_range
        # this follows the implementation in T2CR SevenPair.py
        if self.class_idx == 1:  # diving
            min_score, max_score = 0, 102.6
        elif self.class_idx == 2:  # gym_vault
            min_score, max_score = 0, 16.5
        elif self.class_idx == 3:  # ski_big_air
            min_score, max_score = 0, 100
        elif self.class_idx == 4:  # snowboard_big_air
            min_score, max_score = 0, 100
        elif self.class_idx == 5:  # sync_diving_3m
            min_score, max_score = 0, 102
        elif self.class_idx == 6:  # sync_diving_10m
            min_score, max_score = 0, 102
        else:
            min_score, max_score = 0, 100

        normalized = (score - min_score) / (max_score - min_score) * self.score_range
        return normalized

    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        assert int(sample_1[0]) == self.class_idx
        idx = int(sample_1[1])

        data = {}
        data['video'] = self.load_video(idx)
        data['final_score'] = self.normalize_score(sample_1[2])

        if self.subset == 'test':
            # test phase: choose multiple exemplars from training set
            train_file_list = self.split.copy()
            random.shuffle(train_file_list)
            choosen_sample_list = train_file_list[:self.voter_number]

            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                tmp_idx = int(item[1])
                tmp['video'] = self.load_video(tmp_idx)
                tmp['final_score'] = self.normalize_score(item[2])
                target_list.append(tmp)

            return data, target_list
        else:
            # train phase: choose one exemplar
            file_list = self.split.copy()

            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))

            # choose one exemplar
            tmp_idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[tmp_idx]
            target = {}
            target['video'] = self.load_video(int(sample_2[1]))
            target['final_score'] = self.normalize_score(sample_2[2])

            return data, target

    def __len__(self):
        return len(self.dataset)
