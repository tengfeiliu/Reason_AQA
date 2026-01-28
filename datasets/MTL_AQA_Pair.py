import torch
import numpy as np
import os
import pickle
import random
import glob
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class MTL_AQA_Pair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        self.subset = subset
        self.transforms = transform
        self.dive_number_choosing = args.action_number_choosing
        self.usingDD = args.usingDD
        self.length = args.frame_length
        self.voter_number = args.voter_number
        self.args = args

        # file path
        self.data_root = args.data_root
        self.label_path = args.label_path
        self.label_dict = self.read_pickle(self.label_path)

        with open(args.train_split, 'rb') as f:
            self.train_dataset_list = pickle.load(f)
        with open(args.test_split, 'rb') as f:
            self.test_dataset_list = pickle.load(f)

        # temporal augmentation for training
        self.temporal_shift = random.randint(0, self.args.temporal_aug)

        # build dive number dict for exemplar choosing
        self.dive_number_dict = {}
        self.difficulties_dict = {}

        # label encoder for dive class
        self.label_encoder = None
        self.onehot_labels = None
        self.preprocess_class()

        if self.subset == 'train':
            self.dataset = self.train_dataset_list
        else:
            self.dataset = self.test_dataset_list
            self.dive_number_dict_test = {}
            self.difficulties_dict_test = {}

        self.choose_list = self.train_dataset_list.copy()
        # if self.dive_number_choosing:
        self.preprocess()
        self.check_exemplar_dict()

    def preprocess_class(self):
        """Preprocess dive class labels for one-hot encoding"""
        dive_label = []
        for item in self.label_dict:
            dive_label.append(self.label_dict.get(item)['dive_number'])
        dive_label = list(set(dive_label))
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(dive_label)
        onehot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
        self.onehot_labels = onehot_encoder.fit_transform(encoded_labels.reshape(-1, 1))

    def preprocess(self):
        """Build dictionaries for dive number based exemplar selection"""
        if self.dive_number_choosing:
            for item in self.train_dataset_list:
                dive_number = self.label_dict.get(item)['dive_number']
                if self.dive_number_dict.get(dive_number) is None:
                    self.dive_number_dict[dive_number] = []
                self.dive_number_dict[dive_number].append(item)

            if self.subset == 'test':
                for item in self.test_dataset_list:
                    dive_number = self.label_dict.get(item)['dive_number']
                    if self.dive_number_dict_test.get(dive_number) is None:
                        self.dive_number_dict_test[dive_number] = []
                    self.dive_number_dict_test[dive_number].append(item)
        else:
            ## DD
            for item in self.train_dataset_list:
                difficulty = self.label_dict.get(item)['difficulty']
                if self.difficulties_dict.get(difficulty) is None:
                    self.difficulties_dict[difficulty] = []
                self.difficulties_dict[difficulty].append(item)

            if self.subset == 'test':
                for item in self.test_dataset_list:
                    difficulty = self.label_dict.get(item)['difficulty']
                    if self.difficulties_dict_test.get(difficulty) is None:
                        self.difficulties_dict_test[difficulty] = []
                    self.difficulties_dict_test[difficulty].append(item)

    def check_exemplar_dict(self):
        """Verify the correctness of exemplar dictionaries"""
        if self.dive_number_choosing:
            if self.subset == 'train':
                for key in sorted(list(self.dive_number_dict.keys())):
                    file_list = self.dive_number_dict[key]
                    for item in file_list:
                        assert self.label_dict[item]['dive_number'] == key
            if self.subset == 'test':
                for key in sorted(list(self.dive_number_dict_test.keys())):
                    file_list = self.dive_number_dict_test[key]
                    for item in file_list:
                        assert self.label_dict[item]['dive_number'] == key
        else:
            if self.subset == 'train':
                for key in sorted(list(self.difficulties_dict.keys())):
                    file_list = self.difficulties_dict[key]
                    for item in file_list:
                        assert self.label_dict[item]['difficulty'] == key
            if self.subset == 'test':
                for key in sorted(list(self.difficulties_dict_test.keys())):
                    file_list = self.difficulties_dict_test[key]
                    for item in file_list:
                        assert self.label_dict[item]['difficulty'] == key

    def load_video(self, video_file_name, phase='train'):
        """Load video frames for MTL-AQA dataset

        Args:
            video_file_name: tuple (event_id, video_id)
            phase: 'train' or 'test'

        Returns:
            transformed video tensor
        """
        # MTL-AQA frames are stored as: data_root/{event_id:02d}/*.jpg
        image_list = sorted(
            glob.glob(os.path.join(self.data_root, str('{:02d}'.format(video_file_name[0])) + '_' + str('{:02d}'.format(video_file_name[1])), '*.jpg'))
        )

        # end_frame = self.label_dict.get(video_file_name).get('end_frame')
        end_frame = self.length

        # temporal augmentation for training
        if phase == 'train':
            temporal_aug_shift = self.temporal_shift
            end_frame = end_frame + temporal_aug_shift

        start_frame = end_frame - self.length

        # load frames
        video = [Image.open(image_list[start_frame + i]) for i in range(self.length)]

        return self.transforms(video)

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        data = {}
        data['video'] = self.load_video(sample_1, self.subset)
        data['final_score'] = self.label_dict.get(sample_1).get('final_score')
        data['difficulty'] = self.label_dict.get(sample_1).get('difficulty')
        data['completeness'] = (data['final_score'] / data['difficulty'])

        # add judge scores and variance
        judge_scores = self.label_dict.get(sample_1).get('judge_scores')
        data['judge_scores'] = np.sort(judge_scores)[2:5]  # remove highest and lowest 2 scores
        data['var'] = np.var(data['judge_scores'])

        # add dive class (one-hot encoded)
        sample_label_encoded = self.label_encoder.transform([self.label_dict.get(sample_1).get('dive_number')])
        data['dive_class'] = self.onehot_labels[sample_label_encoded]

        # choose exemplar(s)
        if self.subset == 'train':
            # train phase: choose one exemplar
            if self.dive_number_choosing:
                file_list = self.dive_number_dict[self.label_dict[sample_1]['dive_number']].copy()
            elif self.usingDD:
                # dd
                # file_list = self.train_dataset_list.copy()
                file_list = self.difficulties_dict[self.label_dict[sample_1]['difficulty']]
            else:
                # randomly
                file_list = self.train_dataset_list.copy()

            # exclude self
            # if len(file_list) > 1:
            #     file_list.pop(file_list.index(sample_1))

            # choose one exemplar
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))
            idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[idx]
            target = {}
            target['video'] = self.load_video(sample_2, 'train')
            target['final_score'] = self.label_dict.get(sample_2).get('final_score')
            target['difficulty'] = self.label_dict.get(sample_2).get('difficulty')
            target['completeness'] = (target['final_score'] / target['difficulty'])

            judge_scores_2 = self.label_dict.get(sample_2).get('judge_scores')
            target['judge_scores'] = np.sort(judge_scores_2)[2:5]
            target['var'] = np.var(target['judge_scores'])

            sample_label_encoded_2 = self.label_encoder.transform([self.label_dict.get(sample_2).get('dive_number')])
            target['dive_class'] = self.onehot_labels[sample_label_encoded_2]

            return data, target
        else:
            # test phase: choose multiple exemplars
            if self.dive_number_choosing:
                train_file_list = self.dive_number_dict[self.label_dict[sample_1]['dive_number']]
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            elif self.usingDD:
                # dd
                train_file_list = self.difficulties_dict[self.label_dict[sample_1]['difficulty']]
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            else:
                # randomly
                train_file_list = self.choose_list
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]

            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                tmp['video'] = self.load_video(item, 'test')
                tmp['final_score'] = self.label_dict.get(item).get('final_score')
                tmp['difficulty'] = self.label_dict.get(item).get('difficulty')
                tmp['completeness'] = (tmp['final_score'] / tmp['difficulty'])

                judge_scores_tmp = self.label_dict.get(item).get('judge_scores')
                tmp['judge_scores'] = np.sort(judge_scores_tmp)[2:5]
                tmp['var'] = np.var(tmp['judge_scores'])

                sample_label_encoded_tmp = self.label_encoder.transform([self.label_dict.get(item).get('dive_number')])
                tmp['dive_class'] = self.onehot_labels[sample_label_encoded_tmp]

                target_list.append(tmp)

            return data, target_list

    def __len__(self):
        return len(self.dataset)
