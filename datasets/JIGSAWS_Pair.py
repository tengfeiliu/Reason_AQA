import torch
import os
import pickle
import random
import numpy as np
from PIL import Image


class JIGSAWS_Pair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        self.subset = subset
        self.transforms = transform

        # JIGSAWS specific settings
        self.cls = args.jigsaws_task  # task name: 'Suturing', 'Needle_Passing', 'Knot_Tying'
        self.fold = args.jigsaws_fold  # cross-validation fold: 0, 1, 2, 3

        # file path
        self.data_root = args.data_root
        self.info_dir = args.jigsaws_info_dir

        # load label dictionary
        with open(os.path.join(self.info_dir, 'label.pkl'), 'rb') as f:
            self.label_dict = pickle.load(f)

        # setting
        self.length = args.frame_length
        self.voter_number = args.voter_number

        # load fold split
        self.load_fold(self.fold)

    def load_fold(self, fold):
        """Load cross-validation fold split

        Args:
            fold: fold index (0, 1, 2, 3)
        """
        folds = [0, 1, 2, 3]
        if self.subset == 'train':
            folds.pop(fold)
        else:
            folds = [fold]

        with open(os.path.join(self.info_dir, 'splits.pkl'), 'rb') as f:
            cv_file = pickle.load(f)  # info of cross validation

        self.name_list = []
        all_list = cv_file[self.cls]
        for fold_idx in folds:
            for vid in all_list[fold_idx]:
                self.name_list.append(vid + '_capture1')  # only loads left view

        self.dataset = self.name_list.copy()

    def load_video(self, vname):
        """Load video frames for JIGSAWS dataset

        Args:
            vname: video name

        Returns:
            transformed video tensor
        """
        path = os.path.join(self.data_root, vname)
        img_names = sorted(os.listdir(path))

        # sample frames uniformly
        partition = np.linspace(0, len(img_names) - 1, num=self.length, dtype=np.int32)

        video = []
        for i in range(self.length):
            img_path = os.path.join(path, img_names[partition[i]])
            img = Image.open(img_path)
            video.append(img)

        return self.transforms(video)

    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        name_1 = sample_1[:-9]  # remove '_capture1'

        data = {}
        data['video'] = self.load_video(sample_1)
        data['final_score'] = torch.tensor(self.label_dict[name_1])

        if self.subset == 'test':
            # test phase: choose multiple exemplars from training set
            # need to get training set
            folds = [0, 1, 2, 3]
            folds.pop(self.fold)

            with open(os.path.join(self.info_dir, 'splits.pkl'), 'rb') as f:
                cv_file = pickle.load(f)

            train_name_list = []
            all_list = cv_file[self.cls]
            for fold_idx in folds:
                for vid in all_list[fold_idx]:
                    train_name_list.append(vid + '_capture1')

            random.shuffle(train_name_list)
            choosen_sample_list = train_name_list[:self.voter_number]

            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                tmp['video'] = self.load_video(item)
                name_tmp = item[:-9]
                tmp['final_score'] = torch.tensor(self.label_dict[name_tmp])
                target_list.append(tmp)

            return data, target_list
        else:
            # train phase: choose one exemplar
            file_list = self.dataset.copy()

            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))

            # choose one exemplar
            idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[idx]
            name_2 = sample_2[:-9]

            target = {}
            target['video'] = self.load_video(sample_2)
            target['final_score'] = torch.tensor(self.label_dict[name_2])

            return data, target

    def __len__(self):
        return len(self.dataset)
