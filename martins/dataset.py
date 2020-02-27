import pickle
from torch.utils.data import Dataset
import os
import random
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, root_dir, file_name, train=True):
        f = open(os.path.join(root_dir, file_name), "rb")
        dataset = pickle.load(f)
        if train:
            self.data = dataset['tr_data']
            self.label = dataset['tr_lbl']
        else:
            self.data = dataset['te_data']
            self.label = dataset['te_lbl']
        self.len = self.label.shape[0]
        self.data_mean = self.data.mean()
        self.data_std = self.data.std()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

class TimeSeriesDatasetConcat(Dataset):
    # load train and test at the same time
    def __init__(self, root_dir, file_name, seed, train=True):
        f = open(os.path.join(root_dir, file_name), "rb")
        dataset = pickle.load(f)
        self.tr_data = dataset['tr_data'][:5000]
        self.tr_label = dataset['tr_lbl'][:5000]
        self.te_data = dataset['te_data_only']
        self.tr_len = self.tr_data.shape[0]
        self.te_len = self.te_data.shape[0]

        # random sample from train data so two sets 
        # have same number of data
        random.seed(seed)
        index = random.sample(range(self.tr_len), self.te_len) # sampling

        self.tr_data = self.tr_data[index]
        self.tr_label = self.tr_label[index]

        self.tr_data_mean = self.tr_data.mean()
        self.te_data_mean = self.te_data.mean()
        self.tr_data_std = self.tr_data.std()
        self.te_data_std = self.te_data.std()

    def __len__(self):
        return self.te_len

    def __getitem__(self, idx):
        return self.tr_data[idx], self.te_data[idx]
