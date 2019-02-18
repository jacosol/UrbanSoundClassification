import numpy as np
import librosa as lb
import pandas as pd
import torch
import glob
import os
from torch.utils.data import Dataset


class UrbanSoundDataset(Dataset):

    def __init__(self, datapath, transforms, train_or_test, sample_len=4 * 5000):
        self.ToT = train_or_test
        self.sample_len = sample_len
        self.path = datapath
        self.files = [y for x in os.walk(self.path) for y in glob.glob(os.path.join(x[0], '*.wav'))]
        self.len = len(self.files)
        self.transforms = transforms
        if train_or_test == 'train':
            self.infos = pd.read_csv(os.path.join(self.path, 'train.csv'))
        elif train_or_test == 'test':
            self.infos = pd.read_csv(os.path.join(self.path, 'test.csv'))
        else:
            print('WARNING: the train_or_test argument should be a string <train> or <test> ')
            1 / 0

        self.classes = sorted(set(self.infos.Class))

    def __getitem__(self, index):
        sampleID = self.files[index].split(os.sep)[-1].split('.')[0]
        sample, sr = lb.load(self.files[index], sr=5000, mono=True)
        if len(sample) < self.sample_len:
            sample = np.insert(sample, 0, np.zeros(self.sample_len - len(sample)))
        if len(sample) > self.sample_len:
            sample = sample[:self.sample_len]
        sample = torch.FloatTensor(sample)
        sample = np.expand_dims(sample, 0)

        label_str = self.infos.loc[self.infos['ID'] == int(sampleID)].Class.values
        label = [i for i in range(len(self.classes)) if self.classes[i] == label_str]
        label = torch.LongTensor([label])[0][0]

        if self.transforms:
            sample = self.transforms(sample)
        return (sample, label)

    def __len__(self):
        return self.len

