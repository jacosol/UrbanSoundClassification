import numpy as np
import librosa as lb
import pandas as pd
import torch
import glob
import os
from torch.utils.data import Dataset

def rollT(x, n):
    return torch.cat((x[-n:], x[:-n]))

class UrbanSoundDataset_aug(Dataset):

    def __init__(self, datapath, transforms, model_mode, sample_len=6 * 5000):
        self.model_mode = model_mode
        self.sample_len = sample_len
        self.path = datapath
        self.files = [y for x in os.walk(self.path) for y in glob.glob(os.path.join(x[0], '*.wav'))]
        self.len = len(self.files)
        self.transforms = transforms
        if self.model_mode == 'train':
            self.infos = pd.read_csv(os.path.join(self.path, 'train.csv'))
            self.classes = sorted(set(self.infos.Class))
        elif self.model_mode == 'test':
            self.infos = pd.read_csv(os.path.join(self.path, 'test.csv'))
            self.classes = sorted(set(self.infos.Class))
        elif self.model_mode == 'submission':
            self.infos = []
        else:
            print('WARNING: the train_or_test argument should be a string <train> or <test> ')
            1 / 0


    def __getitem__(self, index):
        sampleID = self.files[index].split(os.sep)[-1].split('.')[0]
        sample, sr = lb.load(self.files[index], sr=5000, mono=True)
        sample = sample - np.mean(sample)
        sample = sample / np.std(sample)
        sample = sample[:int(self.sample_len/2)]
        if len(sample) < self.sample_len:
            tmplen = len(sample)
            sample = np.insert(sample, 0, np.zeros(1+int(0.5 * (self.sample_len - tmplen) )))
            sample = np.insert(sample, len(sample), np.zeros(1+int(0.5 * (self.sample_len - tmplen) )))
        if len(sample) > self.sample_len:
            sample = sample[:self.sample_len]
        roll_len_max = int(self.sample_len/4)

        roll_len = int( (np.random.uniform()*2 -1) *roll_len_max )
        sample = torch.FloatTensor(sample)
        sample = rollT(sample, roll_len)

        sample = np.expand_dims(sample, 0)

        if self.transforms:
            sample = self.transforms(sample)
        if self.model_mode in ['train', 'test']:
            label_str = self.infos.loc[self.infos['ID'] == int(sampleID)].Class.values
            label = [i for i in range(len(self.classes)) if self.classes[i] == label_str]
            label = torch.LongTensor([label])[0][0]
            return (sample, label)
        else:
            return(sample, sampleID)

    def __len__(self):
        return self.len

