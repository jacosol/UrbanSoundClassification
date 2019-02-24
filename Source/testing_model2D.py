from UrbanSoundDataset2D_aug import *
from Classifier2D import *
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from  tkinter import *
import tkinter.filedialog

#open a folder with the training info

# root = Tk()
# loadpath = tkinter.filedialog.askdirectory()
# root.destroy()

# load the model
loadpath = r'C:\Users\Copo\source\repos\UrbanSoundClassification\trainings2D\Sun_Feb_24_143840_2019_1DCNN_longersamples'
modelname= 'model_epoch_20.pt'
model = os.path.join(loadpath, modelname)
model = torch.load(model)
model.to('cuda')
print(model)
print(modelname)
model.eval()

test_idx = np.load(os.path.join(loadpath, 'val_idx.npy'))

trainpath = r'C:\\Users\\Copo\\source\\repos\\UrbanSoundClassification\\Data\\train'
DS = UrbanSoundDataset2D_aug(trainpath, None, 'train')

batch_size = 10
test_sampler = SubsetRandomSampler(test_idx)
test_loader = DataLoader(DS, batch_size=batch_size,
                          drop_last=True, sampler=test_sampler)



accuracy = 0
counter = 0

classes_right = np.zeros(10)
classes_wrong = np.zeros(10)
file_wrong = []

for sample, label in test_loader:
    counter = counter + 1
    sample = sample.to('cuda')
    label = label.to('cuda')

    predictions = torch.exp(model(sample)).topk(1, dim=1)[1].t() - label
    print(predictions)
    for i in np.where(predictions.cpu().numpy()==0)[1]:
        classes_right[i] = classes_right[i] + 1
    for i in np.where(predictions.cpu().numpy()!=0)[1]:
        classes_wrong[i] = classes_wrong[i] + 1
        file_wrong.append(sample[i].cpu().numpy())
    accuracy += sum(sum(predictions.cpu().numpy() == 0))
    if counter % 10 ==0:
        print(counter*batch_size/len(test_idx))




accuracy = accuracy/len(test_idx)
print(f'Accuracy on the test set for {modelname} : {accuracy}')

