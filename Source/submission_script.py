from UrbanSoundDataset_aug import *
from UrbanSoundDataset2D_aug import *
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

# load the csv for submission
csvpath = r'C:\Users\Copo\source\repos\UrbanSoundClassification\Data\submission_dataset'
submission = pd.read_csv(os.path.join(csvpath,'sub00.csv'))
classes = ['air_conditioner',
 'car_horn',
 'children_playing',
 'dog_bark',
 'drilling',
 'engine_idling',
 'gun_shot',
 'jackhammer',
 'siren',
 'street_music']

# load the model
#loadpath = r'C:\Users\Copo\source\repos\UrbanSoundClassification\trainings\Mon_Feb_18_203600_2019'
loadpath = r'C:\Users\Copo\source\repos\UrbanSoundClassification\trainings2D\Sat_Feb_23_165712_2019_1DCNN_longersamples'
#loadpath = r'C:\Users\Copo\source\repos\UrbanSoundClassification\trainings\Mon_Feb_18_160248_2019'
modelname = 'model_epoch_17.pt'
model = os.path.join(loadpath, modelname)
model = torch.load(model)
model.to('cuda')
print(model)

model.eval()

datapath = r'C:\\Users\\Copo\\source\\repos\\UrbanSoundClassification\\Data\\submission_dataset'
DS = UrbanSoundDataset2D_aug(datapath, None, 'submission')

batch_size = 10

test_loader = DataLoader(DS, batch_size=batch_size)

accuracy = 0
counter = 0
for sample, ID in test_loader:
    counter = counter + 1
    sample = sample.to('cuda')

    predictions = torch.exp(model(sample)).topk(1, dim=1)[1].t()
    if counter % 10 ==0:
        print(counter/len(test_loader))
    ind = 0
    for i in predictions.cpu().numpy()[0]:
        submission = submission.append({'Class': classes[i],
                                    'ID':  int(ID[ind]) }, ignore_index=True)
        ind += 1

