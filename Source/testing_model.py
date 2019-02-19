from UrbanSoundDataset import *
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from  tkinter import *
import tkinter.filedialog

#open a folder with the training info

# root = Tk()
# loadpath = tkinter.filedialog.askdirectory()
# root.destroy()

loadpath = r'C:\Users\Copo\source\repos\UrbanSoundClassification\trainings\Mon_Feb_18_203600_2019'
loadpath = r'C:\Users\Copo\source\repos\UrbanSoundClassification\trainings\Mon_Feb_18_223935_2019_FC'
modelname = 'model_epoch_10.pt'
model = os.path.join(loadpath, modelname)
model = torch.load(model)
model.to('cuda')
print(model)

model.eval()

test_idx = np.load(os.path.join(loadpath, 'test_idx.npy'))

trainpath = r'C:\\Users\\Copo\\source\\repos\\UrbanSoundClassification\\Data\\train'
DS = UrbanSoundDataset(trainpath, None, 'train')

batch_size = 10
test_sampler = SubsetRandomSampler(test_idx)
test_loader = DataLoader(DS, batch_size=batch_size,
                          drop_last=True, sampler=test_sampler)


accuracy = 0
counter = 0
for sample, label in test_loader:
    counter = counter + 1
    sample = sample.to('cuda')
    label = label.to('cuda')

    predictions = torch.exp(model(sample)).topk(1, dim=1)[1].t() - label
    print(predictions)
    accuracy += sum(sum(predictions.cpu().numpy() == 0))
    if counter % 10 ==0:
        print(counter*batch_size/len(test_idx))

accuracy = accuracy/len(test_idx)
print(f'Accuracy on the test set for {modelname} : {accuracy}')

