from UrbanSoundDataset import *
from Classifier1D import *
from FullyConnected import *
import torch
import os
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import time

trainpath = 'C:\\Users\\Copo\\source\\repos\\UrbanSoundClassification\\Data\\train'
testpath = 'C:\\Users\\Copo\\source\\repos\\UrbanSoundClassification\\Data\\test'
validationpath = 'C:\\Users\\Copo\\source\\repos\\UrbanSoundClassification\\Data\\validation'
savepath = 'C:\\Users\\Copo\\source\\repos\\UrbanSoundClassification\\trainings\\' + ''.join('_'.join(time.ctime().split()).split(':')) + '_FC'
try:
    os.mkdir(savepath)
except:
    print('directory already exists')
resume = 0

batch_size = 10
epochs = 10

DS = UrbanSoundDataset(trainpath, None, 'train')

num_train = len(DS)
testval_fraction = 0.3
indices = list(range(num_train))
split_idx = int(num_train*(1-testval_fraction))
train_idx = indices[:split_idx]

train_sampler = SubsetRandomSampler(train_idx)

indices_testval = indices[split_idx:]
split_idx_testval = int(num_train*testval_fraction*0.5)
val_idx = indices_testval[:split_idx_testval]
test_idx = indices_testval[split_idx_testval:]

val_sampler = SubsetRandomSampler(val_idx)
test_sampler = SubsetRandomSampler(test_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = DataLoader(DS, batch_size=batch_size,
                          drop_last=True, sampler=train_sampler)
val_loader = DataLoader(DS, batch_size=batch_size,
                          drop_last=True, sampler=val_sampler)
test_loader = DataLoader(DS, batch_size=batch_size,
                          drop_last=True, sampler=test_sampler)

#saves the sampler indeces for the specific training
np.save(os.path.join(savepath, 'train_idx.npy'), np.array(train_idx))
np.save(os.path.join(savepath, 'test_idx.npy'), np.array(test_idx))
np.save(os.path.join(savepath, 'val_idx.npy'), np.array(val_idx))



if resume:
    model_with_val = torch.load(r'C:\Users\Copo\source\repos\UrbanSoundClassification\trainings\_feb18Mon_Feb_18_160248_2019\model_epoch_8.pt')
    train_loss_overtime = list(np.load(r'C:\Users\Copo\source\repos\UrbanSoundClassification\trainings\_feb18Mon_Feb_18_160248_2019\trainloss.npy'))
    test_loss_overtime = list(np.load(r'C:\Users\Copo\source\repos\UrbanSoundClassification\trainings\_feb18Mon_Feb_18_160248_2019\testloss.npy'))
    starting_epoch = 8
else:
    train_loss_overtime = []
    test_loss_overtime = []
    #model_with_val = Classifier1D()
    model_with_val = FullyConnected()
    starting_epoch = 0
model_with_val.to('cuda')

# train script with validation

criterion = nn.NLLLoss()
optimizer = optim.Adam(model_with_val.parameters())

for e in range(starting_epoch, starting_epoch + epochs):

    train_loss = 0.0
    test_loss = 0.0
    model_with_val.train()
    status = 0
    for sample, label in train_loader:
        status += 1
        if status % 10 == 0:
            print(f'epoch progress: {int (status*batch_size/len(train_loader.dataset)*100)}%')
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        sample = Variable(sample.to('cuda'))
        output = model_with_val(sample)
        output.to('cuda')
        # calculate the batch loss
        label = Variable(label.to('cuda'))
        loss = criterion(output, label)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item() * sample.size(0)

    ######################
    # validate the model #
    ######################
    model_with_val.eval()
    status = 0
    for data, target in val_loader:

        status += 1
        if status % 2 == 0:
            print(f'validating: {int (status*batch_size/len(val_loader.sampler)*100)}%')
        # forward pass: compute predicted outputs by passing inputs to the model
        data = Variable(data.to('cuda'))
        output = model_with_val(data)
        output.to('cuda')
        target = Variable(target.to('cuda'))
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        test_loss += loss.item() * data.size(0)

    # calculate average losses
    train_loss = train_loss / len(train_loader.dataset)
    test_loss = test_loss / len(val_loader.dataset)

    test_loss_overtime.append(test_loss)
    train_loss_overtime.append(train_loss)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(
        e, train_loss, test_loss))
    if e > starting_epoch and test_loss_overtime[e] < test_loss_overtime[e - 1]:
        torch.save(model_with_val, os.path.join(savepath, f'model_epoch_{e}.pt'))

    np.save(os.path.join(savepath, 'trainloss.npy'), np.array(train_loss_overtime))
    np.save(os.path.join(savepath, 'testloss.npy'), np.array(test_loss_overtime))

