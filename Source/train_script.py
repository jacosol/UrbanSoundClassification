from UrbanSoundDataset import *
from Classifier1D import *
import torch
import os
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable


trainpath = 'C:\\Users\\Copo\\source\\repos\\UrbanSoundClassification\\Data\\train'
testpath = 'C:\\Users\\Copo\\source\\repos\\UrbanSoundClassification\\Data\\test'
validationpath = 'C:\\Users\\Copo\\source\\repos\\UrbanSoundClassification\\Data\\validation'
savepath = 'C:\\Users\\Copo\\source\\repos\\UrbanSoundClassification\\trainings\\_feb17'
resume = 1

batch_size = 10
epochs = 10

DS = UrbanSoundDataset(trainpath, None, 'train')
DS_test = UrbanSoundDataset(testpath, None, 'test')
DS_validation = UrbanSoundDataset(validationpath, None, 'valid')
num_train = len(DS)
indices = list(range(num_train))

# prepare data loaders (combine dataset and sampler)
train_loader = DataLoader(DS, batch_size=batch_size, shuffle=True,
                          drop_last=True)
test_loader = DataLoader(DS_test, batch_size=batch_size, shuffle=True,
                          drop_last=True)

if resume:
    model_with_val = torch.load('C:\\Users\\Copo\\source\\repos\\UrbanSoundClassification\\trainings\\_feb17\\model_epoch_9.pt')
    train_loss_overtime = np.load('C:\\Users\\Copo\\source\\repos\\UrbanSoundClassification\\trainings\\_feb17\\trainloss.npy')
    test_loss_overtime = np.load('C:\\Users\\Copo\\source\\repos\\UrbanSoundClassification\\trainings\\_feb17\\testloss.npy')

else:
    train_loss_overtime = []
    test_loss_overtime = []
    model_with_val = Classifier1D()
model_with_val.to('cuda')

# train script with validation

criterion = nn.NLLLoss()
optimizer = optim.Adam(model_with_val.parameters())

train_loss_overtime = []
test_loss_overtime = []
for e in range(epochs):

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
    for data, target in test_loader:

        status += 1
        if status % 2 == 0:
            print(f'validating: {int (status*batch_size/len(test_loader.sampler)*100)}%')
        # forward pass: compute predicted outputs by passing inputs to the model
        data = Variable(data.to('cuda'))
        output = model_with_val(data)
        output.to('cuda')
        target = Variable(target.to('cuda'))
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item() * data.size(0)

    # calculate average losses
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(test_loader.dataset)

    test_loss_overtime.append(valid_loss)
    train_loss_overtime.append(train_loss)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        e, train_loss, valid_loss))
    if e > 0 and test_loss_overtime[e] < test_loss_overtime[e - 1]:
        torch.save(model_with_val, os.path.join(savepath, f'model_epoch_{e}.pt'))

np.save(os.path.join(savepath, 'trainloss.npy'), np.array(train_loss_overtime))
np.save(os.path.join(savepath, 'testloss.npy'), np.array(test_loss_overtime))
