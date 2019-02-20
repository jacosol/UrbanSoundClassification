import torch.nn.functional as F
from torch import nn

class Classifier1D(nn.Module):

    def __init__(self):
        super(Classifier1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 9, padding=4)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv1d(64, 128, 5, padding=2)
        self.conv5 = nn.Conv1d(128, 256, 5, padding=2)

        self.pool = nn.MaxPool1d(4, 4)
        # linear layer
        self.fc1 = nn.Linear(int(4864), 64)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(64, 1024)
        self.fc3 = nn.Linear(1024, 10)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv5(x)))

        # flatten audio input
        x = x.view(10, -1)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

