import torch.nn.functional as F
from torch import nn

class FullyConnected(nn.Module):

    def __init__(self):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(int(5000*4), 1200)
        self.fc2 = nn.Linear(1200, 300)
        self.fc3 = nn.Linear(300, 10)

        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc2(x))
        x = self.dropout(x)
        x = x.view(10, -1)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

