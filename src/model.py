import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*30*2, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 8)
        self.bn2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, num_classes)

    def forward(self, x):
        x = x.to(torch.float)
        x = self.flatten(x)
        x = self.bn1(self.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(self.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    