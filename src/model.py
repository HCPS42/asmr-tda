import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=8, kernel_size=60, stride=1)
        self.fc1 = nn.Linear(8, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(torch.float)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
