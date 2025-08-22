import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionCNN(nn.Module):
    def __init__(self, dropout=0.5, hidden_units=128):
        super(FashionCNN, self).__init__()
        # Layer 1: Conv -> ReLU -> Pool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Layer 2: Conv -> ReLU -> Pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        # Layer 3: Fully Connected
        self.fc1 = nn.Linear(64 * 6 * 6, hidden_units)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_units, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
