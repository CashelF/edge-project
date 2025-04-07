"""
Conv5 in PyTorch.
See the paper "FedMAX: Mitigating Activation Divergence for Accurate and Communication-Efficient Federated Learning"
for more details.
Reference: https://github.com/weichennone/FedMAX/blob/master/digit_object_recognition/models/Nets.py
"""

from torch import nn
import torch.nn.functional as F


class Conv5(nn.Module):

    def __init__(self, num_classes=10, channels=1, loss_type='fedavg'):
        super(Conv5, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.loss_type = loss_type

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))
        x_out = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x_out))
        x = self.fc2(x)
        if self.loss_type == 'fedmax':
            return x, x_out
        else:
            return x


class Conv5_small(nn.Module):
    def __init__(self, num_classes=10, channels=1, loss_type='fedavg'):
        super(Conv5_small, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.loss_type = loss_type

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))
        x_out = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x_out))
        x = self.fc2(x)
        if self.loss_type == 'fedmax':
            return x, x_out
        else:
            return x
        
class Conv5_small_BN(nn.Module):
    def __init__(self, num_classes=10, channels=1, loss_type='fedavg'):
        super(Conv5_small_BN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.loss_type = loss_type

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(F.relu(self.bn5(self.conv5(x))))
        x_out = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x_out))
        x = self.fc2(x)
        if self.loss_type == 'fedmax':
            return x, x_out
        else:
            return x