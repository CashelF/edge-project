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
        

class Conv3_small_BN(nn.Module):
    def __init__(self, num_classes=10, channels=1, loss_type='fedavg'):
        super(Conv3_small_BN, self).__init__()
        # First convolution: channels -> 16
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolution: 16 -> 32
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolution: 32 -> 32
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Assuming input images are 32x32 pixels, after 3 poolings (each halving the side)
        # the feature map size becomes 4x4. Thus, the flattened dimension is 32*4*4.
        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.loss_type = loss_type

    def forward(self, x):
        # Apply conv1, bn, relu, and pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Apply conv2, bn, relu, and pooling
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Apply conv3, bn, relu, and pooling
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Flatten the output
        x_out = x.view(x.size(0), -1)
        # Pass through fully-connected layers
        x = F.relu(self.fc1(x_out))
        x = self.fc2(x)
        if self.loss_type == 'fedmax':
            return x, x_out
        else:
            return x
        
class Conv2_small_BN(nn.Module):
    """
    2‐layer ConvNet with BatchNorm, mirroring the style of our other *_BN models.
    Assumes inputs are 32×32 (→ pools to 16×16 → 8×8).
    """
    def __init__(self, num_classes=10, channels=1, loss_type='fedavg'):
        super(Conv2_small_BN, self).__init__()
        # conv1: channels → 16
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # conv2: 16 → 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # FCs: flatten 32×8×8 → 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.loss_type = loss_type

    def forward(self, x):
        # conv → BN → ReLU → pool
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # conv → BN → ReLU → pool
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # flatten
        x_out = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x_out))
        x = self.fc2(x)
        
        # if using fedmax, also return features before FC
        if self.loss_type == 'fedmax':
            return x, x_out
        return x
        
class Conv1_small_BN(nn.Module):
    def __init__(self, num_classes=10, channels=1, loss_type='fedavg'):
        super(Conv1_small_BN, self).__init__()
        # Single conv layer: in channels → 16 feature maps
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # halves 32×32 → 16×16
        
        # After pool: 16 channels × 16 × 16 spatial = 4096 features
        self.fc1       = nn.Linear(16 * 16 * 16, 256)
        self.fc2       = nn.Linear(256, num_classes)
        self.loss_type = loss_type

    def forward(self, x):
        # conv → BN → ReLU → pool
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # flatten
        x_out = x.view(x.size(0), -1)    # (batch, 4096)
        
        # FC layers
        x = F.relu(self.fc1(x_out))
        x = self.fc2(x)
        
        # for fedmax, also return the pre-FC feature vector
        if self.loss_type == 'fedmax':
            return x, x_out
        return x