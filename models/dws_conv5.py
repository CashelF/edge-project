import torch.nn.functional as F
from torch import nn


class DepthWiseConv(nn.Module):

    def __init__(self, in_fts, stride=(1, 1)):
        super(DepthWiseConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_fts,
                in_fts,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                groups=in_fts,
            ),
            nn.BatchNorm2d(in_fts),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x


class PointWiseConv(nn.Module):

    def __init__(self, in_fts, out_fts):
        super(PointWiseConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, out_fts, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_fts),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x


class DepthWiseSeperableConv(nn.Module):

    def __init__(self, in_fts, out_fts, stride=(1, 1)):
        super(DepthWiseSeperableConv, self).__init__()
        self.depthwise = DepthWiseConv(in_fts, stride=stride)
        self.pointwise = PointWiseConv(in_fts, out_fts)

    def forward(self, input_image):
        x = self.pointwise(self.dw(input_image))
        return x


"""
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
"""


class Conv5_small_BN_DWS(nn.Module):

    def __init__(self, num_classes=10, channels=1, loss_type="fedavg"):
        super(Conv5_small_BN_DWS, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = DepthWiseSeperableConv(16, 32)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = DepthWiseSeperableConv(32, 32)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv4 = DepthWiseSeperableConv(32, 32)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = DepthWiseSeperableConv(32, 32)
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
        if self.loss_type == "fedmax":
            return x, x_out
        else:
            return x


"""
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
"""


class Conv3_small_BN_DWS(nn.Module):

    def __init__(self, num_classes=10, channels=1, loss_type="fedavg"):
        super(Conv3_small_BN_DWS, self).__init__()
        # First convolution: channels -> 16
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second convolution: 16 -> 32
        self.conv2 = DepthWiseSeperableConv(16, 32)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third convolution: 32 -> 32
        self.conv3 = DepthWiseSeperableConv(32, 32)
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
        if self.loss_type == "fedmax":
            return x, x_out
        else:
            return x
