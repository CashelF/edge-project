import torch.nn.functional as F
from torch import nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        # depthwise
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        # pointwise
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class Conv5_small_DWS(nn.Module):
    def __init__(self, num_classes=10, channels=1, loss_type="fedavg"):
        super(Conv5_small_DWS, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = SeparableConv2d(16, 32)
        self.conv3 = SeparableConv2d(32, 32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv4 = SeparableConv2d(32, 32)
        self.conv5 = SeparableConv2d(32, 32)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.loss_type = loss_type

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.conv2(x)
        x = self.pool2(self.conv3(x))
        x = self.conv4(x)
        x = self.pool3(self.conv5(x))

        x_out = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x_out))
        x = self.fc2(x)
        if self.loss_type == "fedmax":
            return x, x_out
        else:
            return x


class Conv3_small_DWS(nn.Module):
    def __init__(self, num_classes=10, channels=1, loss_type="fedavg"):
        super(Conv3_small_DWS, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = SeparableConv2d(16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = SeparableConv2d(32, 32)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.loss_type = loss_type

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))

        x_out = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x_out))
        x = self.fc2(x)
        if self.loss_type == "fedmax":
            return x, x_out
        else:
            return x
