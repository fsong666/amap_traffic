import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, out_dim=256):
        super(ResNet, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.fcn1 = nn.Linear(resnet.fc.in_features, 512)  # 2048 -> 512
        self.bn1 = nn.BatchNorm1d(512, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        self.fcn2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.01)

        self.fcn3 = nn.Linear(512, out_dim)
        self.out_dim = out_dim

    def forward(self, x_3d):
        out = []
        for t in range(x_3d.size()[1]):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # bs x 2048 x 1 x 1
                # print('x=', x.shape)                # bs x 2048
                x = x.view(-1, 2048)

            x = self.fcn1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.fcn2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.fcn3(x)

            # x = self.fcn3(x)
            # print('x=', x.shape)

            out.append(x)

        # x.shape= (time, batch, predict)
        x = torch.stack(out, dim=0)   # time x bs x 256

        # # x.shape= (batch, time, predict)
        # x = x.transpose(0, 1)
        # x = x.mean(1)

        return x


class LSTM(nn.Module):
    def __init__(self, hidden_size=256, out_dim=128):
        super(LSTM, self).__init__()
        self.resnet = ResNet()
        self.lstm = nn.LSTM(input_size=self.resnet.out_dim, hidden_size=hidden_size, num_layers=3,
                            dropout=0.5, bidirectional=False)
        self.fcn1 = nn.Linear(hidden_size, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)

        self.fcn2 = nn.Linear(out_dim, 3)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.resnet(x)
        self.lstm.flatten_parameters()
        x, (hn, cn) = self.lstm(x)  # time x bs x hidden_size
        # print('lstm_output=', x.shape)

        x = self.fcn1(x[-1])        # bs x hidden_size
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fcn2(x)            # bs x 3
        x = self.log_softmax(x)

        return x


class Traffic(nn.Module):
    """
    H_{out}= floor((n - k + 2) / stride + 1
    """

    def __init__(self):
        super(Traffic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fcn1 = nn.Linear(10 * 22 * 40, 120)
        self.fcn2 = nn.Linear(120, 84)
        self.fcn3 = nn.Linear(84, 3)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.validation = False

    def set_validation(self, validation=False):
        self.validation = validation

    def forward(self, x):

        x = x.view(-1, 3, 720, 1280)  # bs x 3  x 720 x 1280
        x = self.relu(self.conv1(x))  # bs x 6  x 720 x 1280
        x = self.pool1(x)  # bs x 6  x 360 x 640

        x = self.relu(self.conv2(x))  # bs x 16 x 180 x 320
        x = self.pool1(x)  # bs x 16 x  90 x 160

        x = self.relu(self.conv3(x))  # bs x 10 x 45 x 80
        x = self.pool1(x)  # bs x 10 x  22 x 40

        # x = x.view(-1, 10 * 22 * 40)
        x = torch.flatten(x, 1)
        x = self.relu(self.fcn1(x))  # bs x 120
        x = self.relu(self.fcn2(x))  # bs x 84
        x = self.fcn3(x)  # bs x 3
        if self.validation:
            x = self.softmax(x)
        return x


