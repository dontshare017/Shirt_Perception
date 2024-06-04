import torch
import torch.nn as nn
from utils import unetConv2

class CoordinateUNet(nn.Module):
    # def __init__(self, feature_scale=4, in_channels=4, is_batchnorm=True):    # uncomment for regular RGB images
    def __init__(self, feature_scale=4, in_channels=2, is_batchnorm=True):
        super(CoordinateUNet, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        center_size = 42 * 48 * filters[4]

        self.fc1 = nn.Linear(center_size, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(256, 4)  # coordinates of 4 keypoints

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        flattened = center.view(center.size(0), -1)

        x = self.fc1(flattened)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        coordinates = self.output(x)

        return coordinates

# dummy_input = torch.randn(2, 4, 672, 768)  # Batch size of 2
# model = CoordinateUNet()
# output = model(dummy_input)
# print(output.size())
# print(output)
