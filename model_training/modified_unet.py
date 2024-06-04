import torch
import torch.nn as nn
from utils import unetConv2, unetUp

class ModifiedUNet(nn.Module):

    def __init__(self, feature_scale=4, is_deconv=True, in_channels=4, is_batchnorm=True):
        super(ModifiedUNet, self).__init__()
        self.is_deconv = is_deconv
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

        # Flattened size calculation
        center_size = 42 * 48 * filters[4]

        # Fully connected layers for bounding box regression
        self.fc1 = nn.Sequential(
            nn.Linear(center_size, 1024), nn.BatchNorm1d(1024), nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU()
        )
        self.fc3 = nn.Sequential(nn.Linear(256, 4))

        # # Upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # output convolutional layer
        self.final = nn.Sequential(
            nn.Conv2d(filters[0], 32, 3, padding=1),
            nn.ReLU(inplace=True), # applies relu for non linearity
            nn.AdaptiveAvgPool2d((1, 1)), # taking avg of feature map reducing the spatial dimensions to 1, 1
            nn.Flatten(), # -> [dataset, 32]
            nn.Linear(32, 4)  # output for two points (x_start, y_start, x_end, y_end)
        )

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

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        return final


# dummy_input = torch.randn(2, 4, 672, 768)  # Batch size of 2
# model = ModifiedUNet()
# output = model(dummy_input)
# print(output.size())
# print(output)
