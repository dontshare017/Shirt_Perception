
import torch
import torch.nn as nn
import torch.nn.functional as F

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, True)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)

        offset = - outputs2.size()[3] + inputs1.size()[3]
        offset2 = - outputs2.size()[2] + inputs1.size()[2]
        if offset % 2:
            if offset2 % 2:
                padding = [offset // 2, offset // 2 + 1, offset2 // 2, offset2 // 2 + 1]
            else:
                padding = [offset // 2, offset // 2 + 1, offset2 // 2, offset2 // 2]
            #padding = 2 * [offset // 2, offset // 2 + 1]
        else:
            if offset2 % 2:
                padding = [offset // 2, offset // 2, offset2 // 2, offset2 // 2 + 1]
            else:
                padding = [offset // 2, offset // 2, offset2 // 2, offset2 // 2 ]
            #padding = 2 * [offset // 2, offset // 2]
        #print("-----")
        #print(np.shape(inputs1))
        #print(padding)

        outputs2 = F.pad(outputs2, padding)
        #print(np.shape(inputs1))
        #print(np.shape(outputs2))
        return self.conv(torch.cat([inputs1, outputs2], 1))
    """
    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)

        offset = outputs2.size()[3] - inputs1.size()[3]
        offset2 = outputs2.size()[2] - inputs1.size()[2]
        if offset % 2:
            if offset2 % 2:
                padding = [offset // 2, offset // 2 + 1, offset2 // 2, offset2 // 2 + 1]
            else:
                padding = [offset // 2, offset // 2 + 1, offset2 // 2, offset2 // 2]
            #padding = 2 * [offset // 2, offset // 2 + 1]
        else:
            if offset2 % 2:
                padding = [offset // 2, offset // 2, offset2 // 2, offset2 // 2 + 1]
            else:
                padding = [offset // 2, offset // 2, offset2 // 2, offset2 // 2 ]
            #padding = 2 * [offset // 2, offset // 2]
        print("-----")
        print(np.shape(inputs1))
        print(padding)

        outputs1 = F.pad(inputs1, padding)
        print(np.shape(outputs1))
        print(np.shape(outputs2))
        return self.conv(torch.cat([outputs1, outputs2], 1))
    """