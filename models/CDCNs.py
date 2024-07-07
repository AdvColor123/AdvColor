'''
Code of 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing'
By Zitong Yu & Zhuo Su, 2019

If you use the code, please cite:
@inproceedings{yu2020searching,
    title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
    author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
    booktitle= {CVPR},
    year = {2020}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2020
'''

import math
import torch
import torch.nn.functional as F
from torch import nn


# Centeral-difference (second order, with 9 param
# | a1 a2 a3 |   | w1 w2 w3 |
# | a4 a5 a6 | * | w4 w5 w6 | --> output = \sum_{i=1}^{9}(ai * wi) - \sum_{i=1}^{9}wi * a5 --> Conv2d (k=3) - Conv2d (k=1)
# | a7 a8 a9 |   | w7 w8 w9 |
##
# --> output =
# | a1 a2 a3 |   |  w1  w2  w3 |
# | a4 a5 a6 | * |  w4  w5  w6 |  -  | a | * | w\_sum |     (kernel_size=1x1, padding=0)
# | a7 a8 a9 |   |  w7  w8  w9 |

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, 
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, 
                                stride=self.conv.stride, padding=0, 
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class SpatialAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel//2, 
                               bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


class CDCNpp(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.0, transform=None, 
                 device='cuda:0'):   
        super(CDCNpp, self).__init__()
        self.transform = transform
        self.device = device
        self.mean = torch.Tensor([0.5, 0.5, 0.5]).to(device)
        self.std = torch.Tensor([0.5, 0.5, 0.5]).to(device)

        self.conv1 = nn.Sequential(
                    basic_conv(3, 64, kernel_size=3, stride=1, padding=1, 
                               bias=False, theta= theta), 
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
        )

        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, 
                       bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # basic_conv(128, int(128*1.6), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            # nn.BatchNorm2d(int(128*1.6)),
            # nn.ReLU(),
            # basic_conv(int(128*1.6), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
        )

        self.Block2 = nn.Sequential(
            basic_conv(128, int(128 * 1.2), kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(128 * 1.2)),
            nn.ReLU(),
            basic_conv(int(128 * 1.2), 128, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # basic_conv(128, int(128*1.4), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            # nn.BatchNorm2d(int(128*1.4)),
            # nn.ReLU(),
            # basic_conv(int(128*1.4), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
        )

        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, 
                       bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, int(128 * 1.2), kernel_size=3, stride=1, 
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(128 * 1.2)),
            nn.ReLU(),
            basic_conv(int(128 * 1.2), 128, kernel_size=3, stride=1, 
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Original

        self.lastconv1 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, 
                       bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 1, kernel_size=3, stride=1, padding=1, 
                       bias=False, theta= theta),
            nn.ReLU(),    
        )

        self.sa1 = SpatialAttention(kernel=7)
        self.sa2 = SpatialAttention(kernel=5)
        self.sa3 = SpatialAttention(kernel=3)
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

    def forward(self, x): # x [3, 256, 256]
        if self.transform is not None:
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            x = x.to(self.device).to(torch.float32)
            x = ((x / 255) - self.mean) / self.std
            x = x.permute(0, 3, 1, 2)

        x_input = x
        x = self.conv1(x)

        x_Block1 = self.Block1(x)
        attention1 = self.sa1(x_Block1)
        x_Block1_SA = attention1 * x_Block1
        x_Block1_32x32 = self.downsample32x32(x_Block1_SA)

        x_Block2 = self.Block2(x_Block1)
        attention2 = self.sa2(x_Block2)
        x_Block2_SA = attention2 * x_Block2
        x_Block2_32x32 = self.downsample32x32(x_Block2_SA)

        x_Block3 = self.Block3(x_Block2)
        attention3 = self.sa3(x_Block3)
        x_Block3_SA = attention3 * x_Block3
        x_Block3_32x32 = self.downsample32x32(x_Block3_SA)

        x_concat = torch.cat((x_Block1_32x32, x_Block2_32x32, 
                              x_Block3_32x32), dim=1)

        # pdb.set_trace()

        map_x = self.lastconv1(x_concat)

        map_x = map_x.squeeze(1)

        return map_x, x_concat, attention1, attention2, attention3, x_input
