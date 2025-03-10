from __future__ import print_function
from networks.module import Module
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from networks.layers import SingleConv3D
import math
import time

class Unet_seg(Module):
    def __init__(self,  inc=1, n_classes=5, base_chns=16, droprate=0, norm='in', depth=False, dilation=1):
        super(Unet_seg, self).__init__()
        self.model_name = "seg"
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')  # 1/4(h,h)
        self.downsample = nn.MaxPool3d(2, 2)  # 1/2(h,h)
        self.drop = nn.Dropout(droprate)

        self.conv1_1 = SingleConv3D(inc, base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_2 = SingleConv3D(base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv2_1 = SingleConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_2 = SingleConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv3_1 = SingleConv3D(4*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_2 = SingleConv3D(4 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv4_1 = SingleConv3D(8*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')
        self.conv4_2 = SingleConv3D(8 * base_chns, 16 * base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')

        self.conv5_1 = SingleConv3D(24*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_2 = SingleConv3D(8 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv6_1 = SingleConv3D(12*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_2 = SingleConv3D(4 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv7_1 = SingleConv3D(6*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_2 = SingleConv3D(4* base_chns, 4* base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.classification = nn.Conv3d(in_channels=4*base_chns, out_channels=n_classes, kernel_size=1)
        
    def forward(self, x, out_seg=False, inference=False):
        result = {}
        
        out = self.conv1_1(x)
        conv1 = self.conv1_2(out)
        out = self.downsample(conv1)  # 1/2
        out = self.conv2_1(out)
        conv2 = self.conv2_2(out)  #
        out = self.downsample(conv2)  # 1/4
        out = self.conv3_1(out)
        conv3 = self.conv3_2(out)  #
        out = self.downsample(conv3)  # 1/8
        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.drop(out)

        up5 = self.upsample(out)  # 1/4
        out = t.cat((up5, conv3), 1)
        out = self.conv5_1(out)
        out = self.conv5_2(out)

        up6 = self.upsample(out)  # 1/2
        out = t.cat((up6, conv2), 1)
        out = self.conv6_1(out)
        out = self.conv6_2(out)
                
        up7 = self.upsample(out)
        out = t.cat((up7, conv1), 1)
        out = self.conv7_1(out)
        out = self.conv7_2(out)
        
        
        if out_seg:
            out = self.classification(out)
            if inference:
                predic = F.sigmoid(out)
                result['prob'] = predic
            else:
                result['logit'] = out
        
        return result
