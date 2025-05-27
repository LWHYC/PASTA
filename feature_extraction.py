#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import sys
import argparse
import warnings
import os
warnings.filterwarnings('ignore')

import time
import socket
from datetime import datetime
import random
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.functional as F


from classification.networks.get_network import get_network, load_pretrained
from classification.dataloaders.Torchio_contrast_dataloader import MultiThreshNormalized
from classification.networks.generic_UNet import Generic_UNet_classify
from classification.dataloaders.data_process_func import img_multi_thresh_normalized_torch

def get_pasta():
    num_input_channels = 1
    base_num_features = 64
    conv_per_stage = 2
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
    net_num_pool_op_kernel_sizes = net_conv_kernel_sizes = None

    net = Generic_UNet_classify(num_input_channels, base_num_features, 1, 5,
                        conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                        dropout_op_kwargs,
                        net_nonlin, net_nonlin_kwargs, True, False,
                        net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True).float().cuda()
    return net




#### Example code for feature extraction using PASTA ####
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    checkpoint_path = '/data/leiwenhui/Code/PASTA_final.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net = get_pasta()
    
    # Filter out mismatched shapes
    model_dict = net.state_dict()
    filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}

    # Update model state_dict with filtered pretrained weights
    model_dict.update(filtered_dict)
    net.load_state_dict(model_dict, strict=False)
    
    # Log the number of skipped parameters. 
    # !!!!! It's okay to skip the head/seg_outputs layers for feature extraction !!!!!
    skipped_params = set(checkpoint.keys()) - set(filtered_dict.keys())
    print(f"Skipped loading {skipped_params} due to shape mismatch.")
    
    net.eval()
    net.cuda()
    
    ct_img = torch.randint(low=-1000, high=1000, size=(1,1,128,128,128)).cuda() # shape need to be divisible by 32, since pasta have 4 pooling layers
    thresh_ls = [-1000, -200, 200, 1000]
    norm_ls = [0,0.2,0.8,1]
    ct_img = img_multi_thresh_normalized_torch(ct_img, thresh_ls, norm_ls, data_type=torch.float32)
    
    with torch.no_grad():
        features = net(ct_img, output_feature=True)
        print(features.shape)
