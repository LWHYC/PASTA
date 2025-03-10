import sys
import argparse
import warnings
import os
warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.functional as F

from networks.generic_UNet import Generic_UNet, Generic_UNet_classify
from networks.ModelsGenesis import UNet3D_modelgenesis_classify, UNet3D_modelgenesis
from networks.SupreM import UNet3D_suprem_classify, UNet3D_suprem

def get_network(args):
    ###### Create UNet ######
    if args.net_type in['Generic_UNet']:
        num_input_channels = args.input_channel
        base_num_features = args.base_feature_number
        conv_per_stage = 2
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        net_num_pool_op_kernel_sizes = net_conv_kernel_sizes = None

        net = Generic_UNet(num_input_channels, base_num_features, args.output_channel, 5,
                            conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                            dropout_op_kwargs,
                            net_nonlin, net_nonlin_kwargs, True, False,
                            net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True).float().cuda()
        
    elif args.net_type in['Generic_UNet_classify']:
        num_input_channels = args.input_channel
        base_num_features = args.base_feature_number
        conv_per_stage = 2
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        net_num_pool_op_kernel_sizes = net_conv_kernel_sizes = None

        net = Generic_UNet_classify(num_input_channels, base_num_features, args.output_channel, 5,
                            conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                            dropout_op_kwargs,
                            net_nonlin, net_nonlin_kwargs, True, False,
                            net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True).float().cuda()
    elif args.net_type in['UNet3D_modelgenesis']:
        net = UNet3D_modelgenesis(n_class=args.output_channel)
    elif args.net_type in['UNet3D_modelgenesis_classify']:
        net = UNet3D_modelgenesis_classify(n_class=args.output_channel)
    elif args.net_type in['UNet3D_suprem']:
        net = UNet3D_suprem(input_channels=1, n_class=args.output_channel)
    elif args.net_type in['UNet3D_suprem_classify']:
        net = UNet3D_suprem_classify(input_channels=1, n_class=args.output_channel)
    
    return net

def load_pretrained(args, net, logger):
    try:
        logger.print("=> loading checkpoint '{}'".format(args.pretrained_model_path))

        # Load checkpoint on CPU to avoid GPU 0 memory spike
        checkpoint = torch.load(args.pretrained_model_path, map_location='cpu')
        if args.pretrained_model_path.endswith('tar'):
                pretrained_dict = checkpoint['state_dict']
        else:
            if args.net_type in ['Generic_UNet_classify', 'Generic_UNet']:
                pretrained_dict = checkpoint['network_weights']
            elif args.net_type in ['UNet3D_modelgenesis', 'UNet3D_modelgenesis_classify']:
                pretrained_dict = checkpoint
            elif args.net_type in ['UNet3D_suprem', 'UNet3D_suprem_classify']:
                pretrained_dict = checkpoint['net']
                pretrained_dict = {key.replace('module.backbone.', ''): pretrained_dict[key] for key in pretrained_dict.keys()}
        # optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Filter out mismatched shapes
        model_dict = net.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

        # Update model state_dict with filtered pretrained weights
        model_dict.update(filtered_dict)
        net.load_state_dict(model_dict, strict=False)

        # Log the number of skipped parameters
        skipped_params = set(pretrained_dict.keys()) - set(filtered_dict.keys())
        logger.print(f"Skipped loading {skipped_params} due to shape mismatch.")
        
        
        logger.print("=> loaded checkpoint '{}' ".format(args.pretrained_model_path))
        # Clean up
        del checkpoint, filtered_dict, pretrained_dict
        torch.cuda.empty_cache()
    except Exception as e:
        logger.print(e.message, e.args)
        raise ValueError
        
    return net