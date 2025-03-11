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
import json

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.Torchio_contrast_dataloader import custom_collate_fn
from dataloaders.get_loader import get_test_classify_loader
from networks.get_network import get_network, load_pretrained
from util.logger import Logger

def test(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    current_file_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(current_file_directory)
    log_dir = os.path.join(parent_directory, "runs", "test_" + current_time + "_" + socket.gethostname())
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logger = Logger(log_path=writer.log_dir)

    # ==== 1. Initialize parameters ====
    logger.print('1. Initialize parameters')
    cudnn.benchmark = True
    cudnn.deterministic = True
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    
    class_num = args.class_num
    class_num_index = []
    start = 0
    for c in class_num:
        end = start + c
        class_num_index.append((start, end))
        start = end
    
    # ==== 2. Load model ====
    logger.print('2. Create & load model')
    net = get_network(args).to(device)
    
    if args.pretrained_model_path is not None and os.path.isfile(args.pretrained_model_path):
        net = load_pretrained(args, net, optimizer=None, logger=logger)  
    else:
        logger.print("No valid pretrained model path provided, or file not found. Using model with random init.")
        raise ValueError("No valid pretrained model path provided, or file not found. Using model with random init.")
        
    net.eval() 

    # ==== 3. Load validation data ====
    logger.print('3. Load validation data')
    validData = get_test_classify_loader(args)
    
    validLoader = DataLoader(
        validData,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    # ==== 4. Start testing ====
    logger.print('4. Start testing')
    valid_acc_by_group = [[] for _ in range(len(class_num_index))]

    all_predictions = []

    with torch.no_grad():
        for sample_batch in validLoader:
            sample_path = sample_batch.get('image_path')[0]
            img_batch = sample_batch['image'].to(device)
            label_batch = torch.tensor(sample_batch['label']).to(device)
            label_batch = F.one_hot(label_batch, num_classes=class_num[0])

            with amp.autocast():
                pred_logit = net(img_batch)

            group_pred_classes = []
            group_target_classes = []
            
            for group_i, (start, end) in enumerate(class_num_index):
                group_logit = pred_logit[:, start:end]  
                group_label = label_batch[:, start:end]
                
                _, group_target = torch.max(group_label, dim=1)  
                group_pred_class = torch.argmax(group_logit, dim=1)
                
                correct = (group_pred_class == group_target).sum().item()
                acc = correct / group_target.numel()
                valid_acc_by_group[group_i].append(acc)

                group_pred_classes.append(int(group_pred_class.item()))
                group_target_classes.append(int(group_target.item()))
            
            sample_pred_info = {
                "sample_path": sample_path,
                "prediction": group_pred_classes, 
                "target": group_target_classes,  
                "prediction_prob": torch.softmax(pred_logit, dim=1).detach().cpu().tolist()
            }
            all_predictions.append(sample_pred_info)

    group_acc = [float(np.mean(acc_list)) for acc_list in valid_acc_by_group]
    overall_acc = float(np.mean(group_acc))

    for i, g_acc in enumerate(group_acc):
        writer.add_scalar(f'Validation/Acc_Group_{i+1}', g_acc)
        logger.print(f'Validation - Group {i+1} Acc: {g_acc:.4f}')
    writer.add_scalar('Validation/Acc_Overall', overall_acc)
    logger.print(f'Validation - Overall Acc: {overall_acc:.4f}')
    
    logger.print("Testing Finished.")
    writer.close()

    # ==== 5. Save results ====
    if args.output_json is not None:
        json_result = {
            "metrics": {
                "group_acc": group_acc,
                "overall_acc": overall_acc
            },
            "predictions": all_predictions
        }
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        logger.print(f"Results have been saved to {args.output_json}")


def get_args():
    parser = argparse.ArgumentParser(description="Testing arguments")

    parser.add_argument('--valid_image_list', type=str, default='./CT_data/valid.txt', help='Path to the text file containing validation images.')
    parser.add_argument('--valid_label_list', type=str, default='./CT_data/valid_label.txt', help='Path to the text file containing validation labels.')
    
    parser.add_argument('--net_type', type=str, default='resnet50', help='Network architecture.')
    parser.add_argument('--input_channel', default=1, type=int, help='Number of input channels.')
    parser.add_argument('--output_channel', default=2, type=int, help='Number of output channels (e.g., segmentation classes).')
    parser.add_argument('--base_feature_number', default=32, type=int, help='Base number of feature maps.')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to a pretrained model checkpoint.')
    
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training.')
    parser.add_argument('--class_num', nargs='+', type=int, default=[5, 3, 2, 2, 2], help='Number of classes for each group, e.g., 5 3 2 2 2')

    parser.add_argument('--crop_shape', nargs='+', type=int, default=[96, 96, 96], help='Crop shape (if needed).')

    parser.add_argument('--output_json', type=str, default=None, help='Path to save predictions and metrics in JSON format.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    
    test(args)
