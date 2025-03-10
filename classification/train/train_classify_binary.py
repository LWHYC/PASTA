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
import sklearn
from sklearn.metrics import roc_auc_score

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.Torchio_contrast_dataloader import custom_collate_fn
from dataloaders.get_loader import get_classify_binary_loader
from networks.get_network import get_network, load_pretrained
from util.save_model import Save_checkpoint, move_optimizer_to_cpu
from util.logger import Logger
from losses.loss_function import MultiClassCrossEntropy_Loss

def train(args, rank, world_size):
    
    torch.cuda.set_device(rank)
    
    # ========== 1. Create logger and tensorboard writer ==========
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    current_file_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(current_file_directory)
    
    log_dir = os.path.join(
        parent_directory, "runs", current_time + "_" + socket.gethostname()
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logger = Logger(log_path=writer.log_dir)
    
    # ========== 2. Initialize parameters ==========
    logger.print('1. Initialize parameters')
    cudnn.benchmark = True
    cudnn.deterministic = True

    random.seed(rank)
    np.random.seed(rank)
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    
    # ========== 3. Load parameters ==========
    batch_size          = args.batch_size
    num_workers         = args.num_workers
    class_num           = args.class_num
    lr                  = args.learning_rate
    decay               = args.decay
    args.total_step    *= world_size
    start_step          = args.start_step
    save_step           = args.save_step
    accumulation_steps  = args.accumulation_steps
    model_save_name     = args.model_save_name
    class_num_index = []
    class_index_start = 0
    for c in class_num:
        class_index_end = class_index_start + c
        class_num_index.append((class_index_start, class_index_end))
        class_index_start = class_index_end

    os.makedirs(os.path.dirname(model_save_name), exist_ok=True)
    save_model = Save_checkpoint()

    # ========== 4. Build DataLoader ==========
    logger.print('2. Load data')
    trainData, validData = get_classify_binary_loader(args)
    
    trainSampler = DistributedSampler(trainData, num_replicas=world_size, rank=rank, shuffle=True)
    trainLoader = DataLoader(
        trainData,
        batch_size=batch_size,
        sampler=trainSampler,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        shuffle=False
    )

    validLoader = DataLoader(
        validData,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # ========== 4. Create model ==========
    logger.print('3. Create model')
    net = get_network(args).cuda(rank)
    ddp_model = DDP(net, device_ids=[rank], find_unused_parameters=True, output_device=rank)
    # ========== 5. Optimizer, Scheduler, GradScaler ==========
    optimizer = optim.Adam(ddp_model.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainLoader))
    scaler = GradScaler()
    train_step = start_step

    # ========== 6. Load pretrained model ==========
    if args.pretrained_model_path is not None:
        net = load_pretrained(args, net, logger)

    # ========== 7. Resume training ==========
    if isinstance(class_num, list):
        loss_fn = MultiClassCrossEntropy_Loss(multiclass_num=class_num, weight_ls=args.class_weight)
    else:
        loss_fn = MultiClassCrossEntropy_Loss(multiclass_num=[class_num], weight_ls=args.class_weight)
    
    

    # ========== 8. Start training ==========
    logger.print('4. Start training')
    best_auc = 0.0
    
    while train_step < len(trainLoader):
        ddp_model.train()

        for i_batch, sample_batch in enumerate(trainLoader):
            img_batch = sample_batch['image'].cuda(rank)
            label_batch = torch.tensor(sample_batch['label']).cuda(rank)
            label_batch = F.one_hot(label_batch, num_classes=class_num[0])
            if i_batch % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            with autocast():
                pred_logit = ddp_model(img_batch)
                loss_value = loss_fn(pred_logit, label_batch)

            scaler.scale(loss_value).backward()

            # Accumulate gradients
            if (i_batch + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_step += 1

                # ============= Print for every log_freq steps =============
                if rank == 0 and (i_batch + 1) % args.log_freq == 0:
                    progress_epoch = (i_batch + 1) / len(trainLoader)
                    progress_percent = 100.0 * progress_epoch
                    cur_loss = loss_value.detach().cpu().item()
                    logger.print(
                        f"Batch [{i_batch+1}/{len(trainLoader)}] "
                        f"({progress_percent:.2f}%) "
                        f"Loss: {cur_loss:.4f}"
                    )

                

                if train_step % save_step == 0:
                    ddp_model.eval()
                    
                    # ============= Validate =============
                    valid_probs_by_group = [[] for _ in range(len(class_num_index))]
                    valid_labels_by_group = [[] for _ in range(len(class_num_index))]
                    valid_acc_by_group = [[] for _ in range(len(class_num_index))]
                    
                    with torch.no_grad():
                        for j_batch, valid_sample in enumerate(validLoader):
                            valid_img = valid_sample['image'].cuda(rank)
                            valid_label = torch.tensor(valid_sample['label']).cuda(rank)
                            valid_label = F.one_hot(valid_label, num_classes=class_num[0])
                            valid_pred_logit = ddp_model(valid_img)
                            
                            for group_i, (start, end) in enumerate(class_num_index):
                                # Extract group logits and labels
                                group_logit = valid_pred_logit[:, start:end]    # shape: (B, group_class_num)
                                group_label = valid_label[:, start:end]         # same shape, one-hot
                                
                                _, group_target = torch.max(group_label, dim=1)   # shape: (B,)
                                
                                # Compute accuracy
                                group_pred_class = torch.argmax(group_logit, dim=1)  # shape: (B,)
                                correct = (group_pred_class == group_target).sum().item()
                                acc = correct / group_target.numel()
                                valid_acc_by_group[group_i].append(acc)
                                
                                # Collect probabilities and labels
                                if group_logit.shape[1] == 2:
                                    prob = torch.softmax(group_logit, dim=1)[:,1]  # shape: (B,)
                                    valid_probs_by_group[group_i].append(prob.detach().cpu().numpy())
                                    valid_labels_by_group[group_i].append(group_target.detach().cpu().numpy())
                                else:
                                    pass
                    
                    group_acc = [float(np.mean(acc_list)) for acc_list in valid_acc_by_group]
                    overall_acc = float(np.mean(group_acc))

                    if rank == 0:
                        # Calculate AUC
                        group_auc = []
                        for group_i in range(len(class_num_index)):
                            if valid_probs_by_group[group_i]:
                                all_probs = np.concatenate(valid_probs_by_group[group_i], axis=0)
                                all_labels = np.concatenate(valid_labels_by_group[group_i], axis=0)
                                if len(np.unique(all_labels)) == 2:
                                    auc_val = roc_auc_score(all_labels, all_probs)
                                else:
                                    auc_val = 0.0  # or skip
                            else:
                                auc_val = 0.0
                            group_auc.append(auc_val)
                        
                        overall_auc = float(np.mean(group_auc))

                        # Group Acc & AUC
                        for i, (g_acc, g_auc) in enumerate(zip(group_acc, group_auc)):
                            writer.add_scalar(f'Validation/Acc_Group_{i+1}', g_acc, train_step)
                            writer.add_scalar(f'Validation/AUC_Group_{i+1}', g_auc, train_step)
                            logger.print(f'Validation - Group {i+1} Acc: {g_acc:.4f}, AUC: {g_auc:.4f}')
                        
                        # Overall
                        writer.add_scalar('Validation/Acc_Overall', overall_acc, train_step)
                        writer.add_scalar('Validation/AUC_Overall', overall_auc, train_step)
                        logger.print(f'Validation - Overall Acc: {overall_acc:.4f}, Overall AUC: {overall_auc:.4f}')

                        # Save model
                        filename = f"{model_save_name}_latest_Auc.tar"
                        if overall_auc > best_auc:
                            best_auc = overall_auc
                            is_best = True
                            bestname = f"{model_save_name}_best_Auc.tar"
                        else:
                            is_best = False
                            bestname = None
                        
                        state = {
                            'train_step': train_step,
                            'state_dict': {k: v.cpu() for k, v in ddp_model.module.state_dict().items()},
                            'best_auc': best_auc,
                            'optimizer': move_optimizer_to_cpu(optimizer)
                        }
                        save_model.save_checkpoint(state=state, is_best=is_best, filename=filename, logger=logger, bestname=bestname)
                        
                        del state
                        torch.cuda.empty_cache()
                        
                    ddp_model.train()

                if train_step >= len(trainLoader):
                    break

    logger.print("Training Finished.")
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description="Training arguments")

    # ========== Data parameters ==========
    parser.add_argument('--train_image_list', type=str, default='./CT_data/train.txt', help='Path to the text file containing training images.')
    parser.add_argument('--train_label_list', type=str, default='./CT_data/train.txt', help='Path to the text file containing training images.')
    parser.add_argument('--valid_image_list', type=str, default='./CT_data/valid.txt', help='Path to the text file containing validation images.')
    parser.add_argument('--valid_label_list', type=str, default='./CT_data/valid.txt', help='Path to the text file containing validation images.')

    # ========== Network parameters ==========
    parser.add_argument('--net_type', type=str, default='resnet50', help='Network architecture (e.g., "resnet50", "unet", etc.).')
    parser.add_argument('--input_channel', default=1, type=int, help='Number of input channels (e.g., 1 for grayscale, 3 for RGB).')
    parser.add_argument('--output_channel', default=2, type=int, help='Number of output channels (e.g., for segmentation classes).')
    parser.add_argument('--base_feature_number', default=32, type=int, help='Base number of feature maps for the network (e.g., 32, 64).')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to a pretrained model checkpoint.')
    parser.add_argument('--model_save_name', type=str, default=None, help='Path to save model checkpoint.')

    # ========== Training parameters ==========
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for the DataLoader.')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Initial learning rate.')
    parser.add_argument('--decay', default=1e-5, type=float, help='Weight decay for the optimizer.')
    parser.add_argument('--total_step', default=1000, type=int, help='Total number of training steps.')
    parser.add_argument('--start_step', default=0, type=int, help='Starting step (useful when resuming training).')
    parser.add_argument('--save_step', default=100, type=int, help='Save model checkpoint every N steps.')
    parser.add_argument('--log_freq', default=100, type=int, help='Save model checkpoint every N steps.')
    parser.add_argument('--accumulation_steps', default=2, type=int, help='Number of steps to accumulate gradients before updating.')
    parser.add_argument('--class_num', nargs='+', type=int, default=[1], help='Number of classes. Can be a single int or multiple ints for multi-head tasks.')
    parser.add_argument('--class_weight', nargs='+', type=int, default=[1], help='Weight of classes for CE loss. Can be a single int or multiple ints for multi-head tasks.')
    parser.add_argument('--crop_shape', nargs='+', type=int, default=[96, 96, 96], help='Crop shape (e.g., 96 96 96).')
    parser.add_argument('--load_memory', action='store_true', help='Whether to load memory features (if applicable).')
    parser.add_argument('--memory_length', type=int, default=0, help='Length of memory if load_memory is True.')
    

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    # ========== Start training ==========
    train(args, rank, world_size)
