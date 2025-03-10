# coding:utf8
import random
from inspect import TPFLAGS_IS_ABSTRACT

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import eq, gt, sum 
from torch.autograd import Variable
from torch.nn import init
from torchvision import transforms


class MultiClassCrossEntropy_Loss(nn.Module):
    def __init__(self, multiclass_num=[], weight_ls=[]):
        """
        Args:
            multiclass_num (list of int): the number of classes for each classification task.
        """
        super(MultiClassCrossEntropy_Loss, self).__init__()
        self.multiclass_num = multiclass_num
        self.totalclass_num = np.sum(multiclass_num)
        if len(weight_ls) >0:
            assert len(weight_ls) == self.totalclass_num, f"length of weight_ls {len(weight_ls)} does not match totalclass_num {self.totalclass_num}"
            self.weight_ls = weight_ls
        else:
            self.weight_ls = []
        self.indices = []
        current = 0
        for num in self.multiclass_num:
            start = current
            end = current + num
            self.indices.append((start, end))
            current = end
            
    def forward(self, logit, label):
        """
        Args:
            logit (torch.Tensor): (batch_size, totalclass_num)
            label (torch.Tensor): (batch_size, totalclass_num), one-hot
        """
        assert logit.shape[1] == self.totalclass_num, f"Logit dimension {logit.shape[1]} does not match totalclass_num {self.totalclass_num}"
        assert label.shape[1] == self.totalclass_num, f"Label dimension {label.shape[1]} does not match totalclass_num {self.totalclass_num}"
        
        total_loss = 0.0
        for (start, end) in self.indices:
            group_logit = logit[:, start:end]
            group_label = label[:, start:end]
            if len(self.weight_ls) == self.totalclass_num:
                group_weight = self.weight_ls[start:end]
            else:
                group_weight = [1.0 for i in range(end-start)]
            _, target = torch.max(group_label, dim=1)
            
            loss = F.cross_entropy(group_logit, target, weight = torch.tensor(group_weight).float().cuda(group_logit.device))
            total_loss += loss
        
        average_loss = total_loss / len(self.multiclass_num)
        
        return average_loss
