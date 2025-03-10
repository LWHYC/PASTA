#!/usr/bin/env python
import os
import numpy as np
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import SimpleITK as sitk

def convert_to_one_hot(volume, class_number):
    '''
    one hot编码
    :param volume: label
    :param C: class number
    :return:
    '''
    shape = [class_number] + list(volume.shape)
    volume_one = np.eye(class_number)[volume.reshape(-1)].T
    volume_one = volume_one.reshape(shape)
    return volume_one

def read_file_list(filename, prefix=None, suffix=None):
    '''
    Reads a list of files from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist



label_list = read_file_list('../config/verse19/test_label.txt') # label file path list
seg_list = read_file_list('../config/verse19/test_label.txt') # seg file path list
class_num = 26  # with bg
inter_sum = np.zeros(class_num)
label_sum = np.zeros(class_num)
seg_sum = np.zeros(class_num)


for i in range(len(label_list)):
    num = 0
    label_path = label_list[i]
    seg_path = seg_list[i]
    label = sitk.ReadImage(label_path)
    label = sitk.GetArrayFromImage(label)
    oh_label = convert_to_one_hot(label, class_number=class_num).reshape(class_num, -1)
    seg = sitk.ReadImage(seg_path)
    seg = sitk.GetArrayFromImage(seg)
    oh_seg = convert_to_one_hot(seg, class_number=class_num).reshape(class_num, -1)
    inter_sum += np.sum(oh_label*oh_seg, axis=1)
    label_sum += np.sum(oh_label, axis=1)
    seg_sum += np.sum(oh_seg, axis=1)
    print(label_path, seg_path)

dice = 2*(inter_sum+0.1)/(label_sum+seg_sum+0.2)
recall = (inter_sum+0.1)/(seg_sum+0.1)
precision = (inter_sum+0.1)/(label_sum+0.1)

print(dice, recall, precision)
print(np.mean(dice[1::]), np.mean(recall[1::]), np.mean(precision[1::]))

