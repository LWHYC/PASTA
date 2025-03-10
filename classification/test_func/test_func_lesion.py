import math
import multiprocessing
import os
import time
from collections import OrderedDict

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import inspect

import numpy as np

def create_gaussian_kernel(patch_size, sigma_scale=0.1):
    # Generate grid of indices
    ax = [np.arange(size) - (size - 1) / 2.0 for size in patch_size]
    xx, yy, zz = np.meshgrid(*ax, indexing="ij")

    # Calculate squared distances
    square_dist = xx**2 + yy**2 + zz**2

    # Compute the standard deviation based on the size of the patch
    sigma = [(size * sigma_scale) for size in patch_size]
    sigma_squared = np.array([s**2 for s in sigma])

    # Because we want to broadcast the sigma values for the entire grid,
    # we need to expand the sigma_squared to match the grid dimensions.
    sigma_squared = sigma_squared[:, np.newaxis, np.newaxis, np.newaxis]

    # Calculate Gaussian weights
    kernel = np.exp(-0.5 * square_dist / sigma_squared.sum())
    kernel -= kernel.min()

    # Normalize the kernel values so that the maximum is 1
    kernel /= np.max(kernel)

    # Adjust the minimum value to 0.1
    kernel = kernel * (0.9) + 0.05

    return kernel


def extract_patch(args):
    image, mask, zs, ys, xs, patch_size, dtype = args
    test_patch = image[:,:,zs:zs+patch_size[0], ys:ys+patch_size[1], xs:xs+patch_size[2]]
    if mask:
        mask_patch = mask[:,zs:zs+patch_size[0], ys:ys+patch_size[1], xs:xs+patch_size[2]]
    else:
        mask_patch = False
    return test_patch, mask_patch, (zs, ys, xs)

def test_single_case(net, image, stride, patch_size, dtype, mask=False, 
                     gaussian_weight=True, **kwargs):
    _, w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(0,0),(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    _,ww,hh,dd = image.shape

    sz = math.ceil((ww - patch_size[0]) / stride[0]) + 1
    sy = math.ceil((hh - patch_size[1]) / stride[1]) + 1
    sx = math.ceil((dd - patch_size[2]) / stride[2]) + 1
    # print("{}, {}, {}".format(sz, sy, sx))
    predic_all = {}
    cnt = np.zeros(image.shape[1::]).astype(dtype)
    image = torch.from_numpy(np.expand_dims(image,axis=0).astype(dtype))
    if mask:
        mask=torch.from_numpy(np.expand_dims(mask,axis=0).astype(dtype))

    if gaussian_weight:
        gaussian_kernel = create_gaussian_kernel(patch_size).astype(dtype)
    else:
        gaussian_kernel = np.ones(patch_size).astype(dtype)

    args_list = []
    for z in range(0, sz):
        zs = max(0,min(stride[0]*z-1, ww-patch_size[0]-1))
        for y in range(0, sy):
            ys = max(0,min(stride[1] * y-1,hh-patch_size[1]-1))
            for x in range(0, sx):
                xs = max(0, min(stride[2] * x-1, dd-patch_size[2]-1))
                args_list.append((image, mask, zs, ys, xs, patch_size, dtype))
    # Use multiprocessing to extract patches in parallel
    with multiprocessing.Pool(4) as pool:
        results = pool.map(extract_patch, args_list)
    
    patches, mask_patches, coords = zip(*results)
    for idx, (zs, ys, xs) in enumerate(coords):
        params = kwargs
        params['x'] = patches[idx].cuda()
        if mask:
            params['mask'] = mask_patches[idx].cuda()
        if hasattr(net, 'module'):
            # 使用原始模型的签名
            sig = inspect.signature(net.module.forward)
        else:
            # 直接使用模型的签名
            sig = inspect.signature(net.forward)
        args_for_function = {k: v for k, v in params.items() if k in sig.parameters}
        predic = net(**args_for_function)
        for key in predic.keys():
            if torch.is_tensor(predic[key]) and 'cam' not in key:
                if key in predic_all.keys():
                    predic_all[key][:, zs:zs+patch_size[0], ys:ys+patch_size[1], xs:xs+patch_size[2]] += predic[key].cpu().data.numpy()[0].astype(dtype)*gaussian_kernel
                else:
                    predic_all[key] = np.zeros((predic[key].shape[1], ) + image.shape[2::], dtype=dtype)
                    predic_all[key][:, zs:zs+patch_size[0], ys:ys+patch_size[1], xs:xs+patch_size[2]] += predic[key].cpu().data.numpy()[0].astype(dtype)*gaussian_kernel
        cnt[zs:zs+patch_size[0], ys:ys+patch_size[1], xs:xs+patch_size[2]] += gaussian_kernel

    for key in list(predic_all.keys()):
        predic_all[key] /= cnt[np.newaxis,:]
        predic_all[key] = predic_all[key] [:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        if 'prob' in key:
            predic_all[key.replace('prob','label')]=np.argmax(predic_all[key], axis=0).astype(np.int16)
    predic_all['cnt'] = cnt
    return predic_all


def cal_metric(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    total_precision = np.zeros(num-1)
    total_recall = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = 1.0*(prediction==i)
        label_tmp = 1.0*(label==i)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp)+0.01)
        precision = np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp)+0.01)
        recall = np.sum(prediction_tmp * label_tmp) / (np.sum(label_tmp)+0.01)
        total_dice[i - 1] += dice
        total_precision[i - 1] += precision
        total_recall[i - 1] += recall

    return total_dice, total_precision, total_recall
