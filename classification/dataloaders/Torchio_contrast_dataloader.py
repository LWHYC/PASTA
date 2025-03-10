import concurrent.futures
import copy
import json
import os
import random
import sys
from typing import Any
sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.ndimage import label, find_objects
import json

from .data_process_func import *
import time

terminal=sys.stdout

def create_sample(file_path_dic=False, respacing=False, target_spacing=1):
    sample = {}
    reference_shape = None
    
    for key in file_path_dic.keys():
        if file_path_dic[key]['dtype'] not in ['json', 'txt', 'str', 'list', 'str_float', 'str_int']:
            if file_path_dic[key]['dtype'] == 'float':
                dtype = torch.float16
                mode = 'image'
            elif file_path_dic[key]['dtype'] == 'int':
                dtype = torch.int16
                mode = 'label'
                
            file = torch.from_numpy(
                load_nifty_volume_as_array(
                    file_path_dic[key]['path'], 
                    reorient=True, 
                    respacing=respacing, 
                    target_spacing=target_spacing, 
                    mode = mode
                )['data'].copy()
            ).to(dtype)
            
            if len(file.shape) == 3:
                file = file.unsqueeze(0) 
            elif len(file.shape) != 4:
                raise ValueError('Please input the right file dimension! Expected 3 or 4 dimensions.')
            
            # Check the shape of the file
            if reference_shape is None:
                reference_shape = file.shape[-3:]  
            elif file.shape[-3:] != reference_shape:
                all_shapes = {k: v.shape[-3:] for k, v in sample.items() if isinstance(v, torch.Tensor)}
                raise ValueError(f"Shape mismatch for key '{key}' with shape {file.shape[-3:]}. "
                                 f"Pach {file_path_dic[key]['path']}. "
                                 f"Expected shape: {reference_shape}. All shapes: {all_shapes}")
            
            sample[key] = file
            sample[f'{key}_path'] = file_path_dic[key]['path']
            
        elif file_path_dic[key]['dtype'] in ['json']:
            # Parse the json file
            with open(file_path_dic[key]['path'], 'r') as file:
                json_file = json.load(file)
            sample[key] = json_file
            sample[f'{key}_path'] = file_path_dic[key]['path']
        
        elif file_path_dic[key]['dtype'] in ['txt']:
            # Load the txt file as a list
            converted_list = read_file_list(file_path_dic[key]['path'])

            sample[key] = converted_list
            sample[f'{key}_path'] = file_path_dic[key]['path']
            
        elif file_path_dic[key]['dtype'] in ['str']:
            sample[key] = file_path_dic[key]['path']
            sample[f'{key}_path'] = file_path_dic[key]['path']
        
        elif file_path_dic[key]['dtype'] in ['list']:
            converted_list = file_path_dic[key]['path'].split(',')

            converted_list = [float(item.strip()) for item in converted_list]
            sample[key] = converted_list
            sample[f'{key}_path'] = file_path_dic[key]['path']
            
        elif file_path_dic[key]['dtype'] in ['str_float']:
            sample[key] = float(file_path_dic[key]['path'])
            sample[f'{key}_path'] = file_path_dic[key]['path']
        
        elif file_path_dic[key]['dtype'] in ['str_int']:
            sample[key] = int(file_path_dic[key]['path'])
            sample[f'{key}_path'] = file_path_dic[key]['path']
    
    return sample


def process_image(index, file_path_dic_ls, respacing=False, target_spacing=1):
    file_path_dic = file_path_dic_ls[index]
    return index, create_sample(file_path_dic, respacing=respacing, target_spacing=target_spacing)


class TorchioDataloader(Dataset):
    def __init__(self, iter_num=0, batch_size=0, num=None, transform=None, 
                 random_sample=True, load_memory=True, memory_length=0, file_ls_dic={}, load_numworker=32,
                 respacing=False, target_spacing=1):
        """
        Args:
            iter_num (int): Number of iterations for the dataloader.
            batch_size (int): Size of each batch.
            num (int, optional): Number of samples to load. If None, load all samples.
            transform (callable, optional): Optional transform to be applied on a sample.
            random_sample (bool): Whether to sample randomly.
            load_memory (bool): Whether to load data into memory.
            memory_length (int): Number of samples to keep in memory.
            load_numworker (int): Number of workers to use for loading data.
            normalize (bool): Whether to normalize the images.
            thresh_ls (list): List of threshold values for normalization. Could be empty list.
            norm_ls (list): List of normalization values. Could be empty list.
            respacing (bool): Whether to respace the images.
            target_spacing (int): Target spacing value for resampling.
        """
        # Initialize the data loader with various parameters and lists
        self.image_task_dic = {}  # Dictionary to store processed image samples
        self.cur_file_index_ls = []  # List to store currently loaded images
        self.left_file_index_ls = []  # List to store remaining images to be loaded

        # Assigning input parameters to class variables
        self._iternum = iter_num
        self.batch_size = batch_size
        self.transform = transform
        self.random_sample = random_sample
        self.load_memory = load_memory
        self.memory_length = memory_length
        self.load_numworker = load_numworker
        self.respacing = respacing
        self.target_spacing = target_spacing
        
        self.file_path_dic_ls = []
        for key in file_ls_dic.keys():
            file_ls = read_file_list(file_ls_dic[key]['file_list'])
            if len(self.file_path_dic_ls) == 0:
                for i in range(len(file_ls)):
                    self.file_path_dic_ls.append({key:{'path': file_ls[i], 'dtype': file_ls_dic[key]['dtype']}})
            else:
                for i in range(len(file_ls)):
                    self.file_path_dic_ls[i][key] = {'path': file_ls[i], 'dtype': file_ls_dic[key]['dtype']}
        
        # Loading images into memory if load_memory is set to True
        if self.load_memory:
            if self.memory_length:
                self.cur_file_index_ls = random.sample(list(range(len(self.file_path_dic_ls))), self.memory_length)
            else:
                self.cur_file_index_ls = list(range(len(self.file_path_dic_ls)))

            self.left_file_index_ls = list(set(range(len(self.file_path_dic_ls))) - set(self.cur_file_index_ls))
            
            # Using ThreadPoolExecutor to process images concurrently
            with concurrent.futures.ThreadPoolExecutor(self.load_numworker) as executor:
                futures = {executor.submit(process_image, cur_file_index, self.file_path_dic_ls,
                            self.respacing, self.target_spacing): cur_file_index for cur_file_index in self.cur_file_index_ls}
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
                    i = futures[future]
                    try:
                        cur_file_index, image_sample = future.result()
                        self.image_task_dic[cur_file_index] = image_sample
                    except Exception as exc:
                        print(f'Generated an exception: {exc}')
            
            
            random.shuffle(self.left_file_index_ls)
        
        if num is not None:
            self.file_path_dic_ls = self.file_path_dic_ls[:num]
                
        print("total {} samples".format(len(self.file_path_dic_ls)))

    def __len__(self):
        # Returns the length of the dataset
        if self.random_sample:
            return self._iternum * self.batch_size
        else:
            if self.load_memory:
                return len(self.cur_file_index_ls)
            else:
                return len(self.file_path_dic_ls)

    def updata_memory(self):
        # Updates the in-memory samples by replacing half of them with new ones
        new_samples = {}
        new_index_list = []
        
        for _ in range(len(self.cur_file_index_ls) // 2):
            out_file_index = self.cur_file_index_ls.pop(0)
            del self.image_task_dic[out_file_index]
            self.left_file_index_ls.append(out_file_index)

            in_file_index = self.left_file_index_ls.pop(0)
            new_index_list.append(in_file_index)
            self.cur_file_index_ls.append(in_file_index)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for in_file_index in new_index_list:
                file_path_dic = self.file_path_dic_ls[in_file_index]

                futures.append(executor.submit(create_sample, file_path_dic, 
                            self.respacing, self.target_spacing))

            pbar = tqdm(total=len(futures), desc="Updating memory", dynamic_ncols=True)

            for future, in_file_index in zip(futures, new_index_list):
                try:
                    nsample = future.result()
                    new_samples[in_file_index] = nsample
                    pbar.update(1)
                except Exception as e:
                    print(f"An error occurred with image '{in_file_index}': {e}")
                    if in_file_index in self.cur_file_index_ls:
                        self.cur_file_index_ls.remove(in_file_index)

            pbar.close()

        self.image_task_dic.update(new_samples)

    def __getitem__(self, idx):
        keep_sample = True
        # Returns a sample from the dataset
        if self.load_memory:
            while keep_sample:
                try:
                    if self.random_sample:
                        idx = random.randint(0, len(self.cur_file_index_ls) - 1)
                    file_index = self.cur_file_index_ls[idx]
                    sample = copy.deepcopy(self.image_task_dic[file_index])
                    keep_sample = False
                except Exception as e:
                    print(e)
        else:
            while keep_sample:
                if self.random_sample:
                    idx = random.choice(range(len(self.file_path_dic_ls)))
                
                file_path_dic = self.file_path_dic_ls[idx]
                try:
                    sample = create_sample(file_path_dic, self.respacing, self.target_spacing)
                    keep_sample = False
                except Exception as e:
                    print(e)
                    print(file_path_dic)

        if self.transform:
            sample = self.transform(sample)
        return sample

class VolumeStatics(object):  
    def __init__(self, mode='label', image='image', num_workers=16):  
        self.mode = mode  
        self.image = image  
        self.num_workers = num_workers
        
    def compute_stats(self, class_id, images, labels):
        mask = (labels == class_id)  
        mean = np.mean(images[mask])  
        # std = np.std(images[mask])  
        return class_id, mean.item()
    
    def __call__(self, sample):
        # time0 = time.time() 
        # 提取标签和图像数据  
        labels = sample[self.mode].numpy()  
        images = sample[self.image].numpy()  
        # 将标签转换为uint8，并排除背景（假设背景为0）  
        unique_class_ids = np.unique(labels.astype(np.uint8))  
        unique_class_ids = unique_class_ids[unique_class_ids != 0]  
          
        # 使用Pool并行计算每个类别的统计信息
        volume_statics_dic = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for class_id in unique_class_ids:
                # Submit the create_sample task to the executor
                futures.append(executor.submit(self.compute_stats, class_id, images, labels))

            # Wait for all futures to complete
            for future in futures:
                class_id, mean = future.result()
                volume_statics_dic[str(class_id.item())] = [mean] 
        # 将统计信息添加到样本中  
        sample['volume_statics_dic'] = volume_statics_dic  
        # 返回处理后的样本 
        # time1 = time.time()
        # print('static time cost:', time1-time0)
        return sample  


class CropBound(object):
    def __init__(self, pad=[0,0,0], mode='label', class_determine=False):
        self.pad = pad
        self.mode = mode
        self.class_determine=class_determine
    def __call__(self, sample):
        if self.class_determine:
            file = torch.isin(sample[self.mode], torch.tensor(self.class_determine))
        else:
            file = sample[self.mode]
        file = torch.sum(file, dim=0) 
        file_size = file.shape # DWH
        nonzeropoint = torch.as_tensor(torch.nonzero(file))
        maxpoint = torch.max(nonzeropoint, 0)[0].tolist()
        minpoint = torch.min(nonzeropoint, 0)[0].tolist()
        for i in range(len(self.pad)):
            maxpoint[i] = min(maxpoint[i] + self.pad[i], file_size[i])
            minpoint[i] = max(minpoint[i] - self.pad[i], 0)
            if 'bbox' in sample.keys():
                sample['bbox'] -= np.array(minpoint)[np.newaxis,:]
                sample['bbox'][sample['bbox'] < 0] = 0

        sample['minpoint']=minpoint
        sample['maxpoint']=maxpoint
        sample['shape'] = file_size

        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                sample[key]=sample[key][:, minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
        return sample


class PadToScale(object):
    def __init__(self, scale, mode='label'):
        self.scale = scale
        self.mode = mode

    def _compute_pad(self, size):
        # 计算每个维度需要的padding数量，使其可以被scale整除
        return [int((self.scale - s % self.scale) % self.scale) for s in size]

    def __call__(self, sample):
        file = sample[self.mode][1::]
        file_size = file.shape[-3:]  # 假设tensor的shape为[C, D, H, W]
        
        pad_values = self._compute_pad(file_size)
        # 只在每个维度的一侧进行padding
        pad = (0, pad_values[2], 0, pad_values[1], 0, pad_values[0])
        sample['pad'] = pad
        # 更新sample中的所有tensor
        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                sample[key] = F.pad(sample[key], pad=pad, mode='constant', value=0)

        return sample


class ExtractCertainClass(object):
    def __init__(self, class_wanted=[1]):
        self.class_wanted = class_wanted
    def __call__(self, sample):
        label = sample['label']
        nlabel = label.index_select(0, torch.tensor([0]+self.class_wanted))
        sample ['label'] = nlabel
        if 'coarseg' in sample:
            ncoarseg = sample['coarseg'].index_select(0, torch.tensor([0]+self.class_wanted))
            sample ['coarseg'] = ncoarseg
                
        return sample


class RandomNoiseAroundClass(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma
    def __call__(self, sample):
        image = sample['image']
        noise = torch.clamp(self.sigma * torch.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        sample['image'] = image
        return sample

class SynLabelProcess(object):
    def __init__(self, syn_path_root_ls = [], lesion_key=''):
        self.syn_path_root_ls = syn_path_root_ls
        self.lesion_key = lesion_key
    def __call__(self, sample):
        image_path = sample['image_path']
        is_syn = False
        for syn_path_root in self.syn_path_root_ls:
            if syn_path_root in image_path:
                is_syn = True
        if is_syn:
            label = sample[self.lesion_key]
            label[label==1] = 0
            label[label>1] = 1
            sample[self.lesion_key] = label
        return sample
    
class BinaryLabelProcess(object):
    def __init__(self, lesion_key=''):
        self.lesion_key = lesion_key
    def __call__(self, sample):
        label = sample[self.lesion_key]
        label = np.clip(label, a_min=0, a_max=2)
        sample[self.lesion_key] = label
        return sample


class CropAroundPoint(object):
    def __init__(self, output_size, center_point_key='', target_mode_key='label', target_type=torch.float32):
        '''
        Args:
            output_size (int, tuple, list): 裁剪输出的大小。可以是整数（各维度相同），也可以是包含每个维度大小的元组或列表。
            center_point_key (str): 指定中心点的键，通常是样本字典中的一个键。
            target_mode_key (str): 指定目标模式的键，通常是样本字典中的一个键。
            target_type (torch.dtype): 目标数据类型。用于转换裁剪后的数据类型。
        '''
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            self.output_size = output_size
        self.center_point_key = center_point_key
        self.target_mode_key = target_mode_key
        self.target_type = target_type

    def __call__(self, sample):
        '''
        Args:
            sample (dict): 包含数据的样本字典，例如 {'image': tensor, 'label': tensor}。
        '''
        cshape = sample[self.target_mode_key].shape[1:]  # 获取目标模式（如标签）的形状，不包括channel维度（DWH：深度、高度、宽度）

        # 获取裁剪尺寸
        crop_d, crop_h, crop_w = self.output_size
        max_d, max_h, max_w = cshape
        center_point = sample[self.center_point_key]

        if center_point is not None:
            # 指定中心点裁剪
            cz, cx, cy = center_point

            # 确保中心点的裁剪范围不超出边界
            w1 = max(0, min(cx - crop_w // 2, max_w - crop_w))
            h1 = max(0, min(cy - crop_h // 2, max_h - crop_h))
            d1 = max(0, min(cz - crop_d // 2, max_d - crop_d))
        else:
            # 随机裁剪
            w1 = torch.randint(0, max_w - crop_w + 1, (1,)).item()
            h1 = torch.randint(0, max_h - crop_h + 1, (1,)).item()
            d1 = torch.randint(0, max_d - crop_d + 1, (1,)).item()

        # 对样本的每个张量进行裁剪并转换为目标数据类型
        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                cropped = sample[key][:, w1:w1 + crop_w, h1:h1 + crop_h, d1:d1 + crop_d].to(self.target_type)

                # 获取裁剪后张量的实际大小
                actual_d, actual_h, actual_w = cropped.shape[1:]

                # 如果裁剪后的大小小于目标大小，则进行填充
                pad_d = max(0, crop_d - actual_d)
                pad_h = max(0, crop_h - actual_h)
                pad_w = max(0, crop_w - actual_w)

                # 填充顺序为 (前面填充, 后面填充)
                padded = F.pad(cropped, 
                               (0, pad_w, 0, pad_h, 0, pad_d),  # 宽度、高度、深度的填充
                               mode='constant', value=0)

                # 更新样本
                sample[key] = padded

        return sample

class Pad2Shape(object):
    def __init__(self, size=[128,128,128]):
        self.size = size
    def __call__(self, sample):
        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                shape = sample[key].shape
                pad = [0,0,0,0,0,0]
                for i in range(len(self.size)):
                    pad[2*i] = (self.size[i]-shape[i+1])//2
                    pad[2*i+1] = self.size[i]-shape[i+1]-pad[2*i]
                pad = tuple(np.clip(pad, 0, 100000))
                sample[key]=F.pad(sample[key], pad, mode='constant', value=0)
        return sample

class Resize(object):
    def __init__(self, size=[128,128,128]):
        self.size = size
    def __call__(self, sample):
        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                old_type = sample[key].dtype
                sample[key]=F.interpolate(sample[key].float().unsqueeze(dim=0), size=self.size, mode='trilinear', align_corners=False)[0]
                sample[key] = sample[key].type(old_type)
        return sample


class RandomNoise(object):
    def __init__(self, mean=0, std=0.1,include=['image'], prob=0):
        self.prob = prob
        self.add_noise = tio.RandomNoise(mean=mean, std=std, include=include)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample= self.add_noise(sample)
        return sample


class RandomFlip(object):
    def __init__(self, include=['image'], axes=[], prob=0):
        self.flip_probability = prob
        self.include = include
        self.axes = axes
    def __call__(self, sample):
        axes = random.choice(self.axes)
        flip = tio.RandomFlip(axes=axes, flip_probability=self.flip_probability, include = self.include)
        sample= flip(sample)
        return sample

class RandomTranspose(object):
    def __init__(self, include=['image'], prob=0):
        self.prob = prob
        self.include = include
    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            dim = [1,2,3]
            random.shuffle(dim)
            for key in self.include:
                sample[key] = sample[key].permute(0, dim[0], dim[1], dim[2])
        return sample

class RandomAffine(object):
    def __init__(self, scales=[0.2,0.2,0.2], degrees=[10,10,10],
                include=['image','label'], prob=0):
        self.prob = prob
        self.add_elas = tio.RandomAffine(
            scales=scales,
            degrees=degrees,
            include=include)
        self.include = include

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            for key in self.include:
                sample[key] = sample[key].float()
            sample=self.add_elas(sample)
        return sample
    
class RandomAnisotropy(object):
    def __init__(self, axes=[0], downsampling=(1, 5),
                include=['image'], prob=0):
        self.prob = prob
        self.add_anis = tio.RandomAnisotropy(
            axes=axes,
            downsampling=downsampling,
            include=include)
        self.include = include

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            for key in self.include:
                sample[key] = sample[key].float()
            sample=self.add_anis(sample)
        return sample

class RandomSpike(object):
    def __init__(self, num_spikes=3, intensity=1.2,include=['image'], prob=0):
        self.prob = prob
        self.add_spike = tio.RandomSpike(num_spikes=num_spikes, intensity=intensity,include=include)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_spike(sample)
        return sample



class RandomGhosting(object):
    def __init__(self, intensity=0.8, include=['image'], prob=0):
        self.prob = prob
        self.add_ghost = tio.RandomGhosting(intensity=intensity, include=include)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_ghost(sample)
        return sample

class RandomElasticDeformation(object):
    def __init__(self, num_control_points=[5,10,10], max_displacement=[7,7,7], include=['image','label'], prob=0):
        self.prob = prob
        self.add_elas = tio.RandomElasticDeformation(
            num_control_points=num_control_points,
            max_displacement = max_displacement,
            locked_borders=2,
            image_interpolation='linear',
            label_interpolation='nearest',
            include=include)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_elas(sample)
        return sample


class MultiThreshNormalized(object):
    def __init__(self, include=['image'], thresh_ls=[[-1000, -200, 200, 1000]], norm_ls=[[0,0.2,0.8,1]]):
        self.thresh_ls = thresh_ls
        self.norm_ls = norm_ls
        self.include = include

    def __call__(self, sample):
        for key in self.include:
            image = sample[key]
            for i in range(len(self.thresh_ls)):
                if i ==0:
                    tensor=img_multi_thresh_normalized_torch(image, self.thresh_ls[i], self.norm_ls[i], data_type=torch.float32)
                else:
                    tensor=torch.cat((tensor, img_multi_thresh_normalized_torch(image, self.thresh_ls[i], self.norm_ls[i], data_type=torch.float32)),0)
            image = tensor
            # if nan in image, warning
            if torch.isinf(image).any():
                print('inf in normalize')
            sample[key]=image
        print('multithresh', sample['image_path'])
        return sample

class Reshape(object):
    def __init__(self, include=['image'], target_shape=[128, 128, 128], keep_dim_shape=[]):
        self.include = include
        self.target_shape = target_shape
        self.keep_dim_shape = keep_dim_shape

    def __call__(self, sample):
        for key in self.include:
            image = sample[key]
            target_shape = self.target_shape.copy()
            
            if len(self.keep_dim_shape) > 0:
                for dim in self.keep_dim_shape:
                    target_shape[dim] = image.shape[dim+1]
            pad = (0, 0, 0)
            d0, h0, w0 = image.shape[1:]
            d, h, w = target_shape
            sample['shape'] = (image.shape[1:], ((d/d0, h/h0, w/w0), pad))
            if image.dtype in [torch.int16, torch.int32, torch.int64, torch.int8, torch.uint8]:
                interpolation_mode = 'nearest'
                image = torch.nn.functional.interpolate(image.unsqueeze(0), size=target_shape, \
                                                        mode=interpolation_mode).squeeze(0)
            else:
                interpolation_mode = 'trilinear'
                dtype = image.dtype
                image = torch.nn.functional.interpolate(image.float().unsqueeze(0), size=target_shape, \
                                                        mode=interpolation_mode, align_corners=False).squeeze(0)
                image = image.to(dtype)
            sample[key] = image
            
        return sample


class MultiNormalized(object):
    def __init__(self, include=['image'], thresh_ls=[[-1000, 1000]]):
        self.thresh_ls = thresh_ls
        self.include = include

    def __call__(self, sample):
        for key in self.include:
            image = sample[key]
            for i in range(len(self.thresh_ls)):
                if i ==0:
                    tensor=img_normalized_torch(image.float(), downthresh=self.thresh_ls[i][0], upthresh=self.thresh_ls[i][-1], norm=True, thresh=True).half()
                else:
                    tensor=torch.cat((tensor, img_normalized_torch(image.float(), downthresh=self.thresh_ls[i][0], upthresh=self.thresh_ls[i][-1], norm=True, thresh=True)),0).half()
            image = tensor
            sample[key]=image
        return sample
    
class ExtractSynlesionBbox(object):
    def __init__(self, lesion_key, organ_key, attribute_key, lesion_value, random_bound=2):
        """
        初始化时指定从sample字典中提取的key。
        """
        self.lesion_key = lesion_key
        self.organ_key = organ_key
        self.attribute_key = attribute_key
        self.random_bound = random_bound
        self.lesion_value = lesion_value

    def __call__(self, sample):
        """
        从sample中提取指定key的3维矩阵，计算所有类别的连通区域的bbox。
        
        Args:
            sample (dict): 输入的包含多类别3维矩阵的字典。
        
        Returns:
            torch.Tensor: 包含所有类别连通区域bbox的张量，
                          每个bbox用 (类别, min_x, max_x, min_y, max_y, min_z, max_z) 表示。
        """
        # 获取3维矩阵
        lesion_organ_ls = list(sample.get(self.attribute_key).keys())
        lesion_matrix = 1*(sample.get(self.lesion_key).numpy()[0] == self.lesion_value)  # 将 tensor 转换为 numpy
        lesion_matrix = lesion_matrix * sample.get(self.organ_key).numpy()[0]  # 将 tensor 转换为 numpy
        d, w, h = lesion_matrix.shape
        if lesion_matrix is None:
            raise ValueError(f"Key '{self.lesion_key}' not found in sample.")

        # 确保输入是一个3维矩阵
        if lesion_matrix.ndim != 3:
            raise ValueError("Input lesion_matrix must be 3D.")

        # 获取矩阵中的所有唯一类别（排除背景值0）
        unique_labels = np.unique(lesion_matrix)
        nunique_labels = [int(i) for i in unique_labels if str(int(i)) in lesion_organ_ls]
        if len(nunique_labels) > 0:
            unique_label = random.choices(nunique_labels, k=1)  # 每次只保留一个器官
        else:
            unique_label = []
        bboxes = []

        # 对每个类别单独处理
        for label_value in unique_label:
            # 创建类别的二值掩码
            binary_mask = (lesion_matrix == label_value)
            
            # 标记该类别的所有连通区域
            labeled_array, _ = label(binary_mask)

            # 获取每个连通区域的切片对象
            slices = find_objects(labeled_array)

            # 只保留一个非空的slice
            slices = [s for s in slices if s is not None]
            if slices:
                slice_ = random.choice(slices)  # 随机选择一个slice
                
                min_x, max_x = max(0, slice_[0].start - random.randint(0, self.random_bound)), min(d - 1, slice_[0].stop - 1 + random.randint(0, self.random_bound))
                min_y, max_y = max(0, slice_[1].start - random.randint(0, self.random_bound)), min(w - 1, slice_[1].stop - 1 + random.randint(0, self.random_bound))
                min_z, max_z = max(0, slice_[2].start - random.randint(0, self.random_bound)), min(h - 1, slice_[2].stop - 1 + random.randint(0, self.random_bound))
                min_x, max_x = max(0, min(min_x, max_x - 1)), min(d - 1, max(min_x + 1, max_x))
                min_y, max_y = max(0, min(min_y, max_y - 1)), min(w - 1, max(min_y + 1, max_y))
                min_z, max_z = max(0, min(min_z, max_z - 1)), min(h - 1, max(min_z + 1, max_z))
                
                # 将类别和对应的bbox一起存储
                bboxes.append([int(label_value), min_x, max_x, min_y, max_y, min_z, max_z])
        
        sample['bbox'] = bboxes
        return sample


class ExtractReallesionBbox:
    def __init__(self, largest_lesion_key, attribute_key, random_bound=2):
        """
        初始化时指定从sample字典中提取的key。
        """
        self.largest_lesion_key = largest_lesion_key
        self.attribute_key = attribute_key
        self.random_bound = random_bound

    def __call__(self, sample):
        """
        从sample中提取指定key的3维矩阵，计算所有类别的连通区域的bbox。
        
        Args:
            sample (dict): 输入的包含多类别3维矩阵的字典。
        
        Returns:
            torch.Tensor: 包含所有类别连通区域bbox的张量，
                          每个bbox用 (类别, min_x, max_x, min_y, max_y, min_z, max_z) 表示。
        """
        # 获取3维矩阵
        lesion_organ_ls = list(sample.get(self.attribute_key).keys())
        largest_lesion_matrix = sample.get(self.largest_lesion_key).numpy()[0]  # 将 tensor 转换为 numpy
        d, w, h = largest_lesion_matrix.shape
        if largest_lesion_matrix is None:
            raise ValueError(f"Key '{self.key}' not found in sample.")

        # 确保输入是一个3维矩阵
        if largest_lesion_matrix.ndim != 3:
            raise ValueError("Input largest_lesion_matrix must be 3D.")

        bboxes = []
        
        # 标记该类别的所有连通区域
        labeled_array, _ = label(largest_lesion_matrix)

        # 获取每个连通区域的切片对象
        slices = find_objects(labeled_array)

        for slice_ in slices:
            if slice_ is not None:
                min_x, max_x = max(0, slice_[0].start-random.randint(0, self.random_bound)), min(d-1,slice_[0].stop-1+random.randint(0, self.random_bound))
                min_y, max_y = max(0, slice_[1].start-random.randint(0, self.random_bound)), min(w-1,slice_[1].stop-1+random.randint(0, self.random_bound))
                min_z, max_z = max(0, slice_[2].start-random.randint(0, self.random_bound)), min(h-1,slice_[2].stop-1+random.randint(0, self.random_bound))
                min_x, max_x = max(0, min(min_x, max_x-1)), min(d-1,max(min_x+1, max_x))
                min_y, max_y = max(0, min(min_y, max_y-1)), min(w-1,max(min_y+1, max_y))
                min_z, max_z = max(0, min(min_z, max_z-1)), min(h-1,max(min_z+1, max_z))
                # 将类别和对应的bbox一起存储
                bboxes.append([int(lesion_organ_ls[0]), min_x, max_x, min_y, max_y, min_z, max_z])
        sample['bbox'] = bboxes
        return sample

class GetNegAttribute(object):
    def __init__(self, attribute_path='', attribute_key='', bbox_key=''):
        with open(attribute_path, 'r') as f:
            self.attribute_text = json.load(f)
        self.attribute_key = attribute_key
        self.bbox_key = bbox_key
    def __call__(self, sample):
        pos_dic = sample[self.attribute_key].copy()
        bbox_ls = sample[self.bbox_key].copy()
        bbox_organ_ls = [bbox[0] for bbox in bbox_ls]
        neg_dic = {}
        aug_pos_dic = {}
        for organ_key in pos_dic.keys():
            if int(organ_key) in bbox_organ_ls: # 随机裁剪的时候，有的器官上的病灶可能没被裁剪进来，在attribute里需要把该器官去掉
                pos_organ_dic = pos_dic[organ_key]
                neg_dic[organ_key] = {}
                aug_pos_dic[organ_key] = {}
                for attribute_key in pos_organ_dic.keys():
                    pos_attribute = pos_organ_dic[attribute_key].lower()
                    aug_pos_attribute = copy.deepcopy(pos_attribute)
                    attribute_lss = self.attribute_text[attribute_key].copy()
                    neg_attribute_ls = []
                    for attribute_ls in attribute_lss:
                        attribute_ls = [attribute.lower() for attribute in attribute_ls]
                        is_neg = True
                        for attribute in attribute_ls:
                            if attribute in pos_attribute:
                                is_neg = False
                                aug_pos_attribute = aug_pos_attribute.replace(attribute, random.choice(attribute_ls))
                        if is_neg:
                            neg_attribute_ls += attribute_ls
                    neg_dic[organ_key][attribute_key] = neg_attribute_ls
                    aug_pos_dic[organ_key][attribute_key] = aug_pos_attribute
        sample['neg_json'] = neg_dic
        sample['aug_pos_json'] = aug_pos_dic
        return sample

class MultiThreshNormalized(object):
    def __init__(self, include=['image'], thresh_ls=[[-1000, -200, 200, 1000]], norm_ls=[[0,0.2,0.8,1]]):
        self.thresh_ls = thresh_ls
        self.norm_ls = norm_ls
        self.include = include

    def __call__(self, sample):
        for key in self.include:
            image = sample[key]
            for i in range(len(self.thresh_ls)):
                if i ==0:
                    tensor=img_multi_thresh_normalized_torch(image, self.thresh_ls[i], self.norm_ls[i], data_type=torch.float32)
                else:
                    tensor=torch.cat((tensor, img_multi_thresh_normalized_torch(image, self.thresh_ls[i], self.norm_ls[i], data_type=torch.float32)),0)
            image = tensor
            # if nan in image, warning
            if torch.isinf(image).any():
                print('inf in normalize')
            sample[key]=image
        return sample


class Attribute2Onehot(object):
    """
    Convert option like ["A", "B", "B", "B", "B"] to one-hot,
    then flatten into a single vector. For example, if class_num = [4,3,2,2,2],
    the resulting vector length is sum(class_num)=13.
    """
    def __init__(self, keys, class_num):
        self.class_num = class_num
        self.keys = keys
        self.attribute_dic = {'Shape':['Round-like', 'Irregular', 'Wall thickening', 'Punctate, nodular'],
                              'Density':["Hypodense", "Isodense", "Hyperdense"],
                              'Heterogeneity': ["Homogeneous", "Heterogeneous"],
                              'Surface':["Well-defined margin", "Ill-defined margin"],
                              'Invasion':["No close relationship with adjacent structures", "Close relationship with adjacent structures"],
                              }

    def __call__(self, sample):
        for key in self.keys:
            cur_attribute_dic = sample[key]  # For example, ["A", "B", "B", "B", "B"]
            cur_attribute = cur_attribute_dic[list(cur_attribute_dic.keys())[0]]
            cur_onehot = []
            
            for attribute_key in self.attribute_dic.keys():
                try:
                    key_attribute_ls = self.attribute_dic[attribute_key]
                    attribute = cur_attribute[attribute_key]
                    idx = key_attribute_ls.index(attribute)  # "A" -> 0, "B"->1, "C"->2 ...
                    
                    # 
                    n_class = len(key_attribute_ls)
                    one_hot = np.zeros(n_class, dtype=np.float32)
                    
                    
                    one_hot[idx] = 1.0
                    
                    cur_onehot.extend(one_hot.tolist())

                except Exception as e:
                    print(e)
                    print(sample[key+'_path'], attribute_key)
                
            # shape: (sum(class_num),)
            sample[key] = np.array(cur_onehot, dtype=np.float32)

        
        return sample

class Option2Onehot(object):
    """
    Convert option like ["A", "B", "B", "B", "B"] to one-hot,
    then flatten into a single vector. For example, if class_num = [4,3,2,2,2],
    the resulting vector length is sum(class_num)=13.
    
    Args:
        keys (list): List of keys in the sample dictionary that contain the options.
        class_num (list): List of integers, each representing the number of options for a question.
    """
    def __init__(self, keys, class_num):
        self.class_num = class_num
        self.keys = keys
        self.letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __call__(self, sample):
        for key in self.keys:
            cur_option = sample[key]  # e.g., ["A", "B", "B", "B", "B"]
            cur_option = cur_option[0].split(',')
            cur_onehot = []
            for i, ans in enumerate(cur_option):
                idx = self.letters.index(ans)  # "A" -> 0, "B"->1, "C"->2 ...
                
                n_class = self.class_num[i]    # the number of options for this question
                one_hot = np.zeros(n_class, dtype=np.float32)
                
                # if the answer index exceeds the max number of options
                if idx >= n_class:
                    raise ValueError(f"Answer '{ans}' index={idx} "
                                     f"exceeds max={n_class-1} for question {i+1}.")
                
                one_hot[idx] = 1.0
                cur_onehot.extend(one_hot.tolist())
            
            sample[key] = np.array(cur_onehot, dtype=np.float32)
        
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        pass
    def __call__(self, sample):
        for key in sample.keys():
            if isinstance(sample[key], np.ndarray):
                sample[key] = torch.from_numpy(sample[key])
                
            if torch.is_tensor(sample[key]):
                if sample[key].dtype == torch.float16:
                    sample[key] = sample[key].to(torch.float32)
                elif sample[key].dtype == torch.int16:
                    sample[key] = sample[key].to(torch.long)
                
        return sample

def custom_collate_fn(batch):
    processed_batch = {}
    batch_dict = {key: [] for key in batch[0].keys()}
    
    for sample in batch:
        for key, value in sample.items():
            batch_dict[key].append(value)
    
    for key, value_list in batch_dict.items():
        if 'bbox' in key:
            bboxes_list = []
            for bboxes_sample in value_list:
                bboxes_list.append(bboxes_sample)
            processed_batch[key] = bboxes_list
        else:
            try:
                processed_batch[key] = torch.stack(value_list, dim=0)
            except TypeError:
                processed_batch[key] = value_list
    
    return processed_batch



