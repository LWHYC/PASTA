import os
import random
import sys
from typing import Any
sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataloaders.Torchio_contrast_dataloader import *
from torchvision import transforms


def get_classify_binary_loader(args):
    trainData = TorchioDataloader(
                                iter_num= args.total_step,
                                batch_size = args.batch_size,
                                file_ls_dic ={
                                    'image':{'file_list':args.train_image_list, 'dtype':'float'},
                                    'label':{'file_list':args.train_label_list, 'dtype':'str_int'},
                                    },
                                transform=transforms.Compose([
                                    MultiThreshNormalized(include=['image'],thresh_ls=[[-1000, -200, 200, 1000]], norm_ls=[[0,0.2,0.9,1]]),
                                    Pad2Shape(size=args.crop_shape),
                                    Resize(size=args.crop_shape),
                                    RandomElasticDeformation(num_control_points=[5,5,5], max_displacement=[5,5,5], include=['image'], prob=0.5),
                                    RandomAffine(scales=[0.2,0.2,0.2], degrees=[10,10,10], include=['image'], prob=0.5),
                                    RandomTranspose(include=['image'], prob=0.5),
                                    RandomFlip(include=['image'], axes=[0,1,2], prob=0.5),
                                    RandomAnisotropy(downsampling=(1, 2), include=['image'], prob=0.5),
                                    RandomNoise(mean=0, std=0.1, include=['image'], prob=0.0),
                                    
                                    # Option2Onehot(keys=['label'], class_num=args.class_num),
                                    ToTensor(),
                                ]),
                                load_memory = args.load_memory,
                                memory_length = args.memory_length,
                                load_numworker = args.num_workers
                                )
    validData = TorchioDataloader(
                                iter_num = len(args.valid_image_list),
                                batch_size= args.batch_size,
                                file_ls_dic ={
                                    'image':{'file_list':args.valid_image_list, 'dtype':'float'},
                                    'label':{'file_list':args.valid_label_list, 'dtype':'str_int'},
                                    },
                                transform=transforms.Compose([
                                    MultiThreshNormalized(include=['image'],thresh_ls=[[-1000, -200, 200, 1000]], norm_ls=[[0,0.2,0.9,1]]),
                                    Pad2Shape(size=args.crop_shape),
                                    Resize(size=args.crop_shape),
                                    # Option2Onehot(keys=['label'], class_num=args.class_num),
                                    ToTensor(),
                                ]),
                                load_memory=True,
                                load_numworker= 6,
                                respacing=True, target_spacing=1,
                                random_sample=False,
                                )
    
    return trainData, validData

def get_test_classify_binary_loader(args):
    testData = TorchioDataloader(
                                iter_num = len(args.valid_image_list),
                                batch_size= args.batch_size,
                                file_ls_dic ={
                                    'image':{'file_list':args.valid_image_list, 'dtype':'float'},
                                    'label':{'file_list':args.valid_label_list, 'dtype':'str_int'},
                                    },
                                transform=transforms.Compose([
                                    MultiThreshNormalized(include=['image'],thresh_ls=[[-1000, -200, 200, 1000]], norm_ls=[[0,0.2,0.9,1]]),
                                    Pad2Shape(size=args.crop_shape),
                                    Resize(size=args.crop_shape),
                                    # Option2Onehot(keys=['label'], class_num=args.class_num),
                                    ToTensor(),
                                ]),
                                load_memory=True,
                                load_numworker= 6,
                                respacing=True, target_spacing=1,
                                random_sample=False,
                                )
    
    return testData

def get_classify_loader(args):
    trainData = TorchioDataloader(
                                iter_num= args.total_step,
                                batch_size = args.batch_size,
                                file_ls_dic ={
                                    'image':{'file_list':args.train_image_list, 'dtype':'float'},
                                    'label':{'file_list':args.train_label_list, 'dtype':'txt'},
                                    },
                                transform=transforms.Compose([
                                    MultiThreshNormalized(include=['image'],thresh_ls=[[-1000, -200, 200, 1000]], norm_ls=[[0,0.2,0.9,1]]),
                                    Pad2Shape(size=args.crop_shape),
                                    Resize(size=args.crop_shape),
                                    # RandomElasticDeformation(num_control_points=[5,5,5], max_displacement=[5,5,5], include=['image'], prob=0.5),
                                    # RandomAffine(scales=[0.2,0.2,0.2], degrees=[10,10,10], include=['image'], prob=0.5),
                                    RandomTranspose(include=['image'], prob=0.5),
                                    RandomFlip(include=['image'], axes=[0,1,2], prob=0.5),
                                    RandomAnisotropy(downsampling=(1, 2), include=['image'], prob=0.5),
                                    RandomNoise(mean=0, std=0.1, include=['image'], prob=0.0),
                                    Option2Onehot(keys=['label'], class_num=args.class_num),
                                    ToTensor(),
                                ]),
                                load_memory = args.load_memory,
                                memory_length = args.memory_length,
                                load_numworker = args.num_workers
                                )
    validData = TorchioDataloader(
                                iter_num = len(args.valid_image_list),
                                batch_size= args.batch_size,
                                file_ls_dic ={
                                    'image':{'file_list':args.valid_image_list, 'dtype':'float'},
                                    'label':{'file_list':args.valid_label_list, 'dtype':'txt'},
                                    },
                                transform=transforms.Compose([
                                    MultiThreshNormalized(include=['image'],thresh_ls=[[-1000, -200, 200, 1000]], norm_ls=[[0,0.2,0.9,1]]),
                                    Pad2Shape(size=args.crop_shape),
                                    Resize(size=args.crop_shape),
                                    Option2Onehot(keys=['label'], class_num=args.class_num),
                                    ToTensor(),
                                ]),
                                load_memory=True,
                                load_numworker= 6,
                                respacing=True, target_spacing=1,
                                random_sample=False,
                                )
    
    return trainData, validData


def get_test_classify_loader(args):
    testData = TorchioDataloader(
                                iter_num = len(args.valid_image_list),
                                batch_size= args.batch_size,
                                file_ls_dic ={
                                    'image':{'file_list':args.valid_image_list, 'dtype':'float'},
                                    'label':{'file_list':args.valid_label_list, 'dtype':'txt'},
                                    },
                                transform=transforms.Compose([
                                    MultiThreshNormalized(include=['image'],thresh_ls=[[-1000, -200, 200, 1000]], norm_ls=[[0,0.2,0.9,1]]),
                                    Pad2Shape(size=args.crop_shape),
                                    Resize(size=args.crop_shape),
                                    Option2Onehot(keys=['label'], class_num=args.class_num),
                                    ToTensor(),
                                ]),
                                load_memory=True,
                                load_numworker= 6,
                                respacing=True, target_spacing=1,
                                random_sample=False,
                                )
    
    return testData