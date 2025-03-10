import os
import numpy as np
import skimage.measure
from data_process.data_process_func import *
import matplotlib.pyplot as plt

data_root = '/home/uestc-c1501c/StructSeg/Lung_GTV_n/'
filename_list = ['data.nii.gz', 'label.nii.gz']
modelist = [ 'valid','train']
save_as_nifty = True
respacing = False
norm = True
normalize = img_multi_thresh_normalized
organ_num = []; organ_bbox=[]


for mode in modelist:
    filelist = os.listdir(os.path.join(data_root, mode))
    filenum = len(filelist)
    for ii in range(filenum):
        data_path = os.path.join(data_root, mode, filelist[ii], filename_list[0])
        label_path = os.path.join(data_root, mode, filelist[ii], filename_list[1])
        data, spacing = load_nifty_volume_as_array(data_path, return_spacing=True)
        spacing.reverse()
        label = np.int8(load_nifty_volume_as_array(label_path))
        labeled_label = skimage.measure.label(label, connectivity=1)
        label_property = skimage.measure.regionprops(labeled_label)
        if len(label_property)>1:
            print(data_path)
        for sublabel_property in label_property:
            if sublabel_property.area>10:
                organ_num.append(sublabel_property.area)
                organ_bbox.append(list(sublabel_property.bbox))
                print(sublabel_property.area, sublabel_property.bbox)

num_array = np.asarray(organ_num)
print(num_array.shape)
bbox_array = np.asarray(organ_bbox)
size_array = bbox_array[:, -3::]-bbox_array[:, 0:3]
fig, axes = plt.subplots(2,2)
axes[0,0].hist(num_array)
axes[0,0].set_title('voxel num', fontsize=20)
axes[0,1].hist(size_array[:,0:1])
axes[0,1].set_title('z', fontsize=20)
axes[1,0].hist(size_array[:,1:2])
axes[1,0].set_title('x', fontsize=20)
axes[1,1].hist(size_array[:,2:3])
axes[1,1].set_title('y', fontsize=20)
plt.show()
