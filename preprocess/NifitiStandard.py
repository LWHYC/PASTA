#!/usr/bin/env python
from __future__ import absolute_import, print_function
import os
import sys

sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import SimpleITK as sitk
import nibabel
from nibabel.orientations import ornt_transform
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import time
import argparse
from scipy import ndimage



def standard_img(target_orient, img_path, img_save_path):
    os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
    time0 = time.time()
    ##### reorient
    
    img_nii = nibabel.load(img_path)
    img_orient = nibabel.io_orientation(img_nii.affine)
    transform = ornt_transform(img_orient, target_orient)
    img_nii = img_nii.as_reoriented(transform)
    img_array = img_nii.get_fdata()
    time1 = time.time()
    
    ##### respacing
    spacing = list(img_nii.header.get_zooms())
    if len(img_array.shape) == 4:
        img_array = img_array[:, :, :, 0]
        spacing = spacing[:3]
    target_spacing = [1,1,1]
    zoomfactor = list(np.array(spacing) / np.array(target_spacing))
    
    spacing = target_spacing
    if img_array.min()<-10: # CT
        img_array = ndimage.zoom(img_array, zoom=zoomfactor, order=1)
    else: # label
        img_array = ndimage.zoom(img_array, zoom=zoomfactor, order=0)
        
    img_nii = nibabel.Nifti1Image(img_array, img_nii.affine)
    img_nii.header.set_zooms(target_spacing)
    nibabel.save(img_nii, img_save_path)
    img_nii = sitk.ReadImage(img_save_path)
    sitk.WriteImage(img_nii, img_save_path)

    print(img_save_path, time1-time0, time.time() - time1)
    return


if __name__ == '__main__':
    
    ##### Standardize the orientation and spacing of the nifti files.
    ##### Please make sure the input nifti files have the correct orientation and spacing, or the output files will be wrong.
    ##### For the CT files(min<-10), we use linear interpolation to resample the images.
    ##### For the label files(min>-10), we use nearest interpolation to resample the images.
    ##### Target spacing: 1x1x1mm
    ##### Target orientation: RAS
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-indir', type=str, required=True, help='/path/to/original/data/root')
    parser.add_argument('-outdir', type=str, required=True, help='/path/to/save/data/root')
    args = parser.parse_args()

    data_root = args.indir
    save_root = args.outdir

    os.makedirs(save_root, exist_ok=True)

    target_orient = [[ 0., -1.], [ 1., -1.], [ 2.,  1.]]
    filelist = []
    for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith('.nii.gz') or file.endswith('.nii'):
                    filelist.append(os.path.join(root, file))


    num_threads = 8
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for img_path in filelist:
            img_save_path = img_path.replace(data_root, save_root)
            futures.append(executor.submit(standard_img, target_orient, img_path, img_save_path))
            
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Standarding"):
            pass


    print('---done!')