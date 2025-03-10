import os
import sys
import numpy as np
import SimpleITK as sitk
import argparse

from .evluation_index import dc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-predic_root', type=str, required=True, help='/path/to/your/nnUNet_raw/Dataset00x_xxx/PASTA_ft_pred')
    parser.add_argument('-label_root', type=str, required=True, help='/path/to/your/nnUNet_raw/Dataset00x_xxx/labelsTr')
    parser.add_argument('-fg_class_num', type=int, required=True, help='foreground class number. For example, 2 for lung and lung tumor segmentation ')
    args = parser.parse_args()

    predic_root = args.predic_root
    label_root = args.label_root
    fg_class_num = args.fg_class_num  # foreground class number. For example, 2 for lung and lung tumor segmentation 
    
    nii_ls = os.listdir(label_root)
    

    dice_scores_per_class = [[] for _ in range(fg_class_num)]  # dice scores for each class
    dice_scores_per_sample = []  # average dice scores for each sample

    # iterate over all samples
    for file_name in nii_ls:
        pred_path = os.path.join(predic_root, file_name)
        label_path = os.path.join(label_root, file_name)

        pred_nii = sitk.ReadImage(pred_path)
        label_nii = sitk.ReadImage(label_path)

        pred_array = sitk.GetArrayFromImage(pred_nii)
        label_array = sitk.GetArrayFromImage(label_nii)

        dice_per_sample = []
        for class_index in range(fg_class_num):
            pred_binary = (pred_array == class_index+1).astype(np.uint8)
            label_binary = (label_array == class_index+1).astype(np.uint8)
            if np.sum(label_binary)> 0:
                dice = dc(pred_binary, label_binary)
                dice_per_sample.append(dice)
                dice_scores_per_class[class_index].append(dice)

                print(f"Sample: {file_name}, Class: {class_index+1}, Dice: {dice:.4f}")

        avg_dice_per_sample = np.mean(dice_per_sample)
        dice_scores_per_sample.append(avg_dice_per_sample)

        print(f"Sample: {file_name}, Average Dice: {avg_dice_per_sample:.4f}")

    # compute the average dice score for each class
    avg_dice_per_class = [np.mean(scores) for scores in dice_scores_per_class]

    # compute the overall average dice score
    overall_avg_dice = np.mean(dice_scores_per_sample)

    print("\nFinal Results:")
    for class_index in range(fg_class_num):
        print(f"Class {class_index+1}: Average Dice: {avg_dice_per_class[class_index]:.4f}")
    print(f"Overall Average Dice (across all samples and classes): {overall_avg_dice:.4f}")
