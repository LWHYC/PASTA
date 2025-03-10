from Training.util.binary import dc,assd
import os
import numpy as np
from Training.data_process.data_process_func import load_nifty_volume_as_array

def one_hot(img, nb_classes):
    hot_img = np.zeros([nb_classes]+list(img.shape))
    for i in range(nb_classes):
        hot_img[i][np.where(img == i)] = 1
    return hot_img

def evaluation(folder, evaluate_dice, evaluate_assd):
    patient_list = os.listdir(folder)
    dice_all_data = []
    assd_all_data = []
    for patient in patient_list:
        s_name = os.path.join(folder, patient + '/label.npy')
        g_name = os.path.join(folder, patient + '/InterSeg.nii.gz')
        s_volume = np.int64(np.load(s_name))
        g_volume = load_nifty_volume_as_array(g_name)
        s_volume = one_hot(s_volume, nb_classes=5)
        g_volume = one_hot(g_volume, nb_classes=5)
        if evaluate_dice:
            dice=[]
            for i in range(5):
                dice.append(dc(g_volume[i], s_volume[i]))
            dice_all_data.append(dice)
            print(patient, dice)
        if evaluate_assd:
            Assd = []
            for i in range(5):
                Assd.append(assd(g_volume[i], s_volume[i]))
            assd_all_data.append(Assd)
            print(patient, Assd)
    if evaluate_dice:
        dice_all_data = np.asarray(dice_all_data)
        dice_mean = [dice_all_data.mean(axis = 0)]
        dice_std  = [dice_all_data.std(axis = 0)]
        np.savetxt(folder + '/dice_all.txt', dice_all_data)
        np.savetxt(folder + '/dice_mean.txt', dice_mean)
        np.savetxt(folder + '/dice_std.txt', dice_std)
        print('dice mean ', dice_mean)
        print('dice std  ', dice_std)
    if evaluate_assd:
        assd_all_data = np.asarray(assd_all_data)
        assd_mean = [assd_all_data.mean(axis = 0)]
        assd_std  = [assd_all_data.std(axis = 0)]
        np.savetxt(folder + '/dice_all.txt', assd_all_data)
        np.savetxt(folder + '/dice_mean.txt', assd_mean)
        np.savetxt(folder + '/dice_std.txt', assd_std)
        print('assd mean ', assd_mean)
        print('assd std  ', assd_std)



evaluate_dice=False
evaluate_assd=True
if __name__ =='__main__':
    folder = '/lyc/Head-Neck-CT/3D_data/valid'
    evaluation(folder, evaluate_dice, evaluate_assd)


