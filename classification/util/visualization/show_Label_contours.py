import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure,draw


def show_label_contours(label, img):
    '''
    :param label: int,Length*Height
    :param img: float,Length*Height
    '''
    for ii in range(img.shape[1]):
        plt.imshow(img[ii], zorder=10,cmap='gray')
        contours3 = measure.find_contours(label[ii], 0.1)
        for n, contour in enumerate(contours3):
            plt.plot(contour[:, 1], contour[:, 0], 'g', zorder=20)
        plt.show()


CTpath = '/lyc/Head-Neck-CT/3D_data/valid/liming/'
file_wanted = ['Img.npy', 'label.npy']
img = np.load(CTpath + file_wanted[0])
label = np.load(CTpath + file_wanted[1])
show_label_contours(label,img)