import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
import os
def show_performance(method_names, Data, ylabel, output_name, classnum, itertime, edge_color, fill_color):
    fig, ax = plt.subplots(figsize=(2.5,2.8))
    x=[i+1 for i in range(classnum)]
    for ii in range(itertime):
        flierprops = dict(marker='+', markerfacecolor=edge_color[ii], markersize=7,
                          linestyle='none', markeredgecolor=edge_color[ii])
        bp = ax.boxplot(Data[ii], patch_artist=True, flierprops=flierprops)
        for element in ['boxes', 'whiskers', 'means',  'caps']:
            plt.setp(bp[element], color=edge_color[ii])
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)

    ylabel = ylabel
    plt.ylim(0, 1000)
    plt.ylabel(ylabel)
    plt.xticks(x, method_names,rotation=30, ha='right')
    plt.subplots_adjust(left=0.3, right=0.8, top=0.95, bottom=0.25)
    plt.show()
    # fig.savefig(output_name, dpi=400)

data_root = '/lyc/Head-Neck-CT/3D_data/valid/'
output_name = '/home/uestc-c1501b/paper769/pic/Time_compare.png'
patient_list = os.listdir(data_root)
dice_list = [[] for _ in range(4)]
i = 0
classnum = 3
itertime = 1
Data_list = [[[336,304,262,251,290,276,255,273,337,237,252,231,196,199,311], [308,224,208,219,202,288,231,188,269,236,169,182,142,155,156]
                 , [668, 621, 423, 692, 562, 629, 717, 793, 672, 825, 756, 777, 623]]]
method_names = ['Overlay', 'Combine', 'Normal']
edge_color = ['brown']
ylabel = 'User Time (s)'
fill_color = 'white'
all_data = np.asarray(Data_list)
show_performance(method_names, all_data, ylabel, output_name, classnum=classnum, itertime=itertime, edge_color=edge_color, fill_color=fill_color)
