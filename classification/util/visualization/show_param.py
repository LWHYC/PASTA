import torch

def show_param(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：", str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：", str(l))
        k = k + l
    print("总参数数量和：" + str(k))
