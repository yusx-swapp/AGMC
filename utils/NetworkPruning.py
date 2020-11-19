import sys
sys.path.append("..")

import torch.nn.utils.prune as prune
import copy
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
import torch
import torchvision.datasets as datasets


import DNN.resnet as resnet
def pruning_imagnet(net,a_list):
    if not isinstance(net, nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    i = 0
    for name, module in newnet.named_modules():
        if isinstance(module, nn.Conv2d):
            # print("Sparsity ratio",a_list[i])
            prune.ln_structured(module, name='weight', amount=float(1 - a_list[i]), n=2, dim=0)
            i += 1
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=float(1-a_list[i]))
            i += 1
    return newnet
def channel_pruning(net,a_list):
    '''
    :param net: DNN
    :param a_list: pruning rate
    :return: newnet (nn.Module): a newnet contain mask that help prune network's weight
    '''

    if not isinstance(net,nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    i=0
    for name, module in newnet.named_modules():
        if isinstance(module, nn.Conv2d):
            #print("Sparsity ratio",a_list[i])
            prune.ln_structured(module, name='weight', amount=float(1-a_list[i]), n=2, dim=0)
            i+=1

    return newnet

def unstructured_pruning(net,a_list):
    if not isinstance(net,nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    i=0
    for name, module in newnet.named_modules():
        if isinstance(module, nn.Conv2d):
            #print("Sparsity ratio",a_list[i])
            prune.random_unstructured(module, name='weight', amount=float(1-a_list[i]))
            i+=1

    return newnet
def l1_unstructured_pruning(net,a_list):
    if not isinstance(net,nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    i=0
    for name, module in newnet.named_modules():
        if isinstance(module, nn.Conv2d):
            #print("Sparsity ratio",a_list[i])
            prune.l1_unstructured(module, name='weight', amount=float(1-a_list[i]))
            i+=1

    return newnet
if __name__ == '__main__':



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #55 layers
    net = resnet.__dict__['resnet56']()
    net = torch.nn.DataParallel(net)
    net.to(device)
    checkpoint = torch.load('../DNN/pretrained_models/resnet56-4bfd9763.th')

    net.load_state_dict(checkpoint['state_dict'])


