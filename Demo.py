import copy

import numpy
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch_geometric.data import Data
import torch.nn.utils.prune as prune

from DNN import resnet
from DNN.Example_CNN import CNN
from ModelCompression import DNNCompression
from rl_agents.Agent import Agent
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from utils.FeedbackCalculation import *
from utils.GetAction import get_action
from torchvision import models
from utils.NetworkPruning import *
from utils.TestCandidateModel import LoadCompressedModel


def prune_net(net,a_list):
    '''

    :param net: DNN
    :param a_list: pruning rate
    :return: newnet (nn.Module): a newnet contain mask that help prune network's weight
    '''
    """Prune net's weights that have abs(value) approx. 0
    Function that will be use when an iteration is reach

    """
    if not isinstance(net,nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)

    i=0
    for name, module in newnet.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print("Sparsity ratio",a_list[i])
            prune.ln_structured(module, name='weight', amount=float(a_list[i]), n=2, dim=0)
            i+=1

    return newnet

def ComputeReward(a_list,n_layers,DNN,Flops):
    if len(a_list) < n_layers:
        return 0

    total_Flops=0
    for i in  range(len(a_list)):
        total_Flops+=Flops[i]*a_list[i]

    new_net = prune_net(DNN,a_list)
    error = utils.FeedbackCalculation.ErrorCaculation(new_net)
    print("error", error,"accuracy", 100-error)

    reward = (error + numpy.log(total_Flops))*-1
    return reward

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root="DNN/datasets", train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss().to(device)
    net = models.mobilenet_v2(pretrained=True).eval()
    #net = torch.nn.DataParallel(net)
    net.to(device)
    train_loader, val_loader, n_class = get_split_valset_ImageNet("ImageNet", 128, 4,1000, 5000, data_root='DNN/datasets/',
                      use_real_val=True, shuffle=True)
    LoadCompressedModel(net, 'logs/mobilev2.pkl', val_loader, device)
    #validate(val_loader, device,net, criterion)