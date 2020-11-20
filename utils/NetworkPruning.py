import sys
sys.path.append("..")

import torch.nn.utils.prune as prune
import copy
import torch.nn as nn

def network_pruning(net,a_list,args):
    if not isinstance(net, nn.Module):
        print('Invalid input. Must be nn.Module')
        return

    if args.pruning_method == "cp":
        candidate_net = channel_pruning(net,a_list)
    elif args.pruning_method == "fg":
        candidate_net = l1_unstructured_pruning(net, a_list)
    elif args.pruning_method == "cpfg":
        candidate_net = pruning_cp_fg(net, a_list)
    else:
        raise KeyError
    return candidate_net
def pruning_cp_fg(net,a_list):
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
            prune.l1_unstructured(module, name='weight', amount=float(1-a_list[i]))
            i+=1
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=float(1 - a_list[i]))
            i += 1

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

