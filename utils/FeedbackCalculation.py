import time

from torch import optim
from torchvision import models



from FineTune import train_model, train_model_top5
from utils.NetworkPruning import *
from utils.SplitDataset import get_split_valset_ImageNet

import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np, numpy


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def top5validate(val_loader, device,model, criterion):
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()
    losses = AverageMeter()
    with torch.set_grad_enabled(False):
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time

            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            val_top1.update(acc1.item(), input.size(0))
            val_top5.update(acc5.item(), input.size(0))


            # measure elapsed time

    return val_top1.avg,val_top5.avg

def validate(val_loader, device,model, criterion):
    """
    Run evaluation
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            # if args.half:
            #     input_var = input_var.half()

            # compute output
            output = model(input_var)
            #print(output.shape,target_var.shape)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            if i % 50 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def ErrorCaculation_CIFAR(model,val_loader,device,root='../DNN/datasets'):
    cudnn.benchmark = True


    # evaluate on validation set
    criterion = nn.CrossEntropyLoss().to(device)
    acc = validate(val_loader,device, model, criterion)

        #model = torch.load('model.pkl')
    return 100-acc
def ErrorCaculation(model,val_loader,device):
    cudnn.benchmark = True


    # evaluate on validation set
    criterion = nn.CrossEntropyLoss().to(device)
    acc = validate(val_loader,device, model, criterion)

        #model = torch.load('model.pkl')
    return 100-acc
def FlopsCaculation_(DNN,H,W,a_list):



    H_in, W_in = H, W
    Flops = []
    i=0
    for name, module in DNN.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            C_in = module.in_channels
            if i > 0:
                C_in = np.floor((a_list[i-1]) * C_in)
            C_out = module.out_channels
            C_out = np.floor((a_list[i]) * C_out)

            stride_h,stride_w = module.stride
            kernel_h,kernel_w = module.kernel_size
            padding = module.padding
            if padding != (0,0):
                H_out = H_in
                W_out = W_in
            else:
                H_out = (H_in - kernel_h) / stride_h + 1
                W_out = (W_in - kernel_w) / stride_w + 1
            Flop = H_out * W_out * (C_in * (2 * kernel_h * kernel_w - 1) + 1) * C_out
            Flops.append(Flop)
            H_in = H_out
            W_in = W_out
            i+=1


    return Flops

def FlopsCaculation(DNN,H,W):
    H_in, W_in = H, W
    Flops = []
    for name, module in DNN.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            C_in = module.in_channels
            C_out = module.out_channels
            stride_h,stride_w = module.stride
            kernel_h,kernel_w = module.kernel_size
            padding = module.padding
            if padding != (0,0):
                H_out = H_in
                W_out = W_in
            else:
                H_out = (H_in - kernel_h) / stride_h + 1
                W_out = (W_in - kernel_w) / stride_w + 1
            Flop = H_out * W_out * (C_in * (2 * kernel_h * kernel_w - 1) + 1) * C_out
            Flops.append(Flop)
            H_in = H_out
            W_in = W_out


    return Flops
def ErrorCaculation_MNIST(net,root='./DNN/datasets'):
    test_dataset = dsets.MNIST(root=root,
                               train=False,
                               transform=transforms.ToTensor(),download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False)
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    for images, labels in test_loader:
        # images = Variable(images.view(-1, 28 * 28))

        # 获得输出，输出的大小为(batch_size,10)
        outputs = net(images)

        # 获得预测值，输出的大小为(batch_size,1)
        _, predicted = torch.max(outputs.data, 1)

        # labels的size是(100,)
        total += labels.size(0)

        # 返回的是预测值和标签值相等的个数
        correct += (predicted == labels).sum()

    accuracy = 100 * correct // total
    error = 100-accuracy
    return error

def ErrorCaculation_FineTune(model,train_loader,val_loader,device):
    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print("Fine Tuning ...................")
    model, val_acc_history, best_acc=\
        train_model(model, dataloaders, criterion, optimizer_ft,device, num_epochs=20, is_inception=False)
    return model, val_acc_history, best_acc

def ErrorCaculation_ImageNet(model,train_loader,val_loader,device):
    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print("Fine Tuning ...................")
    model, top1, top5 = \
        train_model_top5(model, dataloaders, criterion, optimizer_ft, device, num_epochs=30, is_inception=False)
    return model, top1, top5
def RewardCaculation_ImageNet(args,a_list,n_layers,DNN,best_accuracy,train_loader,val_loader,device,root = './logs'):
    if len(a_list) < n_layers:
        return 0

    new_net = pruning_cp_fg(DNN,a_list)
    #new_net = unstructured_pruning(DNN,a_list)
    #new_net=l1_unstructured_pruning(DNN,a_list)

    new_net,_, acc = ErrorCaculation_ImageNet(new_net,train_loader,val_loader,device)

    error = 100-acc
    if acc > best_accuracy:
        best_accuracy = acc
        torch.save(new_net.state_dict(), root+'/'+args.model+'.pkl')
        f = open(root+"/action_list.txt", "w")
        print(a_list)
        for line in a_list:
            f.write(str(line))
            f.write('\n')
        f.close()
    print("best accuracy",best_accuracy)
    reward = error*-1
    return float(reward),float(best_accuracy)

def RewardCaculation_CIFAR(args,a_list,n_layers,DNN,best_accuracy,train_loader,val_loader,root = './logs'):
    if len(a_list) < n_layers:
        return 0

    new_net = network_pruning(DNN,a_list,args)
    device = torch.device(args.device)
    new_net,_, acc = ErrorCaculation_FineTune(new_net,train_loader,val_loader,device)

    error = 100-acc
    if acc > best_accuracy:
        best_accuracy = acc
        torch.save(new_net.state_dict(), root+'/'+args.model+'.pkl')
        f = open(root+"/action_list.txt", "w")

        for line in a_list:
            f.write(str(line))
            f.write('\n')
        f.close()
    print("Best Accuracy of Compressed Model ",best_accuracy)
    reward = error*-1
    return float(reward),float(best_accuracy)

def RewardCaculation(args,a_list,n_layers,DNN,best_accuracy,train_loader,val_loader,root = './logs'):
    if len(a_list) < n_layers:
        return 0

    new_net = network_pruning(DNN, a_list, args)
    device = torch.device(args.device)

    if args.dataset == 'cifar10':
        new_net, _, acc = ErrorCaculation_FineTune(new_net, train_loader, val_loader, device)
    elif args.dataset == 'ILSVRC':
        new_net, _, acc = ErrorCaculation_ImageNet(new_net, train_loader, val_loader, device)
    error = 100 - acc
    if acc > best_accuracy:
        best_accuracy = acc
        torch.save(new_net.state_dict(), root + '/' + args.model + '.pkl')
        f = open(root + "/action_list.txt", "w")

        for line in a_list:
            f.write(str(line))
            f.write('\n')
        f.close()
    print("Best Accuracy of Compressed Model ", best_accuracy)
    reward = error * -1
    return float(reward), float(best_accuracy)

