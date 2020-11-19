import copy

import numpy
import torch
import torch.nn as nn
from torch import optim

from utils.FeedbackCalculation import *
from utils.NN2Graph import *
torch.backends.cudnn.deterministic = True

from utils.SplitDataset import get_split_valset_CIFAR, get_split_train_valset_CIFAR

def train_model_top5(model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False):

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()

        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        print("train", ' * Prec@1 {top1.avg:.3f}'
              .format(top1=top1))
        print("train", ' * Prec@5 {top5.avg:.3f}'
              .format(top5=top5))
    model.eval()  # Set model to evaluate mode
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
        print("val", ' * Prec@1 {top1.avg:.3f}'
              .format(top1=val_top1))
        print("val", ' * Prec@5 {top5.avg:.3f}'
              .format(top5=val_top5))
    return model, val_top1.avg,val_top5.avg
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        #validate(dataloaders['val'], device, model, criterion)
        losses = AverageMeter()
        top1 = AverageMeter()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                #print(labels)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.

                    outputs = model(inputs)

                    loss = criterion(outputs, labels)



                    prec1 = accuracy(outputs.data, labels)[0]
                    losses.update(loss.item(), inputs.size(0))
                    top1.update(prec1.item(), inputs.size(0))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics

            print(phase,' * Prec@1 {top1.avg:.3f}'
                  .format(top1=top1))
            #print("aaa",top1.avg)

            # deep copy the model
            if phase == 'val' and top1.avg > best_acc:
                best_acc = top1.avg
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(top1.avg)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history,best_acc
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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, n_class = get_split_train_valset_CIFAR('cifar10', 256, 4,20000, 5000, data_root='DNN/datasets',
                                                               use_real_val=True,
                                                               shuffle=True)
    net = resnet.__dict__['resnet56']()
    net = torch.nn.DataParallel(net).cuda()
    net.to(device)
    # checkpoint = torch.load('DNN/pretrained_models/resnet56-4bfd9763.th',map_location=device)
    # net.load_state_dict(checkpoint['state_dict'])
    cudnn.benchmark = True

    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    model, val_acc_history, best_acc=\
        train_model(net, dataloaders, criterion, optimizer_ft,device, num_epochs=20, is_inception=False)
