import os
import time
import argparse
import shutil
import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter
from torch.nn.utils import prune
from torch.optim import Adam
from torchvision import models

from data import resnet
from lib.Utils import accuracy, AverageMeter, progress_bar, get_output_folder
from utils.network_pruning import channel_pruning_mobilenet, channel_pruning, real_pruning
from utils.split_data import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='AMC fine-tune script')
    parser.add_argument('--model', default='mobilenet', type=str, help='name of the model to train')
    parser.add_argument('--dataset', default='imagenet', type=str, help='name of the dataset to train')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--n_worker', default=4, type=int, help='number of data loader worker')
    parser.add_argument('--lr_type', default='exp', type=str, help='lr scheduler (exp/cos/step3/fixed)')
    parser.add_argument('--n_epoch', default=150, type=int, help='number of epochs to train')
    parser.add_argument('--start_epoch', default=0, type=int, help='the start epochs to train')
    parser.add_argument('--device', default="cuda", type=str, help='device')

    parser.add_argument('--wd', default=4e-5, type=float, help='weight decay')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    # resume
    parser.add_argument('--ckpt_path', default=None, type=str, help='checkpoint path to resume from')
    # run eval
    parser.add_argument('--eval', action='store_true', help='Simply run eval')

    return parser.parse_args()


def get_model():
    print('=> Building model..')
    if args.model == 'mobilenet':
        from data.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
        net = torch.nn.DataParallel(net)

        net = channel_pruning_mobilenet(net, torch.ones(100, 1))

        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            checkpoint = torch.load(args.ckpt_path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)

            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        #
        # if use_cuda and args.n_gpu > 1:
        #     net = torch.nn.DataParallel(net)

    elif args.model == 'mobilenet_0.5flops':
        from data.mobilenet import MobileNet
        net = MobileNet(n_class=1000, profile='0.5flops')
    elif args.model == 'vgg16':
        net = models.vgg16(pretrained=True)
        net = channel_pruning(net,torch.ones(100, 1))

        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            checkpoint = torch.load(args.ckpt_path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
            net.cuda()

            for name, module in net.named_modules(): #remove mask

                if isinstance(module, nn.Conv2d):

                    module = prune.remove(module,name='weight')
            net = real_pruning(args,net)

            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
            #optimizer.load_state_dict(checkpoint['optimizer'])

        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == 'mobilenetv2':
        net = models.mobilenet_v2(pretrained=True)
        net = torch.nn.DataParallel(net)

        net = channel_pruning(net,torch.ones(100, 1))

        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            checkpoint = torch.load(args.ckpt_path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
            # net.cuda()

            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
            #optimizer.load_state_dict(checkpoint['optimizer'])

    elif args.model == "resnet56":
        net = resnet.__dict__['resnet56']()
        net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = os.path.join(args.ckpt_path)

            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
            # optimizer = Adam(net.parameters(), lr=args.lr,weight_decay=args.wd)
    elif args.model == "resnet20":
        net = resnet.__dict__['resnet20']()
        net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = os.path.join(args.ckpt_path)

            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
            optimizer = Adam(net.parameters(), lr=args.lr,weight_decay=args.wd)

# if use_cuda and args.n_gpu > 1:
        #     net = torch.nn.DataParallel(net, list(range(args.n_gpu)))


    else:
        raise NotImplementedError

    #if use_cuda and args.n_gpu > 1:

    return net.cuda(), optimizer


def train(epoch, train_loader):
    print('\nEpoch: %d' % epoch)
    net.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # timing
        batch_time.update(time.time() - end)
        end = time.time()

        # progress_bar(batch_idx, len(train_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
        #              .format(losses.avg, top1.avg, top5.avg))
        if batch_idx % 100 == 0:
            print(batch_idx, len(train_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'.format(losses.avg, top1.avg, top5.avg))
    writer.add_scalar('loss/train', losses.avg, epoch)
    writer.add_scalar('acc/train_top1', top1.avg, epoch)
    writer.add_scalar('acc/train_top5', top5.avg, epoch)


def test(epoch, test_loader, save=True):
    global best_acc
    net.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # timing
            batch_time.update(time.time() - end)
            end = time.time()

            # progress_bar(batch_idx, len(test_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
            #              .format(losses.avg, top1.avg, top5.avg))
            if batch_idx % 100 == 0:
                print(batch_idx, len(test_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'.format(losses.avg, top1.avg, top5.avg))

    if save:
        writer.add_scalar('loss/test', losses.avg, epoch)
        writer.add_scalar('acc/test_top1', top1.avg, epoch)
        writer.add_scalar('acc/test_top5', top5.avg, epoch)

        is_best = False
        if top1.avg > best_acc:
            best_acc = top1.avg
            is_best = True

        print('Current best acc: {}'.format(best_acc))
        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'dataset': args.dataset,
            'state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
            'acc': top1.avg,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_dir=log_dir)


def adjust_learning_rate(optimizer, epoch):
    if args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.n_epoch))
    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_type == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError
    print('=> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, 'ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    best_acc = 0  # best test accuracy
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    print('=> Preparing data..')
    train_loader, val_loader, n_class = get_dataset(args.dataset, args.batch_size, args.n_worker,
                                                    data_root=args.data_root)







    criterion = nn.CrossEntropyLoss()
    print('Using SGD...')
    print('weight decay  = {}'.format(args.wd))
    #optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.wd)
    net,optimizer = get_model()  # real training
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    # optimizer.state_dict()


    if args.eval:  # just run eval
        print('=> Start evaluation...')
        test(0, val_loader, save=False)
    else:  # train
        print('=> Start training...')
        print('Training {} on {}...'.format(args.model, args.dataset))
        log_dir = get_output_folder('./logs', '{}_{}_finetune'.format(args.model, args.dataset))
        print('=> Saving logs to {}'.format(log_dir))
        # tf writer
        writer = SummaryWriter(logdir=log_dir)

        for epoch in range(start_epoch, start_epoch + args.n_epoch):
            lr = adjust_learning_rate(optimizer, epoch)
            train(epoch, train_loader)
            test(epoch, val_loader)

        writer.close()
        print('=>  best top-1 acc: {}%'.format( best_acc))
#64