import argparse
import os
import time

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn.utils import prune
from torchvision import models
from torchvision.transforms import transforms

from data.resnet import LambdaLayer

from utils.NetworkPruning import real_pruning, network_pruning, channel_pruning_mobilenet
from utils.SplitDataset import get_split_valset_CIFAR, get_split_train_valset_CIFAR, get_split_valset_ImageNet, \
    get_dataset
from data import resnet
import torch.backends.cudnn as cudnn

from utils.TestCandidateModel import EvalCompressedModel
from utils.feedback_calculation import top5validate


class Identity1(nn.Module):
    def __init__(self,out_c):
        super(Identity1, self).__init__()
        self.out_c = out_c
    def forward(self, x):
        y = torch.zeros([x.shape[0],self.out_c,x.shape[2],x.shape[3]])
        y[:x.shape[0], :x.shape[1], :x.shape[2], :x.shape[3]] = x
        return y.cuda()


class Identity(nn.Module):
    def __init__(self,out_c):
        super(Identity, self).__init__()
        self.out_c = out_c
    def forward(self, x):
        return x[:,:self.out_c,:,:].cuda()
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
def parse_args():
    parser = argparse.ArgumentParser(description='AMC search script')
    #parser.add_argument()
    parser.add_argument('--job', default='train', type=str, help='support option: train/export')
    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
    #graph encoder
    parser.add_argument('--node_feature_size', default=50, type=int, help='the initial node feature size')
    parser.add_argument('--pool_strategy', default='mean', type=str, help='pool strategy(mean/diff), defualt:mean')
    parser.add_argument('--embedding_size', default=30, type=int, help='embedding size of DNN\'s hidden layers')
    parser.add_argument('--model_root', default=None, type=str, help='compressed model dir')

    # env
    parser.add_argument('--model', default='mobilenet', type=str, help='model to prune')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to use (cifar/imagenet)')
    parser.add_argument('--data_root', default='data', type=str, help='dataset path')
    parser.add_argument('--preserve_ratio', default=0.5, type=float, help='preserve ratio of the model')
    parser.add_argument('--lbound', default=0.2, type=float, help='minimum preserve ratio')
    parser.add_argument('--rbound', default=1., type=float, help='maximum preserve ratio')
    parser.add_argument('--reward', default='acc_reward', type=str, help='Setting the reward')
    parser.add_argument('--acc_metric', default='acc5', type=str, help='use acc1 or acc5')
    parser.add_argument('--use_real_val', dest='use_real_val', action='store_true')
    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')

    # pruning

    parser.add_argument('--real_compressed', default=True, type=boolean_string,
                        help='True/False real pruning?')

    parser.add_argument('--compression_ratio', default=0.5, type=float,
                        help='method to prune (fg/cp for fine-grained and channel pruning)')
    parser.add_argument('--pruning_method', default='cp', type=str,
                        help='method to prune (fg/cp for fine-grained and channel pruning)')
    parser.add_argument('--n_calibration_batches', default=60, type=int,
                        help='n_calibration_batches')
    parser.add_argument('--n_points_per_layer', default=10, type=int,
                        help='method to prune (fg/cp for fine-grained and channel pruning)')
    parser.add_argument('--channel_round', default=8, type=int, help='Round channel to multiple of channel_round')
    # ddpg
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for actor')
    parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
    parser.add_argument('--warmup', default=25, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=1., type=float, help='')
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=100, type=int, help='memory size for each layer')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    # noise (truncated normal distribution)
    parser.add_argument('--init_delta', default=0.5, type=float,
                        help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.95, type=float,
                        help='delta decay during exploration')
    # training
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
    parser.add_argument('--output', default='./logs', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_episode', default=800, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
    parser.add_argument('--n_worker', default=16, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=50, type=int, help='number of data batch size')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # export
    parser.add_argument('--ratios', default=None, type=str, help='ratios for pruning')
    parser.add_argument('--channels', default=None, type=str, help='channels after pruning')
    parser.add_argument('--export_path', default=None, type=str, help='path for exporting models')
    parser.add_argument('--use_new_input', dest='use_new_input', action='store_true', help='use new input feature')

    return parser.parse_args()

def load_model(model_name):

    if model_name == "resnet56":
        net = resnet.__dict__['resnet56']()
        net = torch.nn.DataParallel(net).cuda()

    elif model_name == "resnet44":
        net = resnet.__dict__['resnet44']()
        net = torch.nn.DataParallel(net).cuda()

    elif model_name == "resnet110":
        net = resnet.__dict__['resnet110']()
        net = torch.nn.DataParallel(net).cuda()

    elif model_name == "resnet32":
        net = resnet.__dict__['resnet32']()
        net = torch.nn.DataParallel(net).cuda()

    elif model_name == "resnet20":
        net = resnet.__dict__['resnet20']()
        net = torch.nn.DataParallel(net).cuda()

    elif model_name == "vgg16":
        net = models.vgg16(pretrained=True).eval()
        net = torch.nn.DataParallel(net)

    elif model_name == "mobilenetv2":
        net = models.mobilenet_v2(pretrained=True).eval()
        net = torch.nn.DataParallel(net)
    elif model_name == "mobilenet":
        from data.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
        sd = torch.load("data/checkpoints/mobilenet_imagenet.pth.tar")
        if 'state_dict' in sd:  # a checkpoint but not a state_dict
            sd = sd['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        net.load_state_dict(sd)
        net = net.cuda()
        net = torch.nn.DataParallel(net)
    else:
        raise KeyError
    return net

    return train_loader, val_loader, n_class




'''
if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)


    if args.dataset == "ILSVRC":
        path = args.data_root
        train_loader, val_loader, n_class = get_split_valset_ImageNet("ImageNet", 128, 4, 1000, 3000,
                                                                      data_root=path,
                                                                      use_real_val=True, shuffle=True)

    elif args.dataset == "cifar10":
        path = os.path.join(args.data_root, "datasets")
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)

        val_loader = torch.utils.data.DataLoader(val, batch_size=256, shuffle=False,
                                               num_workers=4, pin_memory=True)


    net = load_model(args.model)

    val_top1,val_top5 = EvalCompressedModel(args, net, val_loader, device)
    if args.dataset == "cifar10":
        print("Top-1 val", ' * Prec@1 {top1:.3f}'
              .format(top1=val_top1))
        print("Top-5 val", ' * Prec@5 {top5:.3f}'
              .format(top5=val_top5))
    elif args.dataset == "ILSVRC":
        print("Top-5 val", ' * Prec@5 {top5:.3f}'
              .format(top5=val_top5))
              
'''
if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)

    if args.dataset == "ILSVRC":
        path = args.data_root
        #train_loader, val_loader, n_class = get_dataset("imagenet", 128, 4, data_root=path)
        train_loader, val_loader, n_class = get_split_valset_ImageNet("ImageNet", 32, 4, 1000, 3000,
                                                                       data_root=path,
                                                                       use_real_val=True, shuffle=True)

    elif args.dataset == "cifar10":
        path = os.path.join(args.data_root, "datasets")
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)

        val_loader = torch.utils.data.DataLoader(val, batch_size=args.bsize, shuffle=False,
                                                 num_workers=4, pin_memory=True)


    net = load_model(args.model)
    net.to(device)
    path = args.model_root

    net = channel_pruning_mobilenet(net, torch.ones(100, 1))
    #net = network_pruning(net, [0 for i in range(500)], args)#create mask for pruned network
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    #ResNet 不用这个
    for name, module in net.named_modules(): #remove mask

        if isinstance(module, nn.Conv2d):
            if args.model =='mobilenet' and module.groups == module.in_channels:#TODO
                continue
            module = prune.remove(module,name='weight')
    print(net)

    if args.real_compressed:
        #print("aaaaa")
        net = real_pruning(args, net)
    cudnn.benchmark = True
    print(net)

    criterion = nn.CrossEntropyLoss().to(device)

    start = time.time()
    val_top1,val_top5 = top5validate(val_loader, device, net, criterion)
    end = time.time()

    print(end - start)
    print((end - start)/len(val_loader)/args.bsize)

    start = time.time()
    val_top1,val_top5 = top5validate(val_loader, device, net, criterion)
    end = time.time()
    print(end - start)
    print((end - start)/len(val_loader)/args.bsize)

    if args.real_compressed:
            torch.save(net.state_dict(), "logs" + '/' + args.model + '_real_compressed.pkl')





#python memory_saving.py --dataset cifar10 --model resnet20 --pruning_method cp --data_root ./data --model_root ./logs/resnet20.pkl
#python memory_saving.py --dataset ILSVRC --model mobilenet --pruning_method cp --real_compressed False --data_root data/datasets --model_root ./logs/mobilenet.pkl
#python memory_saving.py --dataset ILSVRC --model mobilenet --pruning_method cp --real_compressed True --data_root data/datasets --model_root ./logs/mobilenet.pkl
#python memory_saving.py --dataset ILSVRC --model vgg16 --pruning_method cp --real_compressed True --data_root data/datasets --model_root ./logs/vgg16.pkl
#python memory_saving.py --dataset ILSVRC --model vgg16 --pruning_method cp --real_compressed False --data_root data/datasets --model_root ./logs/vgg16.pkl
#python memory_saving.py --dataset cifar10 --model resnet56 --real_compressed True --data_root ./data --model_root ./logs/resnet56.pkl
#python memory_saving.py --dataset cifar10 --model resnet56 --real_compressed False --data_root ./data --model_root ./data/pretrained_models/resnet56-4bfd9763.th

#python memory_saving.py --dataset cifar10 --model resnet20 --real_compressed False --data_root ./data --model_root ./data/pretrained_models/resnet20-12fca82f.th
#python memory_saving.py --dataset cifar10 --model resnet20 --real_compressed True --data_root ./data --model_root ./logs/resnet20.pkl

