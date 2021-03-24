import copy
import random

import numpy as np
import torch
import torch.nn as nn

from utils.feedback_calculation import flops_caculation_forward

lbound = 0.2    #minimum preserve ratio
rbound = 1      #maximum preserve ratio
init_delta = 0.5
delta_decay = 0.95
warmup = 32
def to_numpy(var):
    use_cuda = torch.cuda.is_available()
    return var.cpu().data.numpy() if use_cuda else var.data.numpy()

def to_tensor(ndarray, requires_grad=False):  # return a float tensor by default
    tensor = torch.from_numpy(ndarray).float()  # by default does not require grad
    if requires_grad:
        tensor.requires_grad_()
    return tensor.cuda() if torch.cuda.is_available() else tensor
def sample_from_truncated_normal_distribution( lower, upper, mu, sigma, size=1):
    from scipy import stats
    return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)

def select_action( a, episode):
    #assert episode >= self.warmup, 'Episode: {} warmup: {}'.format(episode, self.warmup)
    action = (a.reshape(1, -1)).squeeze(0)
    delta = init_delta * (delta_decay ** (episode - warmup))
    # action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
    #action = sample_from_truncated_normal_distribution(lower=lbound, upper=rbound, mu=action,sigma=delta)
    action = np.clip(action, lbound, rbound)

    # self.a_t = action
    return action
def act_share(net,a_list,args):
    a_share = []
    if 'resnet' in args.model:
        #share the pruning index where layers are connected by residual connection

        a_share.append(a_list[0])
        i=1
        for name, module in net.module.layer1.named_children():
            a_share.append(a_list[i])
            a_share.append(a_list[0])
            i+=1
        for name, module in net.module.layer2.named_children():
            a_share.append(a_list[i])
            a_share.append(a_list[0])
            i+=1
        for name, module in net.module.layer3.named_children():
            a_share.append(a_list[i])
            a_share.append(a_list[0])
            i+=1
    elif args.model == 'mobilenet':
        # prune mobilenet block together(share pruning index between depth-wise and point-wise conv)
        i = 0
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.groups == module.in_channels:
                    a_share.append(1)
                else:
                    a_share.append(a_list[i])
                    i += 1
    elif args.model == 'mobilenetv2':
        # prune mobilenet block together(share pruning index between depth-wise and point-wise conv)

        for i in range(len(a_list)):
            a_share.append(a_list[i])
            a_share.append(1)
            a_share.append(1)
        a_share.append(1)
        # i = 0
        # for name, module in net.named_modules():
        #     if isinstance(module, nn.Conv2d):
        #         if module.groups == module.in_channels:
        #             a_share.append(a_list[i])
        #             i += 1
        #         else:
        #             a_share.append(1)
    elif args.model == 'vgg16':
        #Here in VGG-16 we dont need to share the pruning index
        a_share = a_list
    else:
        a_share = a_list
    return np.array(a_share)

def act_restriction_flops(args, a_list,net):
    device = torch.device(args.device)

    if args.dataset == "ILSVRC":
        input_x = torch.randn([1,3,224,224]).to(device)
    elif args.dataset == "cifar10":
        input_x = torch.randn([1,3,32,32]).to(device)


    flops_org, flops = flops_caculation_forward(net, args, input_x)
    total_flops = sum(flops)

    desired_flops = total_flops * args.compression_ratio

    current_flops_org, current_flops = flops_caculation_forward(net, args, input_x, copy.deepcopy(a_list))

    reduced_flops = total_flops - sum(current_flops_org)

    if args.pruning_method == 'cp':
        #TODO

        print("FLOPS ratio:",1-(reduced_flops / total_flops) )
        if reduced_flops < desired_flops:
            duty = desired_flops - reduced_flops

            s_a = len(a_list) - sum(a_list)

            for i in range(len(a_list)):
                a_list[i] -= ((duty * (1 - a_list[i])) / s_a) / flops[i]
                a_list[i] = np.clip(a_list[i], lbound, rbound)

    return act_share(net,a_list,args)

'''
def get_action(a_list,DNN,t,d_FLOPs,reduced,FLOPs,a_max = 0.75):
    desired = d_FLOPs
    total_FLOPs = sum(FLOPs)
    current_FLOPs=flops_caculation(DNN,32,32,a_list)
    reduced_FLOPs = total_FLOPs-sum(current_FLOPs)
    if reduced_FLOPs < desired:
        duty = desired - reduced

    if t == 0:
        reduced = 0
    rest = sum(FLOPs[t+1:])
    duty = desired - reduced - a_max*rest
    #print("duty",duty,reduced,rest)
    #if duty > 0:
    a = torch.tensor(max(a.numpy(),duty/FLOPs[t]))
    reduced = reduced + (a.numpy())*FLOPs[t]
    # if a > 1:
    #     print("Aaaaaaaaaaaaaaaa123jhasdashdasjkhdjahskjdhkjasjdhjahdjahdkjashdjkahkdja\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    return a,reduced
    
    def act_restriction_FLOPs(args, a_list,DNN,desired,FLOPs):
    total_FLOPs = sum(FLOPs)
    if args.dataset == "ILSVRC":
        h,w = 224,224
    elif args.dataset == "cifar10":
        h,w=32,32
    if args.pruning_method == 'cpfg':
        s = []
        i=0
        for name, module in DNN.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                s.append(a_list[i])
                i += 1
            if isinstance(module, torch.nn.Linear):
                i += 1
        current_FLOPs = flops_caculation(DNN, h, w, s)
        reduced_FLOPs = total_FLOPs - sum(current_FLOPs)
        if reduced_FLOPs < desired:
            duty = desired - reduced_FLOPs

            s_a = len(s)-sum(s)
            i=0
            j=0
            for name, module in DNN.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    a_list[i] -= ((duty * (1 - a_list[i])) / s_a) / FLOPs[j]
                    a_list[i] = np.clip(a_list[i], lbound, rbound)
                    i += 1
                    j += 1
                if isinstance(module, torch.nn.Linear):
                    i += 1
    elif args.pruning_method == 'cp':
        #TODO
        current_FLOPs = flops_caculation(DNN, h, w, a_list)
        reduced_FLOPs = total_FLOPs - sum(current_FLOPs)

        if reduced_FLOPs < desired:
            duty = desired - reduced_FLOPs

            s_a = len(a_list) - sum(a_list)

            for i in range(len(a_list)):
                a_list[i] -= ((duty * (1 - a_list[i])) / s_a) / FLOPs[i]
                a_list[i] = np.clip(a_list[i], lbound, rbound)

    return a_list

'''
