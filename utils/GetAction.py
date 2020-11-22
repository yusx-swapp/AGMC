import random

import numpy as np
import torch
import torch.nn as nn
from utils.FeedbackCalculation import *
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

def act_restriction_FLOPs(args, a_list,DNN,desired,FLOPs,h,w):
    total_FLOPs = sum(FLOPs)

    if args.pruning_method == 'cpfg':
        s = []
        i=0
        for name, module in DNN.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                s.append(a_list[i])
                i += 1
            if isinstance(module, torch.nn.Linear):
                i += 1
        current_FLOPs = FlopsCaculation_(DNN, h, w, s)
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
        current_FLOPs = FlopsCaculation_(DNN, h, w, a_list)
        reduced_FLOPs = total_FLOPs - sum(current_FLOPs)

        if reduced_FLOPs < desired:
            duty = desired - reduced_FLOPs

            s_a = len(a_list) - sum(a_list)

            for i in range(len(a_list)):
                a_list[i] -= ((duty * (1 - a_list[i])) / s_a) / FLOPs[i]
                a_list[i] = np.clip(a_list[i], lbound, rbound)

    return a_list
def get_action(a_list,DNN,t,d_FLOPs,reduced,FLOPs,a_max = 0.75):
    desired = d_FLOPs
    total_FLOPs = sum(FLOPs)
    current_FLOPs=FlopsCaculation(DNN,32,32,a_list)
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
