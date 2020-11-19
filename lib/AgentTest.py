import sys


sys.path.append("..")
import os
import numpy as np
import argparse
from copy import deepcopy
import torch

from DNN import resnet
from models.Encoder_GCN import GraphEncoder_GCN
from utils.FeedbackCalculation import *
from utils.NN2Graph import ResNet2Graph
from utils.SplitDataset import get_split_valset_CIFAR
from utils.GetAction import act_restriction_FLOPs

torch.backends.cudnn.deterministic = True
from lib.agent import DDPG
from lib.Utils import get_output_folder, to_numpy, to_tensor
from models.Decoder_LSTM import Decoder_Env_LSTM as Env



def parse_args():
    parser = argparse.ArgumentParser(description='AMC search script')

    parser.add_argument('--job', default='train', type=str, help='support option: train/export')
    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
    # env
    parser.add_argument('--model', default='mobilenet', type=str, help='model to prune')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to use (cifar/imagenet)')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    parser.add_argument('--preserve_ratio', default=0.5, type=float, help='preserve ratio of the model')
    parser.add_argument('--lbound', default=0.2, type=float, help='minimum preserve ratio')
    parser.add_argument('--rbound', default=1., type=float, help='maximum preserve ratio')
    parser.add_argument('--reward', default='acc_reward', type=str, help='Setting the reward')
    parser.add_argument('--acc_metric', default='acc5', type=str, help='use acc1 or acc5')
    parser.add_argument('--use_real_val', dest='use_real_val', action='store_true')
    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')
    # parser.add_argument('--pruning_method', default='cp', type=str,
    #                     help='method to prune (fg/cp for fine-grained and channel pruning)')
    # only for channel pruning
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
    parser.add_argument('--warmup', default=50, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=1., type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=100, type=int, help='memory size for each layer')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    # noise (truncated normal distribution)
    parser.add_argument('--init_delta', default=0.5, type=float,
                        help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.95, type=float,
                        help='delta decay during exploration')
    # training
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
def train(num_episode, agent, env, output,G,net,n_layer,graph_encoder,val_loader):

    best_accuracy = 0
    ErrorCaculation_CIFAR(net,val_loader,device)

    G_embedding = graph_encoder(G, G_batch).unsqueeze(0)

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    while episode < num_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            #observation = deepcopy(env.reset())
            observation = G_embedding
            agent.reset(observation)

        # agent pick action ...
        if episode <= args.warmup:
            action = agent.random_action()
            # action = sample_from_truncated_normal_distribution(lower=0., upper=1., mu=env.preserve_ratio, sigma=0.5)
        else:
            action = agent.select_action(deepcopy(to_numpy(observation)), episode=episode)
        #print(observation.shape,action.shape)
        # env response with next_observation, reward, terminate_info
        observation2, reward, done = env(observation.reshape(1,1,-1),to_tensor(action.reshape(1,1,-1),True).cuda(), episode_steps)
        #observation2, reward, done, info = env.step(action)
        #observation2 = to_numpy(observation2)
        #observation2 = deepcopy(observation2)
        T.append([reward, deepcopy(to_numpy(observation)), deepcopy(to_numpy(observation2)), action, done])

        # fix-length, never reach here
        # if max_episode_length and episode_steps >= max_episode_length - 1:
        #     done = True

        # [optional] save intermideate model
        if episode % int(num_episode / 3) == 0:
            agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = observation2

        if done:  # end of episode
            a_list = [a for r,s,s1,a,d in T]
            a_list = act_restriction_FLOPs(a_list, net, desired_FLOPs, FLOPs)
            #print(a_list)
            rewards,best_accuracy = RewardCaculation(a_list, n_layer, net, FLOPs, best_accuracy,val_loader,device,root='../DNN/datasets')

            T[-1][0] = rewards
            final_reward = T[-1][0]
            print('final_reward: {}'.format(final_reward))
            # agent observe and update policy
            i=0
            for r_t, s_t, s_t1, a_t, done in T:
                agent.observe(final_reward, s_t, s_t1, a_list[i], done)
                i+=1
                if episode > args.warmup:
                    agent.update_policy()

            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    nb_states = 25
    nb_actions = 1  # just 1 action here

    #device = torch.device("cpu")
    Motifs_embedding = torch.randn([3, 50])
    G = ResNet2Graph(Motifs_embedding)
    G.to(device)

    print("Number of nodes:",G.num_nodes)
    print("Number of edges:", G.num_edges)
    G_batch = torch.zeros([G.num_nodes,1]).long().cuda()

    # load pretrained model 55 layers
    net = resnet.__dict__['resnet56']()
    net = torch.nn.DataParallel(net).cuda()
    net.to(device)

    checkpoint = torch.load('../DNN/pretrained_models/resnet56-4bfd9763.th',map_location=device)
    net.load_state_dict(checkpoint['state_dict'])

    n_layer = 55
    FLOPs = FlopsCaculation(net, 32, 32)
    print("total FLOPs:", sum(FLOPs))
    # 50% Flops reduction
    desired_FLOPs = sum(FLOPs[:]) * 0.5

    graph_encoder = GraphEncoder_GCN(50, 35, 25)
    # graph_encoder = torch.nn.DataParallel(graph_encoder)
    graph_encoder.to(device)

    env = Env(25,55)
    #env = torch.nn.DataParallel(env)
    env.to(device)

    agent = DDPG(nb_states, nb_actions, args)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    #
    # val_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR10(root="../DNN/datasets", train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=128, shuffle=False,
    #     num_workers=4, pin_memory=True)
    train_loader, val_loader,_ = get_split_valset_CIFAR('cifar10', 256, 4, 5000, data_root='../DNN/datasets',use_real_val=True, shuffle=True)
    train(args.train_episode, agent, env , args.output,G,net,n_layer,graph_encoder,val_loader)
