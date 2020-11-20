import os
import argparse
from copy import deepcopy
from data import resnet
from models.Encoder import GraphEncoder
from models.Encoder_GCN import GraphEncoder_GCN
from utils.FeedbackCalculation import *
from utils.NN2Graph import *
from utils.SplitDataset import get_split_valset_ImageNet, get_split_train_valset_CIFAR
from utils.GetAction import act_restriction_FLOPs

torch.backends.cudnn.deterministic = True
from lib.agent import DDPG
from lib.Utils import get_output_folder, to_numpy, to_tensor
from models.Decoder_LSTM import Decoder_Env_LSTM as Env

def parse_args():
    parser = argparse.ArgumentParser(description='AMC search script')
    #parser.add_argument()
    parser.add_argument('--job', default='train', type=str, help='support option: train/export')
    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
    #graph encoder
    parser.add_argument('--node_feature_size', default=50, type=int, help='the initial node feature size')
    parser.add_argument('--pool_strategy', default='mean', type=str, help='pool strategy(mean/diff), defualt:mean')
    parser.add_argument('--embedding_size', default=30, type=int, help='embedding size of DNN\'s hidden layers')

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
def train(agent, env, output,G,net,n_layer,graph_encoder,val_loader,args):

    best_accuracy = 0
    #ErrorCaculation(net,val_loader,device)

    G_embedding = graph_encoder(G, G_batch).unsqueeze(0)

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    while episode < args.train_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            observation = G_embedding
            agent.reset(observation)

        # agent pick action ...
        if episode <= args.warmup:
            action = agent.random_action()
            # action = sample_from_truncated_normal_distribution(lower=0., upper=1., mu=env.preserve_ratio, sigma=0.5)
        else:
            action = agent.select_action(deepcopy(to_numpy(observation)), episode=episode)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done = env(observation.reshape(1,1,-1),to_tensor(action.reshape(1,1,-1),True).cuda(), episode_steps)

        T.append([reward, deepcopy(to_numpy(observation)), deepcopy(to_numpy(observation2)), action, done])

        # fix-length, never reach here
        # if max_episode_length and episode_steps >= max_episode_length - 1:
        #     done = True

        # [optional] save intermideate model
        if episode % int(args.train_episode / 3) == 0:
            agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = observation2

        if done:  # end of episode
            print('-' * 30)
            print("Search Episode: ",episode)

            a_list = [a for r,s,s1,a,d in T]

            if args.dataset == "imagenet":
                a_list = act_restriction_FLOPs(a_list, net, desired_FLOPs, FLOPs,224,224)
                rewards, best_accuracy = RewardCaculation_ImageNet(a_list, n_layer, net, FLOPs, best_accuracy,
                                                                   train_loader, val_loader, device,
                                                                   root=args.output)
            elif args.dataset == "cifar10":
                a_list = act_restriction_FLOPs(a_list, net, desired_FLOPs, FLOPs, 32, 32)
                rewards,best_accuracy = RewardCaculation_CIFAR(args, a_list, n_layer, net, best_accuracy, train_loader, val_loader, root = args.output)
            T[-1][0] = rewards
            final_reward = T[-1][0]
            print('final_reward: {}\n'.format(final_reward))
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


def load_model(model_name,data_root):

    if model_name == "resnet56":
        net = resnet.__dict__['resnet56']()
        net = torch.nn.DataParallel(net).cuda()
        path = os.path.join(data_root, "pretrained_models",'resnet56-4bfd9763.th')
        checkpoint = torch.load(path,map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet44":
        net = resnet.__dict__['resnet44']()
        net = torch.nn.DataParallel(net).cuda()
        path = os.path.join(data_root, "pretrained_models",'resnet44-014dd654.th')
        checkpoint = torch.load(path,map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet110":
        net = resnet.__dict__['resnet110']()
        net = torch.nn.DataParallel(net).cuda()
        path = os.path.join(data_root, "pretrained_models", 'resnet110-1d1ed7c2.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet32":
        net = resnet.__dict__['resnet32']()
        net = torch.nn.DataParallel(net).cuda()
        path = os.path.join(data_root, "pretrained_models", 'resnet32-d509ac18.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet20":
        net = resnet.__dict__['resnet20']()
        net = torch.nn.DataParallel(net).cuda()
        path = os.path.join(data_root, "pretrained_models", 'resnet20-12fca82f.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "vgg16":
        net = models.mobilenet_v2(pretrained=True).eval()
        net = torch.nn.DataParallel(net)
    return net

def get_num_hidden_layer(net,policy):
    n_layer=0
    if policy == "cp":
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                n_layer += 1
    else:
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                n_layer += 1
            if isinstance(module, nn.Linear):
                n_layer += 1
    return n_layer
def load_dataset(args):
    path = os.path.join(args.data_root, "datasets")
    if args.dataset == "imagenet":
        img_h, img_w = 224, 224
        train_loader, val_loader, n_class = get_split_valset_ImageNet("ImageNet", 128, 4, 1000, 1000,
                                                                      data_root=path,
                                                                      use_real_val=True, shuffle=True)
    elif args.dataset == "cifar10":
        img_h, img_w = 32, 32
        train_loader, val_loader, n_class = get_split_train_valset_CIFAR('cifar10', 256, 4, 5000, 1000,
                                                                         data_root=path, use_real_val=False,
                                                                        shuffle=True)

    return train_loader, val_loader, n_class

#python agmc_network_pruning.py --dataset cifar10 --model resnet56 --compression_ratio 0.5 --pruning_method cp --output ./logs
if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    device = torch.device(args.device)
    nb_states = args.embedding_size
    nb_actions = 1  # just 1 action here

    G = DNN2Graph(args.model)
    G.to(device)
    G_batch = torch.zeros([G.num_nodes,1]).long().cuda()

    print("Number of nodes:",G.num_nodes)
    print("Number of edges:", G.num_edges)


    net = load_model(args.model,args.data_root)
    net.to(device)
    cudnn.benchmark = True

    n_layer = get_num_hidden_layer(net,args.pruning_method)

    path = os.path.join(args.data_root, "datasets")
    if args.dataset == "imagenet":
        img_h, img_w = 224, 224
        train_loader, val_loader, n_class = get_split_valset_ImageNet("ImageNet", 128, 4, 1000, 1000,
                                                                      data_root=path,
                                                                      use_real_val=True, shuffle=True)
    elif args.dataset == "cifar10":
        img_h, img_w = 32, 32
        train_loader, val_loader, n_class = get_split_train_valset_CIFAR('cifar10', 256, 4, 5000, 1000,
                                                                         data_root=path, use_real_val=False,
                                                                         shuffle=True)


    FLOPs = FlopsCaculation(net, img_h, img_w)
    print("total FLOPs:", sum(FLOPs))
    desired_FLOPs = sum(FLOPs[:]) * args.compression_ratio

    if args.pool_strategy == "mean":
        graph_encoder = GraphEncoder_GCN(args.node_feature_size, args.embedding_size, args.embedding_size)
    elif args.pool_strategy == "diff":
        graph_encoder = GraphEncoder(args.node_feature_size, 15, 18, 20, args.embedding_size, args.node_feature_size, 3, 10)
    else:
        raise NotImplementedError
    graph_encoder.to(device)

    env = Env(args.embedding_size,n_layer)
    env.to(device)
    # for name, module in net.named_modules():
    params = {
        "graph_encoder": graph_encoder,
        "env": env
    }
    agent = DDPG(nb_states, nb_actions, args,**params)



    train(agent, env, args.output, G, net, n_layer, graph_encoder, val_loader, args)

