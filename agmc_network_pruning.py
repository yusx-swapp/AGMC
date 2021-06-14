import os
import argparse
from copy import deepcopy
from torchvision import models
import torch.backends.cudnn as cudnn
from data import resnet
from models.Encoder import GraphEncoder
from models.Encoder_GCN import GraphEncoder_GCN
from utils.feedback_calculation import *
from utils.NN2Graph import *
from utils.SplitDataset import get_split_valset_ImageNet, get_split_train_valset_CIFAR, get_split_dataset
from utils.get_action import act_restriction_flops
import torch
from torch_geometric.data import DataLoader

from utils.graph_construction import level2_graph

torch.backends.cudnn.deterministic = True
from lib.agent import DDPG
from lib.Utils import get_output_folder, to_numpy, to_tensor
from models.Decoder_LSTM import Decoder_Env_LSTM as Env

_r_ = []

def parse_args():
    parser = argparse.ArgumentParser(description='AGMC search script')
    #parser.add_argument()
    parser.add_argument('--job', default='train', type=str, help='support option: train/export')
    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
    #graph encoder
    parser.add_argument('--node_feature_size', default=50, type=int, help='the initial node feature size')
    parser.add_argument('--pool_strategy', default='mean', type=str, help='pool strategy(mean/diff), defualt:mean')
    parser.add_argument('--embedding_size', default=30, type=int, help='embedding size of DNN\'s hidden layers')

    # datasets and model
    parser.add_argument('--model', default='mobilenet', type=str, help='model to prune')
    parser.add_argument('--dataset', default='ILSVRC', type=str, help='dataset to use (cifar/ILSVRC)')
    parser.add_argument('--data_root', default='data', type=str, help='dataset path')
    parser.add_argument('--preserve_ratio', default=0.5, type=float, help='preserve ratio of the model')
    parser.add_argument('--lbound', default=0.2, type=float, help='minimum preserve ratio')
    parser.add_argument('--rbound', default=1., type=float, help='maximum preserve ratio')
    parser.add_argument('--reward', default='acc_reward', type=str, help='Setting the reward')
    parser.add_argument('--acc_metric', default='acc5', type=str, help='use acc1 or acc5')
    parser.add_argument('--use_real_val', dest='use_real_val', action='store_true')
    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')
    parser.add_argument('--train_size', default=10000, type=int, help='(Fine tuning) training size of the datasets.')
    parser.add_argument('--val_size', default=10000, type=int, help='(Reward caculation) test size of the datasets.')
    parser.add_argument('--f_epochs', default=20, type=int, help='Fast fine-tuning epochs.')

    # pruning

    parser.add_argument('--compression_ratio', default=0.5, type=float,
                        help='compression_ratio')
    parser.add_argument('--pruning_method', default='cp', type=str,
                        help='method to prune (fg/cp/cpfg for fine-grained and channel pruning)')
    parser.add_argument('--n_calibration_batches', default=60, type=int,
                        help='n_calibration_batches')
    parser.add_argument('--n_points_per_layer', default=10, type=int,
                        help='n_points_per_layer')
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
    parser.add_argument('--train_episode', default=100, type=int, help='train iters each timestep')
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
def train(agent, output,G,net,val_loader,args):


    best_accuracy = 0
    # G_embedding = graph_encoder(G, G_batch).unsqueeze(0)
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    while episode < args.train_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            observation = G
            agent.reset(observation)

        # agent pick action ...
        if episode <= args.warmup:
            action = agent.random_action()
            # action = sample_from_truncated_normal_distribution(lower=0., upper=1., mu=env.preserve_ratio, sigma=0.5)
        else:
            action = agent.select_action(deepcopy(observation), episode=episode)

        # env response with next_observation, reward, terminate_info
        #observation2, reward, done = env(observation.reshape(1,1,-1),to_tensor(action.reshape(1,1,-1),True).cuda(), episode_steps)
        observation2,reward, done = observation, 0,True
        T.append([reward, deepcopy(g), deepcopy(g), action, done])

        if episode % int(args.train_episode / 3) == 0:
            agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = observation2

        if done:  # end of episode
            print('-' * 40)
            print("Search Episode: ",episode)

            a_list = T[-1][-2]
            # print(a_list)
            if args.pruning_method == 'cp':
                a_list = 1 - np.array(a_list)
                a_list = act_restriction_flops(args, a_list, net)

            rewards, best_accuracy = reward_caculation(args, a_list, net, best_accuracy,
                                                      val_loader,train_loader, root=args.output)

            T[-1][0] = 100+rewards
            final_reward = T[-1][0]
            print('final_reward: {}\n'.format(final_reward))
            _r_.append(final_reward-100)
            print('best_accuracy: {}\n'.format(best_accuracy))

        # agent observe and update policy
            # print(a_list)

            for r_t, s_t, s_t1, a_t, done in T:
                agent.observe(final_reward, s_t, s_t1, a_t, done)

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
        net = models.vgg16(pretrained=True).eval()
        net = torch.nn.DataParallel(net)
    elif model_name == "mobilenetv2":
        net = models.mobilenet_v2(pretrained=True).eval()
        net = torch.nn.DataParallel(net)

    elif model_name == "mobilenet":
        from data.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
        # net = torch.nn.DataParallel(net)
        sd = torch.load("data/checkpoints/mobilenet_imagenet.pth.tar")
        if 'state_dict' in sd:  # a checkpoint but not a state_dict
            sd = sd['state_dict']
        net.load_state_dict(sd)
        # net = net.cuda()
        net = torch.nn.DataParallel(net)
    else:
        raise KeyError
    return net

def get_prunable_idx(net,args):
    index = []
    if args.model == 'resnet56':
        for i, module in enumerate(net.modules()):
        #for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                index.append(i)


def get_num_hidden_layer(net,args):
    if args.pruning_method == "cp":
        prunable_layer_types = [torch.nn.Conv2d]
    else:
        prunable_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
    n_layer=0

    if args.model== "mobilenet":
        n_layer = len(list(net.module.features.named_children()))+1

    elif args.model == "mobilenetv2":
        net = models.mobilenet_v2()
        for name,layer in net.named_modules():
            if isinstance(layer,nn.Conv2d) :
                if layer.groups == layer.in_channels:
                    n_layer += 1




    elif "resnet" in args.model:

        n_layer+=len(list(net.module.layer1.named_children()))
        n_layer+=len(list(net.module.layer2.named_children()))
        n_layer+=len(list(net.module.layer3.named_children()))
        n_layer+=1
    elif args.model == 'vgg16':
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                n_layer+=1
    else:
        raise NotImplementedError
    return n_layer

#python agmc_network_pruning.py --dataset cifar10 --model resnet56 --compression_ratio 0.9 --pruning_method cp --train_episode 150 --output ./logs1
#python agmc_network_pruning.py --dataset cifar10 --model resnet20 --compression_ratio 0.1 --pruning_method cp --train_episode 180 --output ./logs
#python agmc_network_pruning.py --dataset cifar10 --model resnet56 --compression_ratio 0.5 --pruning_method fg --train_episode 10 --train_size 5000 --val_size 1000 --output ./logs
#python agmc_network_pruning.py --dataset ILSVRC --model mobilenet --compression_ratio 0.5 --pruning_method cp --data_root data/datasets  --train_size 10000 --val_size 10000 --output ./logs
#python agmc_network_pruning.py --dataset ILSVRC --model vgg16 --compression_ratio 0.8 --pruning_method cp --data_root data/datasets --train_size 5000 --val_size 2000 --output ./logs
#python agmc_network_pruning.py --dataset ILSVRC --model mobilenetv2 --compression_ratio 0.5 --pruning_method cp --data_root data/datasets  --train_size 10000 --val_size 10000 --output ./logs

if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    device = torch.device(args.device)
    nb_states = args.embedding_size
    nb_actions = 1  # just 1 action here

    # G = DNN2Graph(args.model,args.node_feature_size)
    g = level2_graph(args.model,args.node_feature_size)
    g.to(device)
    # G_batch = torch.zeros(g.num_nodes).long().cuda()
    # print(G_batch)
    # g.batch = G_batch
    G = DataLoader([g], batch_size=1, shuffle=True)
    for graph in G:
        G = graph
        break
    print("Number of nodes:",g.num_nodes)
    print("Number of edges:", g.num_edges)


    net = load_model(args.model,args.data_root)
    net.to(device)
    cudnn.benchmark = True

    n_layer = get_num_hidden_layer(net,args)
    print('number of hidden layers: ',n_layer)
    nb_actions = n_layer  # just 1 action here


    if args.dataset == "ILSVRC":
        path = args.data_root

        train_loader, val_loader, n_class = get_split_dataset("imagenet", 512, 4, args.train_size, args.val_size,
                                                                      data_root=path,
                                                                      use_real_val=False, shuffle=True)
    elif args.dataset == "cifar10":
        path = os.path.join(args.data_root, "datasets")

        train_loader, val_loader, n_class = get_split_train_valset_CIFAR('cifar10', 256, 4, args.train_size, args.val_size,
                                                                         data_root=path, use_real_val=False,
                                                                         shuffle=True)




    # if args.pool_strategy == "mean":
    #     graph_encoder = GraphEncoder_GCN(args.node_feature_size, args.embedding_size, args.embedding_size)
    # elif args.pool_strategy == "diff":
    #     graph_encoder = GraphEncoder(args.node_feature_size, 15, 18, 20, args.embedding_size, args.node_feature_size, 3, 10)
    # else:
    #     raise NotImplementedError
    # graph_encoder = GraphEncoder_GCN(args.node_feature_size, args.embedding_size, args.embedding_size)
    # graph_encoder.to(device)

    # env = Env(args.embedding_size,n_layer)
    # env.to(device)
    #
    # params = {
    #     "graph_encoder": graph_encoder,
    #     "env": env
    # }

    agent = DDPG(args.node_feature_size,nb_states, nb_actions, args)
    agent.cuda()



    train(agent, args.output, G, net, val_loader, args)
    print(_r_)
    print(_r_[25:])


#python agmc_network_pruning.py --dataset cifar10 --model resnet56 --compression_ratio 0.1 --pruning_method cp --train_episode 3000 --output ./logs

'''
dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    model, val_acc_history, best_acc= \
        train_model(net, dataloaders, criterion, optimizer_ft,device, num_epochs=20, is_inception=False)
    validate(val_loader, device, model, criterion)

    net = load_model('resnet20',args.data_root)
    n_layer = get_num_hidden_layer(net,args)
    G_1 = DNN2Graph('resnet20',args.node_feature_size)
    G_1.to(device)
    G_batch = torch.zeros([G_1.num_nodes,1]).long().cuda()

    G_embedding = graph_encoder(G_1, G_batch).unsqueeze(0)

    observation = G_embedding
    a_list = []
    for i in range(n_layer):
        action = agent.select_action(deepcopy(to_numpy(observation)), episode=args.train_episode)

        observation2, reward, done = env(observation.reshape(1,1,-1),to_tensor(action.reshape(1,1,-1),True).cuda(), i)
        #T.append([reward, deepcopy(to_numpy(observation)), deepcopy(to_numpy(observation2)), action, done])
        a_list.append(action)

    if args.pruning_method == 'cp' or args.pruning_method == 'cpfg':
        a_list = act_restriction_flops(args,  a_list, net)

    print("Final Reward:")
    rewards, best_accuracy = reward_caculation(args, a_list, net, 0,
                                               val_loader, root=args.output)
    model, val_acc_history, best_acc= \
        train_model(net, dataloaders, criterion, optimizer_ft,device, num_epochs=150, is_inception=False)

    validate(val_loader, device, model, criterion)
'''

#amp

#mobilenet 0.45435380477224707


#python agmc_network_pruning.py --dataset ILSVRC --model mobilenetv2 --compression_ratio 0.2 --pruning_method cp --data_root data/datasets  --train_size 1000 --val_size 1000 --output ./logs
