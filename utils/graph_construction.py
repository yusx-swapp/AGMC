import os
import sys
import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
import numpy as np
import logging
#sys.path.append('..')

# from data import resnet
from torchvision import models

logging.disable(30)
#


def create_edge_features(edge_types,type_features,device):
    if max(edge_types)> len(type_features):
        #random initial primitive operation like batch norm
        type_features = torch.cat((type_features,torch.randn((max(edge_types)+1 - len(type_features),type_features.shape[1]),requires_grad=True).to(device)),dim=0)

    for i in range(len(edge_types)):
        if i == 0:
            edge_features = type_features[edge_types[i]].unsqueeze(0)
        else:
            edge_features = torch.cat((edge_features,type_features[edge_types[i]].unsqueeze(0)),dim=0)
    return edge_features


def conv_sub_graph(node_cur,n_filter,edge_list,edge_type,conv_type=None,concat_type=None):
    '''
    Construct a subgraph for conv operation in a DNN
    :param node_cur:
    :param n_filter:
    :param edge_list:
    :return:
    '''
    #node_features.append(torch.randn(feature.size()))
    for i in range(n_filter):
        edge_list.append([node_cur, node_cur + (i+1)])
        edge_type.append(conv_type)

        edge_list.append([node_cur + (i+1), node_cur + n_filter +1])
        edge_type.append(concat_type)

    node_cur += n_filter + 1

    return edge_list,edge_type,node_cur

def depth_sub_graph(node_cur,n_filter,edge_list,edge_type,conv_type=None,concat_type=None,split_type=None):

    for i in range(n_filter):
        edge_list.append([node_cur, node_cur + (i+1)])
        edge_type.append(split_type)

        edge_list.append([node_cur + (i+1), node_cur + n_filter + (i+1)])
        edge_type.append(conv_type)

        edge_list.append([node_cur + n_filter+ (i+1), node_cur + 2*n_filter + 1])
        edge_type.append(concat_type)

    node_cur += 2 * n_filter + 1

    return edge_list,edge_type,node_cur

def level1_graph(in_channel,level1_type_dict,feature_size):
    def conv_motif(n_in_chanel,type_dict,stride = 1,node_cur = 0):
        '''
        Construct a subgraph for conv operation with n input channels in a DNN
        :param node_cur:
        :param n_filter:
        :param edge_list:
        :return:
        '''
        edge_list = []
        edge_type = []
        #node_features.append(torch.randn(feature.size()))
        if n_in_chanel == 0:
            n_in_chanel = 1
        for i in range(n_in_chanel):
            edge_list.append([node_cur, node_cur + (i+1)])
            edge_type.append(type_dict['split'])
            edge_list.append([node_cur + (i+1), node_cur + n_in_chanel +(i+1)])
            if stride == 1:
                edge_type.append(type_dict['conv3x3_1'])
            elif stride == 2:
                edge_type.append(type_dict['conv3x3_2'])
            edge_list.append([node_cur + n_in_chanel +(i+1),node_cur + 2*n_in_chanel +1])
            edge_type.append(type_dict['sum'])

        return np.array(edge_list),edge_type

    level_1_graphs = []
    hierarchical_graph={}

    type_features = torch.randn(len(level1_type_dict),feature_size)

    for in_c in in_channel:
        # conv 3*3 with 3 in channel
        #edge_index,edge_type = conv_pattern(3,level1_type_dict)
        edge_index,edge_type = conv_motif(in_c,level1_type_dict)

        G = Data(edge_index=torch.tensor(edge_index).long().t().contiguous(),edge_type = edge_type)
        G.x = torch.randn([G.num_nodes,feature_size]).to(device)
        G.edge_features = create_edge_features(G.edge_type,type_features,device)
        level_1_graphs.append(G.to(device))
    return level_1_graphs
def level2_graph(net_name,n_features=20):
    '''
    Construct a Graph from ResNet56
    :param initial node features:
    :return: Graph
    type_dict['concatenates']
    '''
    #TODO add edge features
    type_dict = {
        "conv3x3x3":0,
        "conv3x3x16":1,
        "conv3x3x32":2,
        "conv3x3x32_2":3,
        "conv3x3x64":4,
        "conv3x3x64_2":5,
        "concatenates":6,
        "shortCut1":7,
        "shortCut2":8,
        "bacthNorm":9,
        "linear":10,
        "ReLu":11
    }
    node_cur = 0
    edge_list = []
    edge_type = []
    node_features = []

    k = 0
    if net_name == 'vgg16':
        in_channels,out_channels = net_info(net_name)

        for i in range(len(out_channels)):

            ## conv 16*3*3 chanel 64
            edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,type_dict['conv3x3x3'],type_dict['concatenates'])
            k+=1
            #Batch Norm
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(type_dict['ReLu'])
            node_cur += 1


        # #Linear
        # edge_list.append([node_cur,node_cur+1])
        # edge_type.append(type_dict['linear'])
        Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(),edge_type =edge_type)

    elif net_name == 'mobilenet':
        in_channels,out_channels,depth_wise = net_info(net_name)

        for i in range(len(out_channels)):
            if depth_wise[i]:
                edge_list,edge_type,node_cur = depth_sub_graph(node_cur,out_channels[k],edge_list,edge_type,type_dict['conv3x3x3'],type_dict['concatenates'])
                k+=1
                #Batch Norm
                edge_list.append([node_cur,node_cur+1])
                edge_type.append(type_dict['ReLu'])
                node_cur += 1
            else:
                edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,type_dict['conv3x3x3'],type_dict['concatenates'])
                k+=1
                #Batch Norm
                edge_list.append([node_cur,node_cur+1])
                edge_type.append(type_dict['ReLu'])
                node_cur += 1
        Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(),edge_type =edge_type)
    elif net_name == 'mobilenetv2':
        in_channels,out_channels,depth_wise = net_info(net_name)

        for i in range(len(out_channels)):
            if depth_wise[i]:
                edge_list,edge_type,node_cur = depth_sub_graph(node_cur,out_channels[k],edge_list,edge_type,type_dict['conv3x3x3'],type_dict['concatenates'])
                k+=1
                #Batch Norm
                edge_list.append([node_cur,node_cur+1])
                edge_type.append(type_dict['ReLu'])
                node_cur += 1
            else:
                edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,type_dict['conv3x3x3'],type_dict['concatenates'])
                k+=1
                #Batch Norm
                edge_list.append([node_cur,node_cur+1])
                edge_type.append(type_dict['ReLu'])
                node_cur += 1

        # #Linear
        # edge_list.append([node_cur,node_cur+1])
        # edge_type.append(type_dict['linear'])
        Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(),edge_type =edge_type)
    elif 'resnet' in net_name:
        in_channels,out_channels,blocks = net_info(net_name)
        ## conv 16*3*3 chanel 64
        edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,type_dict['conv3x3x3'],type_dict['concatenates'])
        k+=1
        #Batch Norm
        edge_list.append([node_cur,node_cur+1])
        edge_type.append(type_dict['bacthNorm'])
        node_cur += 1

        node_cur_temp = node_cur
        #layer 1 with 9 ResNet blocks
        for i in range(blocks):
            #basic resnet blocks
            edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,type_dict['conv3x3x16'],type_dict['concatenates'])
            k+=1
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(type_dict['bacthNorm'])
            node_cur += 1

            edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,type_dict['conv3x3x16'],type_dict['concatenates'])
            k+=1
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(type_dict['bacthNorm'])
            node_cur += 1

            #shrot cuts
            edge_list.append([node_cur_temp,node_cur])
            edge_type.append(type_dict['shortCut1'])
            node_cur_temp = node_cur

        #layer 2 with 9 ResNet blocks
        for i in range(blocks):
            #basic resnet blocks
            if i == 0:
                edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,type_dict['conv3x3x32_2'],type_dict['concatenates'])
            else:
                edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,type_dict['conv3x3x32'],type_dict['concatenates'])
            k+=1
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(type_dict['bacthNorm'])
            node_cur += 1

            edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,type_dict['conv3x3x32'],type_dict['concatenates'])
            k+=1
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(type_dict['bacthNorm'])
            node_cur += 1

            #shrot cuts
            edge_list.append([node_cur_temp,node_cur])
            if i == 0:
                edge_type.append(type_dict['shortCut2'])
            else:
                edge_type.append(type_dict['shortCut1'])
            node_cur_temp = node_cur

        #layer 3 with 9 ResNet blocks
        for i in range(blocks):
            #basic resnet blocks
            if i == 0:
                edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,type_dict['conv3x3x64_2'],type_dict['concatenates'])
            else:
                edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,type_dict['conv3x3x64'],type_dict['concatenates'])
            k+=1
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(type_dict['bacthNorm'])
            node_cur += 1

            edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,type_dict['conv3x3x64'],type_dict['concatenates'])
            k+=1
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(type_dict['bacthNorm'])
            node_cur += 1

            #shrot cuts
            edge_list.append([node_cur_temp,node_cur])
            if i == 0:
                edge_type.append(type_dict['shortCut2'])
            else:
                edge_type.append(type_dict['shortCut1'])
            node_cur_temp = node_cur

        #Linear
        edge_list.append([node_cur,node_cur+1])
        edge_type.append(type_dict['linear'])
        # Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(),edge_type =edge_type)
        Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous())

    Graph.x = torch.randn([Graph.num_nodes, n_features])
    return Graph


def net_info(model_name):
    if 'resnet' in model_name:
        n_layer = 3
        n_blocks = 9
        in_channels = [3]
        out_channels = [16]
        channels=[[16],[32],[64]]
        if model_name == "resnet56":
            n_layer = 3
            n_blocks = 9
            in_channels = [3]
            out_channels = [16]
            channels=[[16],[32],[64]]
        elif model_name == "resnet44":
            n_layer = 3
            n_blocks = 7
            in_channels = [3]
            out_channels = [16]
            channels=[[16],[32],[64]]
        elif model_name == "resnet110":
            n_layer = 3
            n_blocks = 18
            in_channels = [3]
            out_channels = [16]
            channels=[[16],[32],[64]]

        elif model_name == "resnet32":
            n_layer = 3
            n_blocks = 5
            in_channels = [3]
            out_channels = [16]
            channels=[[16],[32],[64]]

        elif model_name == "resnet20":
            n_layer = 3
            n_blocks = 3
            in_channels = [3]
            out_channels = [16]
            channels=[[16],[32],[64]]

        for i in range(n_layer):
            if i == 0:
                in_channels.extend(channels[i]*(n_blocks*2+1))
            elif i==2:
                in_channels.extend(channels[i]*(n_blocks*2-1))
            else:
                in_channels.extend(channels[i]*n_blocks*2)

            out_channels.extend(channels[i]*n_blocks*2)
        return in_channels,out_channels,n_blocks

    elif model_name == 'mobilenetv2':
        in_channels = []
        out_channels=[]
        depth_wise=[]

        from data.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
        for name,layer in net.named_modules():
            if isinstance(layer,nn.Conv2d):
                in_channels.append(layer.in_channels)
                out_channels.append(layer.out_channels)
                if layer.groups == layer.in_channels:
                    depth_wise.append(True)
                else:
                    depth_wise.append(False)


        return in_channels,out_channels,depth_wise
    elif model_name == 'mobilenet':
        in_channels = []
        out_channels=[]
        depth_wise=[]

        from data.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
        for name,layer in net.named_modules():
            if isinstance(layer,nn.Conv2d):
                in_channels.append(layer.in_channels)
                out_channels.append(layer.out_channels)
                if layer.groups == layer.in_channels:
                    depth_wise.append(True)
                else:
                    depth_wise.append(False)


        return in_channels,out_channels,depth_wise
    elif model_name == 'shufflenet':
        n_blocks = 3
        in_channels = []
        out_channels=[]
        # in_channels.extend([32]*2)
        # in_channels.extend([64]*2)
        # in_channels.extend([128]*4)
        # in_channels.extend([256]*4)
        # in_channels.extend([512]*12)
        # in_channels.extend([1024]*2)
        in_channels.extend([128]*49)
        out_channels.extend([20]*49)
        return in_channels,out_channels,n_blocks
    elif model_name == 'shufflenetv2':
        n_blocks = 3
        in_channels = []
        out_channels=[]
        # in_channels.extend([32]*2)
        # in_channels.extend([64]*2)
        # in_channels.extend([128]*4)
        # in_channels.extend([256]*4)
        # in_channels.extend([512]*12)
        # in_channels.extend([1024]*2)
        in_channels.extend([128]*56)
        out_channels.extend([20]*56)
        return in_channels,out_channels,n_blocks

    elif model_name == 'vgg16':
        in_channels = [3]
        out_channels=[]
        in_channels.extend([64]*2)
        in_channels.extend([128]*2)
        in_channels.extend([256]*3)
        in_channels.extend([512]*5)
        out_channels.extend([64]*2)
        out_channels.extend([128]*2)
        out_channels.extend([256]*3)
        out_channels.extend([512]*6)
        return in_channels,out_channels



if __name__ == '__main__':
    #sys.path.append('..')
    net = models.mobilenet_v2(pretrained=True).eval()
    net = torch.nn.DataParallel(net)
    print(net)
    # G = level2_graph('vgg16')
    # print(G)
    # print(G.num_nodes)
    # print(G.x)
    #
    #
    # from torch_geometric.data import DataLoader
    # train_loader = DataLoader([G], batch_size=64, shuffle=True)
    #
    # dataset_iter = iter(train_loader)
    # print(DataLoader)
    # print(dataset_iter.__next__())

# train_loader

    #G_batch = torch.zeros([G.num_nodes,0]).long().cuda()







