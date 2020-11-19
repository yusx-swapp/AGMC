
import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
import numpy

def ConvSubgraph(node_cur,n_filter,edge_list,node_features,feature):
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
        edge_list.append([node_cur + (i+1), node_cur + n_filter +1])
        node_features.append(feature)

    node_features.append(torch.randn(feature.size()))

    node_cur += n_filter + 1

    return edge_list,node_features,node_cur

def VGG2Graph(Motifs):
    node_cur = 0
    edge_list = []
    node_features = []
    node_features.append(torch.randn(Motifs[0].size()))

    ## conv 3*3 chanel 64
    edge_list,node_features,node_cur = ConvSubgraph(node_cur,64, edge_list,node_features,Motifs[0])
    ## conv 3*3 chanel 64
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 64, edge_list, node_features, Motifs[0])

    # max pool
    edge_list.append([node_cur, node_cur + 1])
    node_cur += 1
    node_features.append(torch.randn(Motifs[0].size()))

    ## conv 3*3 chanel 128
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 128, edge_list, node_features, Motifs[1])
    ## conv 3*3 chanel 128
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 128, edge_list, node_features, Motifs[1])

    # max pool
    edge_list.append([node_cur, node_cur + 1])
    node_cur += 1
    node_features.append(torch.randn(Motifs[0].size()))

    ## conv 3*3 chanel 256
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 256, edge_list, node_features, Motifs[2])
    ## conv 3*3 chanel 256
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 256, edge_list, node_features, Motifs[2])
    ## conv 3*3 chanel 256
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 256, edge_list, node_features, Motifs[2])

    # max pool
    edge_list.append([node_cur, node_cur + 1])
    node_cur += 1
    node_features.append(torch.randn(Motifs[0].size()))

    ## conv 3*3 chanel 512
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 512, edge_list, node_features, Motifs[3])
    ## conv 3*3 chanel 512
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 512, edge_list, node_features, Motifs[3])
    ## conv 3*3 chanel 512
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 512, edge_list, node_features, Motifs[3])

    # max pool
    edge_list.append([node_cur, node_cur + 1])
    node_cur += 1
    node_features.append(torch.randn(Motifs[0].size()))

    ## conv 3*3 chanel 512
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 512, edge_list, node_features, Motifs[3])
    ## conv 3*3 chanel 512
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 512, edge_list, node_features, Motifs[3])
    ## conv 3*3 chanel 512
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 512, edge_list, node_features, Motifs[3])

    # max pool
    edge_list.append([node_cur, node_cur + 1])
    node_cur += 1
    node_features.append(torch.randn(Motifs[0].size()))

    node_features = [t.numpy() for t in node_features]
    node_features = torch.tensor(node_features)
    Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(),x=node_features)
    return Graph

def ResNet56Graph(Motifs):
    '''
    Construct a Graph from ResNet-nlayers
    :param nlayer:
    :return: Graph
    '''
    node_cur = 0
    edge_list = []
    node_features = []

    node_features.append(torch.randn(Motifs[0].size()))
    ## conv 7*7 chanel 64
    edge_list,node_features,node_cur = ConvSubgraph(node_cur,64, edge_list,node_features,Motifs[0])
    node_cur_temp = node_cur

    #3×3 max pool, stride 2
    edge_list.append([node_cur,node_cur+1])
    node_cur += 1
    node_features.append(torch.randn(Motifs[0].size()))

    for i in range(3):
        ## conv block chanel 64-64-256
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 64, edge_list, node_features, Motifs[1])
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 64, edge_list, node_features, Motifs[2])
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 256, edge_list, node_features, Motifs[1])

        ##residual
        edge_list.append([node_cur_temp, node_cur])
        node_cur_temp = node_cur

    for i in range(4):
        ## conv block chanel 128-128-512
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 128, edge_list, node_features, Motifs[1])
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 128, edge_list, node_features, Motifs[2])
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 512, edge_list, node_features, Motifs[1])

        ##residual
        edge_list.append([node_cur_temp, node_cur])
        node_cur_temp = node_cur

    for i in range(6):
        ## conv block chanel 256-256-1024
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 256, edge_list, node_features, Motifs[1])
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 256, edge_list, node_features, Motifs[2])
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 1024, edge_list, node_features, Motifs[1])

        ##residual
        edge_list.append([node_cur_temp, node_cur])
        node_cur_temp = node_cur

    for i in range(6):
        ## conv block chanel 512-512-2048
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 512, edge_list, node_features, Motifs[1])
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 512, edge_list, node_features, Motifs[2])
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 2048, edge_list, node_features, Motifs[1])

        ##residual
        edge_list.append([node_cur_temp, node_cur])
        node_cur_temp = node_cur

    # average pool, stride 2
    edge_list.append([node_cur, node_cur + 1])
    node_features.append(torch.randn(Motifs[0].size()))
    node_cur += 1

    node_features = [t.numpy() for t in node_features]
    node_features = torch.tensor(node_features)
    #32054
    Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(),x=node_features)
    return Graph

def DNN2Graph(model_name):
    # random initial node features
    initial_feature = torch.randn([30, 50])

    if model_name=='resnet56':
        return ResNet56Graph(initial_feature)
    elif model_name == "vgg16":
        return VGG2Graph(initial_feature)
    else:
        return NotImplementedError
def Conv2Motif(module,h,w,node_features=20, nodes_id=0):
    '''
        Construct a Motif for conv operation

    :param module: DNN conv2d layer
    :param h: input image height
    :param w:
    :param node_features: number of node_features
    :param nodes_id: node id
    :return:
    '''
    edge_list = []
    stride_h, stride_w = module.stride
    kernel_h, kernel_w = module.kernel_size
    padding = module.padding
    if padding != (0, 0):
        n_split = kernel_h*kernel_w
    else:
        n_split = int(((w - kernel_w) / stride_w + 1) * ((h - kernel_h) / stride_h + 1))
    num_nodes = n_split * 2 + 3

    for i in range(n_split):
        edge_list.append([nodes_id, nodes_id + i + 2])
        edge_list.append([nodes_id + 1, nodes_id + n_split + 2 + i])
        edge_list.append([nodes_id + i + 2, nodes_id + n_split + 2 + i])
        edge_list.append([nodes_id + n_split + 2 + i, nodes_id + num_nodes - 1])
    nodes_id = nodes_id + num_nodes
    edge_list = torch.tensor(edge_list).t().contiguous()
    x = torch.randn([num_nodes,node_features])
    conv_graph = Data(x=x, edge_index=edge_list)

    return conv_graph
if __name__ == '__main__':
    # DNN = CNN()
    #
    # DNN.load_state_dict(torch.load("../DNN/pretrain/cnn.pkl"))
    # for name, module in DNN.named_modules():
    #     if isinstance(module, torch.nn.Conv2d):
    #         conv_graph = Conv2Motif(module, 18, 28, 20, 0)
    #         print(conv_graph.num_nodes)
    #         print(conv_graph.num_edges)
    Motifs = torch.randn([4,10])

    G = VGG2Graph(Motifs)
