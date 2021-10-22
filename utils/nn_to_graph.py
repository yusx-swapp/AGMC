import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
import numpy


def ConvSubgraph(node_cur, n_filter, edge_list, node_features, feature):
    '''
    Construct a subgraph for conv operation in a DNN
    :param node_cur:
    :param n_filter:
    :param edge_list:
    :return:
    '''
    # node_features.append(torch.randn(feature.size()))
    for i in range(n_filter):
        edge_list.append([node_cur, node_cur + (i + 1)])
        edge_list.append([node_cur + (i + 1), node_cur + n_filter + 1])
        node_features.append(feature)

    node_features.append(torch.randn(feature.size()))

    node_cur += n_filter + 1

    return edge_list, node_features, node_cur


def VGG2Graph(Motifs):
    node_cur = 0
    edge_list = []
    node_features = []
    node_features.append(torch.randn(Motifs[0].size()))

    ## conv 3*3 chanel 64
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 64, edge_list, node_features, Motifs[0])
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
    Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(), x=node_features)
    return Graph


def ResNet56Graph(Motifs):
    '''
    Construct a Graph from ResNet56
    :param initial node features:
    :return: Graph
    '''
    node_cur = 0
    edge_list = []
    node_features = []

    node_features.append(torch.randn(Motifs[0].size()))
    ## conv 7*7 chanel 64
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 64, edge_list, node_features, Motifs[0])
    node_cur_temp = node_cur

    # 3×3 max pool, stride 2
    edge_list.append([node_cur, node_cur + 1])
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
    # 32054
    Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(), x=node_features)
    return Graph


def ResNet20Graph(Motifs):
    '''
    Construct a Graph from ResNet20
    :param initial node features:
    :return: Graph
    '''
    node_cur = 0
    edge_list = []
    node_features = []

    node_features.append(torch.randn(Motifs[0].size()))
    ## conv 3*3 chanel 16
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 16, edge_list, node_features, Motifs[0])
    node_cur_temp = node_cur

    # 3×3 max pool, stride 2
    edge_list.append([node_cur, node_cur + 1])
    node_cur += 1
    node_features.append(torch.randn(Motifs[0].size()))

    for i in range(3):
        ## conv block chanel 3-3-16
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 16, edge_list, node_features, Motifs[1])
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 16, edge_list, node_features, Motifs[1])

        ##residual
        edge_list.append([node_cur_temp, node_cur])
        node_cur_temp = node_cur

    for i in range(3):
        ## conv block chanel 3-3-32
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 32, edge_list, node_features, Motifs[2])
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 32, edge_list, node_features, Motifs[2])

        ##residual
        edge_list.append([node_cur_temp, node_cur])
        node_cur_temp = node_cur

    for i in range(3):
        ## conv block chanel 3-3-64
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 64, edge_list, node_features, Motifs[3])
        edge_list, node_features, node_cur = ConvSubgraph(node_cur, 64, edge_list, node_features, Motifs[3])

        ##residual
        edge_list.append([node_cur_temp, node_cur])
        node_cur_temp = node_cur

    # average pool
    edge_list.append([node_cur, node_cur + 1])
    node_features.append(torch.randn(Motifs[0].size()))
    node_cur += 1

    node_features = [t.numpy() for t in node_features]
    node_features = torch.tensor(node_features)
    Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(), x=node_features)
    return Graph


def MobileNetv2Graph(Motifs):
    node_cur = 0
    edge_list = []
    node_features = []

    node_features.append(torch.randn(Motifs[0].size()))
    ## conv 3*3 chanel 32
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 32, edge_list, node_features, Motifs[0])
    node_cur_temp = node_cur

    # bottleneck1
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 32, edge_list, node_features, Motifs[1])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 32, edge_list, node_features, Motifs[2])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 16, edge_list, node_features, Motifs[3])

    # bottleneck2
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 96, edge_list, node_features, Motifs[4])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 96, edge_list, node_features, Motifs[5])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 24, edge_list, node_features, Motifs[6])

    node_cur_temp = node_cur

    # bottleneck3
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 144, edge_list, node_features, Motifs[7])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 144, edge_list, node_features, Motifs[8])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 24, edge_list, node_features, Motifs[9])

    edge_list.append([node_cur_temp, node_cur])

    # bottleneck4
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 144, edge_list, node_features, Motifs[10])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 144, edge_list, node_features, Motifs[11])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 32, edge_list, node_features, Motifs[12])
    node_cur_temp = node_cur

    # bottleneck5
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 192, edge_list, node_features, Motifs[13])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 192, edge_list, node_features, Motifs[14])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 32, edge_list, node_features, Motifs[15])
    edge_list.append([node_cur_temp, node_cur])
    node_cur_temp = node_cur

    # bottleneck6
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 192, edge_list, node_features, Motifs[16])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 192, edge_list, node_features, Motifs[17])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 32, edge_list, node_features, Motifs[18])
    edge_list.append([node_cur_temp, node_cur])

    # bottleneck7
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 192, edge_list, node_features, Motifs[19])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 192, edge_list, node_features, Motifs[20])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 64, edge_list, node_features, Motifs[21])
    node_cur_temp = node_cur

    # bottleneck8
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 384, edge_list, node_features, Motifs[22])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 384, edge_list, node_features, Motifs[23])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 64, edge_list, node_features, Motifs[24])
    edge_list.append([node_cur_temp, node_cur])
    node_cur_temp = node_cur

    # bottleneck9
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 384, edge_list, node_features, Motifs[25])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 384, edge_list, node_features, Motifs[26])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 64, edge_list, node_features, Motifs[27])
    edge_list.append([node_cur_temp, node_cur])
    node_cur_temp = node_cur

    # bottleneck10
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 384, edge_list, node_features, Motifs[28])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 384, edge_list, node_features, Motifs[29])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 64, edge_list, node_features, Motifs[30])
    edge_list.append([node_cur_temp, node_cur])

    # bottleneck11
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 384, edge_list, node_features, Motifs[31])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 384, edge_list, node_features, Motifs[32])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 96, edge_list, node_features, Motifs[33])
    node_cur_temp = node_cur

    # bottleneck12
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 576, edge_list, node_features, Motifs[34])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 576, edge_list, node_features, Motifs[35])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 96, edge_list, node_features, Motifs[36])
    edge_list.append([node_cur_temp, node_cur])
    node_cur_temp = node_cur

    # bottleneck13
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 576, edge_list, node_features, Motifs[37])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 576, edge_list, node_features, Motifs[38])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 96, edge_list, node_features, Motifs[39])
    edge_list.append([node_cur_temp, node_cur])

    # bottleneck14
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 576, edge_list, node_features, Motifs[40])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 576, edge_list, node_features, Motifs[41])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 160, edge_list, node_features, Motifs[42])
    node_cur_temp = node_cur

    # bottleneck15
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 960, edge_list, node_features, Motifs[43])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 960, edge_list, node_features, Motifs[44])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 160, edge_list, node_features, Motifs[45])
    edge_list.append([node_cur_temp, node_cur])
    node_cur_temp = node_cur

    # bottleneck16
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 960, edge_list, node_features, Motifs[46])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 960, edge_list, node_features, Motifs[47])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 160, edge_list, node_features, Motifs[48])
    edge_list.append([node_cur_temp, node_cur])

    # bottleneck17
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 960, edge_list, node_features, Motifs[49])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 960, edge_list, node_features, Motifs[50])
    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 320, edge_list, node_features, Motifs[51])

    edge_list, node_features, node_cur = ConvSubgraph(node_cur, 1280, edge_list, node_features, Motifs[52])

    node_features = [t.numpy() for t in node_features]
    node_features = torch.tensor(node_features)
    Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(), x=node_features)
    return Graph


def DNN2Graph(model_name, node_feature_size=50) -> object:
    # random initial node features
    initial_feature = torch.randn([70, node_feature_size])

    if model_name == 'resnet56':
        return ResNet56Graph(initial_feature)
    elif model_name == 'resnet20':
        return ResNet20Graph(initial_feature)
    elif model_name == "vgg16":
        return VGG2Graph(initial_feature)
    elif model_name == 'mobilenetv2':
        return MobileNetv2Graph(initial_feature)
    elif model_name == 'mobilenet':
        return MobileNetv2Graph(initial_feature)
    else:
        raise NotImplementedError


def Conv2Motif(module, h, w, node_features=20, nodes_id=0):
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
        n_split = kernel_h * kernel_w
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
    x = torch.randn([num_nodes, node_features])
    conv_graph = Data(x=x, edge_index=edge_list)

    return conv_graph
