import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn import DenseGCNConv
class MotifEmbedding(nn.Module):
    def __init__(self,in_feature, hidden_feature, out_feature, h_hidden_feature,
               h_out_feature,in_node, hidden_node, out_node):
        '''

        :param n_in:
        :param n_hidden:
        :param n_out:
        '''
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.out_feature = out_feature
        self.h_hidden_feature=h_hidden_feature
        self.h_out_feature = h_out_feature
        self.in_node = in_node
        self.hidden_node = hidden_node
        self.out_node = out_node

        self.conv1 = GCNConv(self.n_in, self.n_hidden)
        self.conv2 = GCNConv(self.n_hidden, self.n_out)

        self.pool_linear = torch.nn.Linear(hidden_node * 2 + out_node, out_node)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last,conv_pool_first,conv_pool_block,conv_pool_last):

        layer_out_1 = F.relu(conv_first(x, adj))
        layer_out_1 = self.apply_bn(layer_out_1)
        pooling_tensor_1 = F.relu(conv_pool_first(x, adj))
        pooling_tensor_1 = F.softmax(self.pool_linear_1(pooling_tensor_1), dim=-1)
        x_pool_1, adj_pool_1, _, _ = dense_diff_pool(layer_out_1, adj, pooling_tensor_1)


        layer_out_2 = F.relu(conv_block(x_pool_1, adj_pool_1))
        layer_out_2 = self.apply_bn(layer_out_2)
        pooling_tensor_2 = F.relu(conv_pool_block(x_pool_1, adj_pool_1))
        pooling_tensor_2 = F.softmax(self.pool_linear_2(pooling_tensor_2), dim=-1)
        x_pool_2, adj_pool_2, _, _ = dense_diff_pool(layer_out_2, adj, pooling_tensor_2)

        layer_out_3 = F.relu(conv_last(x_pool_2, adj_pool_2))
        layer_out_3 = self.apply_bn(layer_out_3)
        pooling_tensor_3 = F.relu(conv_pool_last(x_pool_2, adj_pool_2))
        pooling_tensor_3 = F.softmax(self.pool_linear_3(pooling_tensor_3), dim=-1)
        x_pool_3, adj_pool_3, _, _ = dense_diff_pool(layer_out_3, adj, pooling_tensor_3)





        return x_pool_3

    def forward(self, Motifs):
        '''
        :param Motifs:
        :return:
        '''

        Motif_Embeddings = []
        for Motif in Motifs:
            x, edge_index = Motif.x, Motif.edge_index
            x = self.conv1(x,edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            Motif_Embeddings.append(x)
        return Motif_Embeddings

#
# class ComputeGraphEmbedding(nn.Module):
#     def __init__(self,n_in, n_hidden, n_out):
#         self.motif_encoder = MotifEmbedding(n_in, n_hidden, n_out)

