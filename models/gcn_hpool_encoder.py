# coding=utf-8

import torch
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.utils as Utils
from models.gcn_hpool import GcnHpoolSubmodel

from torch_geometric.nn import DenseGCNConv


class GcnHpoolEncoder(Module):

  def __init__(self, in_feature, hidden_feature, out_feature, h_hidden_feature,
               h_out_feature,in_node, hidden_node, out_node):
    super(GcnHpoolEncoder, self).__init__()

    #self._hparams = hparams_lib.copy_hparams(hparams)

    self.in_feature=in_feature
    self.hidden_feature=hidden_feature
    self.out_feature=out_feature
    self.h_hidden_feature=h_hidden_feature
    self.h_out_feature=h_out_feature
    self.in_node=in_node
    self.hidden_node=hidden_node
    self.out_node=out_node
    #self._device = torch.device(self._hparams.device)

    self.build_graph()
  # def reset_parameters(self):
  #   for m in self.modules():
  #     if isinstance(m, gcn_layer.GraphConvolution):
  #       m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
  #       if m.bias is not None:
  #         m.bias.data = torch.nn.init.constant_(m.bias.data, 0.0)

  def build_graph(self):

    # entry GCN
    self.entry_conv_first = DenseGCNConv(
      in_channels=self.in_feature,
      out_channels=self.hidden_feature,
    )
    self.entry_conv_block = DenseGCNConv(
      in_channels=self.hidden_feature,
      out_channels=self.hidden_feature,
    )
    self.entry_conv_last = DenseGCNConv(
      in_channels=self.hidden_feature,
      out_channels=self.out_feature,
    )

    self.gcn_hpool_layer = GcnHpoolSubmodel(
      self.out_feature +self.hidden_feature* 2, self.h_hidden_feature, self.h_out_feature,
      self.in_node, self.hidden_node, self.out_node
      #self._hparams
    )

    self.pred_model = torch.nn.Sequential(
      torch.nn.Linear(2 *self.hidden_feature+2*self.h_hidden_feature+self.h_out_feature+self.out_feature, self.h_hidden_feature),
      #torch.nn.Linear( self.out_feature, self.h_hidden_feature),
      torch.nn.ReLU(),
      torch.nn.Linear(self.h_hidden_feature, self.h_out_feature)
    )

  def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):
    out_all = []

    layer_out_1 = F.relu(conv_first(x, adj))
    layer_out_1 = self.apply_bn(layer_out_1)
    out_all.append(layer_out_1)

    layer_out_2 = F.relu(conv_block(layer_out_1, adj))
    layer_out_2 = self.apply_bn(layer_out_2)
    out_all.append(layer_out_2)

    layer_out_3 = conv_last(layer_out_2, adj)
    out_all.append(layer_out_3)

    out_all = torch.cat(out_all, dim=2)
    if embedding_mask is not None:
      out_all = out_all * embedding_mask

    return out_all
  def forward(self, Graph):
    node_feature,edge_index = Graph.x, Graph.edge_index
    adjacency_mat = Utils.to_dense_adj(edge_index)

    # input mask
    max_num_nodes = adjacency_mat.size()[1]
    # embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
    embedding_mask = None

    #node_feature=node_feature.reshape(1,node_feature.shape[0],node_feature.shape[1])
    # entry embedding gcn
    # print("node_feature",node_feature.shape)
    # print("adjacency_mat",adjacency_mat.shape)
    embedding_tensor_1 = self.gcn_forward(
      node_feature, adjacency_mat,
      self.entry_conv_first, self.entry_conv_block, self.entry_conv_last,
      embedding_mask
    )
    output_1, _ = torch.max(embedding_tensor_1, dim=1)

    # hpool layer
    output_2, _, _, _ = self.gcn_hpool_layer(
      embedding_tensor_1, node_feature, edge_index,adjacency_mat, embedding_mask
    )
    output = torch.cat([output_1, output_2], dim=1)
    ypred = self.pred_model(output)

    return ypred



  def apply_bn(self, x):
      ''' Batch normalization of 3D tensor x
      '''
      #bn_module = torch.nn.BatchNorm1d(x.size()[1])
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      bn_module = torch.nn.BatchNorm1d(x.size()[1]).to(device)
      return bn_module(x)

  def construct_mask(self, max_nodes, batch_num_nodes):
      ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
      corresponding column are 1's, and the rest are 0's (to be masked out).
      Dimension of mask: [batch_size x max_nodes x 1]
      '''
      # masks
      packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
      batch_size = len(batch_num_nodes)
      out_tensor = torch.zeros(batch_size, max_nodes)
      for i, mask in enumerate(packed_masks):
          out_tensor[i, :batch_num_nodes[i]] = mask
      return out_tensor.unsqueeze(2).to(self._device)

if __name__ == '__main__':
  model = GcnHpoolEncoder(50, 15, 18, 20, 30,50, 3, 10)
  #in_feature, hidden_feature, out_feature, h_hidden_feature, h_out_feature,in_node, hidden_node, out_node
  edge_index = torch.tensor([[0, 1],
                             [1, 0],
                             [1, 2],
                             [2, 1],
                             [2,3],
                             [2,4],
                             [2,5]], dtype=torch.long)
  x = torch.randn([6,50])
  data = Data(x=x, edge_index=edge_index.t().contiguous())
  out = model(data)
  print(out.shape)