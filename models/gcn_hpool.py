# coding=utf-8

import torch
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn import DenseGCNConv


class GcnHpoolSubmodel(Module):
  def __init__(self, in_feature, hidden_feature, out_feature, in_node, hidden_node, out_node):
    super(GcnHpoolSubmodel, self).__init__()

    #self._hparams = hparams_lib.copy_hparams(hparams)
    #self.build_graph(in_feature, hidden_feature, out_feature, in_node, hidden_node, out_node)
    self.reset_parameters()

    #self._device = torch.device(self._hparams.device)
    self.pool_tensor = None

    # # embedding blocks
    #
    # self.embed_conv_first = GCNConv(
    #     in_channels=in_feature,
    #     out_channels=hidden_feature,
    # )
    # self.embed_conv_block = GCNConv(
    #     in_channels=hidden_feature,
    #     out_channels=hidden_feature,
    # )
    # self.embed_conv_last = GCNConv(
    #     in_channels=hidden_feature,
    #     out_channels=out_feature,
    # )
    # embedding blocks

    self.embed_conv_first = DenseGCNConv(
        in_channels=in_feature,
        out_channels=hidden_feature,
    )
    self.embed_conv_block = DenseGCNConv(
        in_channels=hidden_feature,
        out_channels=hidden_feature,
    )
    self.embed_conv_last = DenseGCNConv(
        in_channels=hidden_feature,
        out_channels=out_feature,
    )

    # pooling blocks

    self.pool_conv_first = DenseGCNConv(
        in_channels=in_node,
        out_channels=hidden_node,
    )

    self.pool_conv_block = DenseGCNConv(
        in_channels=hidden_node,
        out_channels=hidden_node,
    )

    self.pool_conv_last = DenseGCNConv(
        in_channels=hidden_node,
        out_channels=out_node,
    )

    self.pool_linear = torch.nn.Linear(hidden_node * 2 + out_node, out_node)
  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, DenseGCNConv):
        m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
        if m.bias is not None:
          m.bias.data = torch.nn.init.constant_(m.bias.data, 0.0)

  def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):
    out_all = []

    layer_out_1 = F.relu(conv_first(x, adj))
    layer_out_1 = self.apply_bn(layer_out_1)
    out_all.append(layer_out_1)
    #print('layer_out_1',layer_out_1.size())

    layer_out_2 = F.relu(conv_block(layer_out_1, adj))
    layer_out_2 = self.apply_bn(layer_out_2)
    out_all.append(layer_out_2)
    #print('layer_out_2',layer_out_2.size())

    layer_out_3 = conv_last(layer_out_2, adj)
    #print('layer_out_3',layer_out_3.size())

    out_all.append(layer_out_3)
    out_all = torch.cat(out_all, dim=2)
    if embedding_mask is not None:
      out_all = out_all * embedding_mask

    return out_all


  def forward(self, embedding_tensor, pool_x_tensor, edge_index,adj, embedding_mask=None):

    pooling_tensor = self.gcn_forward(
      pool_x_tensor, adj,
      self.pool_conv_first, self.pool_conv_block, self.pool_conv_last,
      embedding_mask
    )
    pooling_tensor = F.softmax(self.pool_linear(pooling_tensor), dim=-1)

    if embedding_mask is not None:
      pooling_tensor = pooling_tensor * embedding_mask

    x_pool, adj_pool, _, _ = dense_diff_pool(embedding_tensor, adj, pooling_tensor)

    embedding_tensor = self.gcn_forward(
      x_pool, adj_pool,
      self.embed_conv_first, self.embed_conv_block, self.embed_conv_last,
    )
    output, _ = torch.max(embedding_tensor, dim=1)

    self.pool_tensor = pooling_tensor
    return output, adj_pool, x_pool, embedding_tensor



  def apply_bn(self, x):
      ''' Batch normalization of 3D tensor x
      '''
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      #bn_module = torch.nn.BatchNorm1d(x.size()[1]).to(self._device)
      bn_module = torch.nn.BatchNorm1d(x.size()[1]).to(device)
      return bn_module(x)
