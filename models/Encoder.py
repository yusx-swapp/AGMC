import torch
import torch.nn as nn
from torch_geometric.data import Data
from models.gcn_hpool_encoder import GcnHpoolEncoder
from torchvision import models

class GraphEncoder(nn.Module):
    def __init__(self,in_feature, hidden_feature, out_feature, h_hidden_feature,
               h_out_feature,in_node, hidden_node, out_node):
        super(GraphEncoder, self).__init__()

        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.out_feature = out_feature
        self.h_hidden_feature = h_hidden_feature
        self.h_out_feature = h_out_feature
        self.in_node = in_node
        self.hidden_node = hidden_node
        self.out_node = out_node

        # self.encoder_motif = GcnHpoolEncoder(in_feature, hidden_feature, out_feature, h_hidden_feature,
        #        h_out_feature,in_node, hidden_node, out_node)
        self.encoder = GcnHpoolEncoder(in_feature, hidden_feature, out_feature, h_hidden_feature,
               h_out_feature,in_node, hidden_node, out_node)

        #self.decoder = Decoder_LSTM(h_out_feature,n_layer)
        #self.fc1 = nn.Linear(h_out_feature,n_linear_hidden)
        self.relu = torch.relu
        #self.fc2 = nn.Linear(n_linear_hidden,n_linear_out)

    def forward(self,  Graph):


        G_embedding = self.encoder(Graph)
        G_embedding = G_embedding.reshape(G_embedding.shape[0],1,G_embedding.shape[1]) # batch * 1 *emb_size
        #out = self.decoder(G_embedding)
        #out = self.relu(self.fc1(out))
        #out = self.fc2(out)

        return self.relu(G_embedding)

if __name__ == '__main__':
    edge_index = torch.tensor([[0, 1],
                               [1, 0],
                               [1, 2],
                               [2, 1],
                               [2, 3],
                               [2, 4],
                               [2, 5]], dtype=torch.long)
    x = torch.randn([6, 50])
    batch = torch.zeros([6, 1]).long()

    G = Data(x=x, edge_index=edge_index.t().contiguous())
    mode = GraphEncoder(50, 15, 18, 20, 25, 50, 3, 10)
    a = mode(G)
    print(a.shape)