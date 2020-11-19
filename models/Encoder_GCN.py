import torch
import torch.nn as nn
from torch_geometric.data import Data
from models.gcn_hpool_encoder import GcnHpoolEncoder
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from torchvision import models
class GraphEncoder_GCN(nn.Module):
    def __init__(self,in_feature, hidden_feature, out_feature):
        super(GraphEncoder_GCN, self).__init__()

        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.out_feature = out_feature


        self.conv1 = GCNConv(in_feature, hidden_feature)
        self.conv2 = GCNConv(hidden_feature, hidden_feature)
        self.conv3 = GCNConv(hidden_feature,out_feature)

        # self.linear1 = nn.Linear(hidden_feature,hidden_feature)
        self.linear1 = nn.Linear(hidden_feature,out_feature)

        self.linear2 = nn.Linear(hidden_feature,hidden_feature)
        self.linear3 = nn.Linear(out_feature,out_feature)
        self.linear4 = nn.Linear(hidden_feature+hidden_feature+out_feature,out_feature)


        self.relu = torch.relu

    def forward(self,  Graph,batch):
        x, edge_index = Graph.x, Graph.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        embedding_1 = global_mean_pool(x,batch)
        embedding_1 = self.linear1(embedding_1)
        embedding_1 = self.relu(embedding_1)
        #
        # x = self.conv2(x, edge_index)
        # x = self.relu(x)
        # embedding_2 = global_mean_pool(x, batch)
        # embedding_2 = self.linear2(embedding_2)
        # embedding_2 = self.relu(embedding_2)
        #
        # x = self.conv3(x, edge_index)
        # x = self.relu(x)
        # embedding_3 = global_mean_pool(x, batch)
        # embedding_3 = self.linear3(embedding_3)
        # embedding_3 = self.relu(embedding_3)
        #
        # #print(embedding_1.shape,embedding_2.shape,embedding_3.shape)
        # G_embedding = self.linear4(torch.cat([embedding_1,embedding_2,embedding_3],dim=1)) #1x**


        return self.relu(embedding_1)

if __name__ == '__main__':
    edge_index = torch.tensor([[0, 1],
                               [1, 0],
                               [1, 2],
                               [2, 1],
                               [2, 3],
                               [2, 4],
                               [2, 5]], dtype=torch.long)
    x = torch.randn([6, 50])
    batch = torch.zeros([6,1]).long()
    #x.squeeze()
    G = Data(x=x, edge_index=edge_index.t().contiguous())
    mode = GraphEncoder_GCN(50,20,15)
    a = mode(G)
    print(a.unsqueeze(0).shape)