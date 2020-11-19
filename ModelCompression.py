
import torch
import torch.nn as nn
from torch_geometric.data import Data
from models.gcn_hpool_encoder import GcnHpoolEncoder
from models.Decoder import Decoder_Env_LSTM as Decoder_LSTM
from DNN.Example_CNN import CNN
class DNNCompression(nn.Module):
    def __init__(self,in_feature, hidden_feature, out_feature, h_hidden_feature,
               h_out_feature,in_node, hidden_node, out_node,n_layer,n_linear_hidden,n_linear_out):
        super(DNNCompression, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.out_feature = out_feature
        self.h_hidden_feature = h_hidden_feature
        self.h_out_feature = h_out_feature
        self.in_node = in_node
        self.hidden_node = hidden_node
        self.out_node = out_node
        self.n_channel = n_layer
        self.n_linear_out=n_linear_out

        self.encoder = GcnHpoolEncoder(in_feature, hidden_feature, out_feature, h_hidden_feature,
               h_out_feature,in_node, hidden_node, out_node)

        self.decoder = Decoder_LSTM(h_out_feature,n_layer)
        self.fc1 = nn.Linear(h_out_feature,n_linear_hidden)
        self.relu = torch.relu
        self.fc2 = nn.Linear(n_linear_hidden,n_linear_out)

    def forward(self, Graph):
        G_embedding = self.encoder(Graph)
        G_embedding = G_embedding.reshape(G_embedding.shape[0],1,G_embedding.shape[1])
        out = self.decoder(G_embedding)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)

        return self.relu(out)



if __name__ == '__main__':
    edge_index = torch.tensor([[0, 1],
                               [1, 0],
                               [1, 2],
                               [2, 1],
                               [2, 3],
                               [2, 4],
                               [2, 5]], dtype=torch.long)
    x = torch.randn([6, 50])
    data = Data(x=x, edge_index=edge_index.t().contiguous())


    in_c_1 = 1
    out_c_1 = 25
    in_c_2 = 25
    out_c_2 = 50
    DNN = CNN(in_c_1,out_c_1,in_c_2,out_c_2)
    print(DNN)

    model = DNNCompression(50, 15, 18, 20, 30,50, 3, 10,3,5,10)
    #in_feature, hidden_feature, out_feature, h_hidden_feature, h_out_feature,in_node, hidden_node, out_node

    out = model(data).squeeze(0)
    print(out.shape)
    # out_c_1 = int(out[0]*out_c_1)
    # in_c_2 = int(out[0]*in_c_2)
    # out_c_2 = int(out[2]*out_c_2)
    # candidate_DNN = CNN(in_c_1,out_c_1,in_c_2,out_c_2)
    # print(candidate_DNN)