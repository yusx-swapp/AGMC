import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Decoder_Env_LSTM(nn.Module):
    def __init__(self, ninp,n_channel):
        super(Decoder_Env_LSTM, self).__init__()
        self.ninp = ninp
        self.n_channel = n_channel
        self.lstm = nn.LSTM(ninp,ninp)
        self.linear = nn.Linear(ninp+1,ninp)
        self.relu = torch.relu
    def forward(self, s,a, t):
        '''

        :param s: state of the env with batch * 1 *emb_size
        :return: batch*n_channel*emb_size
        '''
        done = False
        if t == self.n_channel-1: #final layer we will caculate rewards outside the Decoder
            done = True
        else:
            rewards = 0

        s = s.transpose(0,1)
        s_, _ = self.lstm(s)
        s_ = torch.cat([s_.transpose(0,1),a],dim=2)#batch * 1 *emb_size
        s_ = self.linear(s_.float())
        s_ = self.relu(s_)



        # for n in range(self.n_channel):
        #     #seqlen * batch *i nputsize
        #
        #     input,_ = self.lstm(input)
        #
        #     if n == 0:
        #         res = input.transpose(0,1)
        #     else:
        #         res = torch.cat([res, input.transpose(0,1)], dim=1)

        return s_, rewards