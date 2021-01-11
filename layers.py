import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout = 0.6, alpha = 0.2, concat = True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size = (in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain = 1.414)
        self.a = nn.Parameter(torch.empty(size = (2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain = 1.414)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        WhiWhj = self._prepare_attention_input(Wh)
        e = self.leakyrelu(torch.matmul(WhiWhj, self.a).squeeze(2))
        
        zero_vec = -9e15 * torch.ones_like(e) # because softmax -> will be 0
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim = 1)
        attention = self.dropout(attention)
        
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attention_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        # The all_combination_matrix:
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim = 1)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)
