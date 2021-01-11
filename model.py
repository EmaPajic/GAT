import torch 
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer

class GAT(nn.Module):
    def __init__(self, n_input, n_hidden, n_classes, dropout, alpha, n_heads):
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        self.attentions = [GraphAttentionLayer(n_input, n_hidden, 
                                               dropout = dropout, alpha = alpha,
                                               concat = True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            
        self.out_attention = GraphAttentionLayer(n_hidden * n_heads, n_classes, 
                                       dropout = dropout, alpha = alpha,
                                       concat = False)
    
    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.cat([att(x, adj) for att in self.attentions], dim = 1)
        x = self.dropout(x)
        x = self.out_attention(x, adj)
        return F.log_softmax(x, dim = 1)