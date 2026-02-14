import torch

import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self,d_model : int, vocab_size : int):
        super().__init__()

        self.d_model = d_model 
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size,d_model)


    def forward(self,x):

        return self.embedding(x)* math.sqrt(d_model)



class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int, seq_len : int,dropout:float) ->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # seq_len times d_model 

        pe = torch.zeros(self.seq_len , self.d_model)

        positions = torch.arange(0,seq_len,dtype=torch.float).unsqueeze()

        div = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(100000)/d_model))










