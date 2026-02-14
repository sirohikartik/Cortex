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

        return self.embedding(x)* math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int, seq_len : int,dropout:float) ->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # seq_len times d_model 

        pe = torch.zeros(self.seq_len , self.d_model)

        positions = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(0)

        div = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(100000)/d_model))

        pe[:,0::2] = torch.sin(positions * div)
        pe[:,1::2] = torch.cos(positions*div)

        pe = pe.unsqueeze(0) #(1,seq_len,d_model)

        self.register_buffer('pe',pe)


    def forward(self,x):
        x =  x + self.pe[:,:x.shape[1],:].requires_grad_(False)
        return self.dropout(x)



class LayerNorm(nn.Module):
    def __init__(self,eps:float= 10e-6):
        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1)) # *
        self.bias = nn.Parameter(torch.ones(1))  # +

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)

        std = x.std(dim=-1,keepdim=True)

        return self.alpha* (x-mean)/(std + self.eps) + self.bias


class FeedForward(nn.Module):
    def __init__(self,d_model : int , d_ff  : int,dropout : float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)


    def forward(self,x):
        # (B,s,d_model) --> (B,s,d_ff) --> (B,s,d_model)
        return self.linear_2(self.dropout(self.linear_1(x)))


class MHA(nn.Module):
    def __init__(self,d_model : int, h : int, dropout : float) -> None:
        super().__init__()

        self.d_model = d_model
        self.h = h

        assert d_model % h == 0 

        self.d_k = d_model//h

        self.w_q = nn.Linear(d_model,d_model)
        self.w_k  = nn.Linear(d_model,d_model)

        self.w_v = nn.Linear(d_model,d_model)


        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout : nn.Dropout):
        d_k =query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask==0,-1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value, attention_scores)


    def forward(self,q,k,v,mask=None):
        query = self.w_q(q)  # b,s,d --> b,s,d
        key = self.w_k(k)
        values = self.w_v(v)


        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)

        # b, h, seq, d_k

        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)

        values = values.view(values.shape[0],values.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores = MHA.attention(query,key,values,mask,self.dropout)
        # batch, h, seq, d_k --> batch, seq, h, d_k --> 

        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        # btach, seq , d_model -> batch, seq, d_model
        return self.w_o(x)

class ResNet(nn.Module):
    def __init__(self,dropout:float) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout) 
        self.norm = LayerNorm()


    def forward(self,x,y):
        return x + self.dropout(y(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block : MHA, feed_forward : FeedForward, dropout : float)->None:
        super().__init__()

        self.attn = self_attention_block
        self.feedforward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.ModuleList(ResNet(dropout) for _ in range(2))

    def forward(self,x,src_mask):
        x = self.residual[0](x,lambda x : self.attn(x,x,x,src_mask))
        x = self.residual[1](x,lambda x : self.feedforward(x))
        return x
class Encoder(nn.Module):

    def __init__(self,layers:nn.ModuleList)->None:
        super().__init__()

        self.layers = layers

        self.norm = LayerNorm()

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

































