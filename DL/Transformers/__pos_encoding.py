import torch
import math

# naive
# non vectorized
def pos_encoding(input_matrix):
    '''input_matrix must have shape seq_len * d_model'''
    pe_tensor = torch.zeros(input_matrix.shape)
    d = input_matrix.shape[1]
    for pos in range(len(pe_tensor)):
        for i in range(len(pe_tensor[pos])):
            val = pos/((10000)**((2*i)/d))
            if i%2==0:
                pe_tensor[pos][i] = torch.sin(val)
            else:
                pe_tensor[pos][i] = torch.cos(val)
    return pe_tensor

#vectorized

def pos_encoding_vectorized(input_matrix):
    '''input matrix has shape seq_len * d_model and we are gonna be vectorizing this now !!'''

    seq_len,d = input_matrix.shape
    pe = torch.zeros(seq_len,d)


    positions = torch.arange(0,seq_len,1).unsqueeze(1).float()

    denom = torch.exp(
        torch.arange(0,d,2).float() * -(2*math.log(10000.0)/d) 
    )

    terms = positions * denom

    pe[:,0::2] = torch.sin(terms)
    pe[:,1::2] = torch.cos(terms)

    return pe







