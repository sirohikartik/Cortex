import torch
import torch.nn.functional as F
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    '''Applying multi head attention'''
    def __init__(self,d_k,d_model,N=1):
        super().__init__()

        self.d_k = d_k
        self.N = N
        self.d_model = d_model
        self.lin_q = nn.ModuleList([nn.Linear(self.d_model,self.d_model//self.N) for _ in range(self.N)])
        self.lin_k = nn.ModuleList([nn.Linear(self.d_model,self.d_model//self.N) for _ in range(self.N)])
        self.lin_v = nn.ModuleList([nn.Linear(self.d_model,self.d_model//self.N) for _ in range(self.N)])
        self.final = nn.Linear(self.d_model,self.d_model)


    def forward(self,Q,K,V):
        heads = []
        for i in range(self.N):

            Q_ = self.lin_q[i](Q)
            K_ = self.lin_k[i](K)
            V_ = self.lin_v[i](V)

            attn_scores = torch.matmul(Q_,K_.transpose(-2,-1))/ torch.sqrt(torch.tensor(self.d_k))
            softmax_attn_scores = F.softmax(attn_scores)
            out = torch.matmul(softmax_attn_scores,V_)

            heads.append((out))

        return self.final(torch.cat(heads,dim=-1))

###### TESTS #####3

def run_mha_test():
    # Setup hyperparameters
    batch_size = 2
    seq_len = 10
    d_model = 128
    num_heads = 8
    d_k = d_model // num_heads # Standard d_k calculation
    
    # Initialize the model
    mha = MultiHeadAttention(d_k=d_k, d_model=d_model, N=num_heads)
    
    # Create dummy input data (Batch, Seq, Features)
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {Q.shape}")
    
    # Run Forward Pass
    try:
        output = mha(Q, K, V)
        print(f"Output shape: {output.shape}")
        
        # Test 1: Shape Validation
        assert output.shape == (batch_size, seq_len, d_model), "❌ Shape Mismatch!"
        print("✅ Test 1: Shape Validation Passed")
        
        # Test 2: Backpropagation Check
        loss = output.sum()
        loss.backward()
        
        # Check if gradients exist in one of the linear layers
        has_grads = mha.lin_q[0].weight.grad is not None
        print(f"✅ Test 2: Gradient Flow Passed" if has_grads else "❌ Test 2: No Gradients Found")
        
    except Exception as e:
        print(f"❌ Test Failed with error: {e}")

run_mha_test()
