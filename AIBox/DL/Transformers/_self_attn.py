import torch
import torch.nn.functional  as F

def Attention_Block(Q,K,V,d_k : float):
    '''Implementation of self attention in transformers'''

    A = F.softmax(torch.matmul(Q,K.transpose(-2,-1))/torch.sqrt(d_k),dim=-1)
    return torch.matmul(A,V)

    

#### TESTS #####


def run_tests(attention_func):
    print(f"Starting 10 Tests for: {attention_func.__name__}\n")
    print(f"{'Test':<8} | {'Input Shape (Q)':<20} | {'Output Shape':<20} | {'Status'}")
    print("-" * 65)

    for i in range(1, 11):
        # Randomize dimensions for variety
        batch_size = torch.randint(1, 8, (1,)).item()
        num_heads = torch.randint(1, 4, (1,)).item()
        seq_len = torch.randint(8, 32, (1,)).item()
        d_k = torch.tensor(64.0) # Standard d_k
        
        # Create tensors (Batch, Heads, Seq, Depth)
        shape = (batch_size, num_heads, seq_len, int(d_k.item()))
        Q = torch.randn(shape)
        K = torch.randn(shape)
        V = torch.randn(shape)

        try:
            output = attention_func(Q, K, V, d_k)
            
            # Validation: Output shape must match Q shape
            success = (output.shape == Q.shape)
            status = "✅ Pass" if success else "❌ Shape Mismatch"
            
            print(f"{i:<8} | {str(list(Q.shape)):<20} | {str(list(output.shape)):<20} | {status}")
            
        except Exception as e:
            print(f"{i:<8} | Error: {e}")
run_tests(Attention_Block)
