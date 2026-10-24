# linear regression with sgd mini batch and batch gradient descent 
import torch
import numpy as np

# Batch gradient descent

def batch_gd(X,y,lr,epochs=100):
    W = torch.rand(X.shape[1],1)
    b = torch.rand(1,1)
# X.shape[0] = N i.e. number of samples and X.shape[1] = d i.e. dimensions so we're essentially doing X (N times d ) @ W (d times 1 ) + b(1 times 1 ) resulting in N times 1 vector y_pred which is same dimension as y 
    
    
    for epoch in range(epochs):
        y_pred = X@W  + b
        W -= lr * (X.t() @ (y_pred - y)) *(2/X.shape[0]) # (N times d).t @ (N times 1 ) i,e, d x N  @ N by 1  => d times 1 exactly shape of W 
        b -= lr * torch.mean(y_pred - y) 

    return W,b




def sgd(X,y,lr,epochs = 100):
    W = torch.rand(X.shape[1],1)
    b = torch.rand(1,1)

    for epoch in range(epochs):
        idx = np.random.randint(0,X.shape[0])
        x_i = X[idx:idx+1]
        y_i = y[idx:idx+1]
        y_pred = x_i@W  + b
        W -= lr * (x_i.t() @ (y_pred - y_i)) *(2/x_i.shape[0]) # (N times d).t @ (N times 1 ) i,e, d x N  @ N by 1  => d times 1 exactly shape of W 
        b -= lr * torch.mean(y_pred - y_i) 

    return W,b

def mini_batch_gd(X,y,lr,epochs= 100):
    W = torch.rand(X.shape[1],1)
    b = torch.rand(1,1)

    batch_size = 5

    for epoch in range(epochs):
        for idx in range(0,X.shape[0],batch_size):
            x_i = X[idx:idx+batch_size]
            y_i = y[idx:idx+batch_size]

            y_pred = x_i @ W + b
            W-= lr * (x_i.t() @ (y_pred - y_i )) * (2/x_i.shape[0])
            b-=lr*(torch.mean(y_pred - y_i))

    return W,b


# test cases running 

# For Batch gradient descent

def run_tests(model_fn, name="Model"):
    print(f"--- Testing {name} ---")
    
    # 1. Generate Synthetic Data
    # True weights: [2.0, 3.0], True bias: 5.0
    torch.manual_seed(42) # For reproducibility
    N, D = 1000, 2
    X = torch.randn(N, D)
    true_W = torch.tensor([[2.0], [3.0]])
    true_b = 5.0
    
    # y = XW + b + small noise
    y = (X @ true_W + true_b) + torch.randn(N, 1) * 0.01
    
    # 2. Run your function
    # Note: Adjust parameters based on your specific function signature
    try:
        if name == "Mini-Batch GD":
            learned_W, learned_b = model_fn(X, y, lr=0.1, epochs=100, batch_size=32)
        else:
            learned_W, learned_b = model_fn(X, y, lr=0.1, epochs=200)

        # 3. Check Results
        print(f"Learned W:\n{learned_W.flatten()}")
        print(f"Learned b: {learned_b.item():.4f}")
        
        # Calculate Mean Absolute Error (MAE)
        y_pred = X @ learned_W + learned_b
        mae = torch.mean(torch.abs(y_pred - y))
        print(f"Final MAE: {mae.item():.6f}")

        # Assertion tests
        assert torch.allclose(learned_W, true_W, atol=1e-1), "W is too far from true values"
        assert torch.allclose(learned_b, torch.tensor(true_b), atol=1e-1), "b is too far from true value"
        print("✅ Test Passed!\n")

    except Exception as e:
        print(f"❌ Test Failed with error: {e}\n")

# --- Example Usage ---
run_tests(batch_gd, name="Batch GD")
# run_tests(mini_batch_gd, name="Mini-Batch GD"multiplied)
run_tests(sgd,name="SGD")
run_tests(mini_batch_gd,name="Mini Batch GD")
    
