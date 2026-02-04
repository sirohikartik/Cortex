import torch
import numpy as np


def s(x):
    ans = 1/(1+torch.exp(-x))
    return ans


def logistic(X,y,lr,epochs):
    W = torch.rand(X.shape[1],1)
    b = torch.rand(1,1)
    # Batch GD
    for epoch in range(epochs):
        #Forward pass
        u =  X@W + b
        y_cap = s(u)
        #Backward Pass
        W -= lr * X.t() @ (y_cap - y) 
        b -= lr * torch.mean(y_cap - y)

    return W,b



######### TEST CASES #########

def run_test_cases():
    print(f"{'Test Case':<15} | {'Samples':<10} | {'Features':<10} | {'Final Accuracy':<15}")
    print("-" * 60)

    for i in range(1, 6):
        # 1. Generate random features
        num_samples = 100 + (i * 50)
        num_features = 2 
        X = torch.randn(num_samples, num_features)
        
        # 2. Create a "true" target based on a simple linear rule
        # e.g., if x1 + x2 > 0, then class 1
        true_weights = torch.tensor([[2.0], [-1.0]])
        true_bias = 0.5
        logits = X @ true_weights + true_bias
        
        # For harder test cases, we add a bit of noise to the labels
        y = (s(logits) > 0.5).float()
        if i > 3: # Add noise to last two cases
            noise_mask = torch.rand(y.shape) < 0.05
            y[noise_mask] = 1 - y[noise_mask]

        # 3. Train
        W_final, b_final = logistic(X, y, lr=0.1, epochs=1000)

        # 4. Inference
        preds = (s(X @ W_final + b_final) > 0.5).float()
        accuracy = (preds == y).float().mean() * 100

        print(f"Case {i}: {'Noise' if i > 3 else 'Clean':<8} | {num_samples:<10} | {num_features:<10} | {accuracy.item():.2f}%")

run_test_cases()


