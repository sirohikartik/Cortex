#Linear Regression Implementation using PyTorch

import torch

device = 'mps'
X = torch.rand(300,300).to('mps')
y = torch.tensor(torch.sum(X,dim=1,keepdims=True)+3).to('mps')

bias_col = torch.ones(X.shape[0],1).to(device)

X = torch.cat([X,bias_col],dim = 1)

print(X.shape)
print(y.shape)


# Normal Equation with L2(Ridge optimization)

l = 0.001

I = torch.eye(X.shape[1]).to(device)

W_optimal = torch.matmul(torch.linalg.inv((torch.matmul(torch.transpose(X,0,1),X)+l*I)),torch.matmul(torch.transpose(X,0,1),y))

print(W_optimal.shape)

y_pred = torch.matmul(X,W_optimal)

mae = (y_pred - y)**2
print(torch.mean(mae))

