import torch

n = 500
n_epochs = 1000

X = torch.rand(n, 1)
y = 3 * X + 0.05 * torch.rand(n, 1)

b = torch.rand(1, 1)
A = torch.rand(1, 1)

lr = 0.01
loss = 0.0
for i in range(n_epochs):
    y_pred = X @ A + b
    loss = (y_pred - y).pow(2).mean()

    grad_y = (y_pred-y)*2.0/y_pred.shape[0]
    grad_A = X.T @ grad_y
    grad_b = grad_y.sum()

    A -= lr * grad_A
    b -= lr * grad_b
    if (i % 50 == 0):
        print("total loss: ", loss)

print("A: ", A)
print("b: ", b)
    

    
