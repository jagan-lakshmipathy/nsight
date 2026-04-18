import torch
import torch.nn as nn
import torch.optim as optim

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

device = "cuda"
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Fake data
batch_size = 64
x = torch.randn(batch_size, 3, 32, 32, device=device)
y = torch.randint(0, 10, (batch_size,), device=device)

# Training loop
for step in range(20):
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    if step == 5:
        torch.cuda.synchronize()
        break

    if step % 5 == 0:
        print(f"step {step}, loss={loss.item():.4f}")
