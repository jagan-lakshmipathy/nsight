import torch
import torch.nn as nn
import torch.optim as optim

# Simple RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size=32, hidden_size=64, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 10)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

device = "cuda"
model = SimpleRNN().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Fake data
batch_size = 64
seq_len = 50
x = torch.randn(batch_size, seq_len, 32, device=device)
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
