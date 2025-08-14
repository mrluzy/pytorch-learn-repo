import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# TODO: import your dataset and model

# 配置
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
batch_size = 64
lr = 1e-3
epochs = 10

# TODO: prepare your dataset and dataloader

# TODO: define your model
model = nn.Module()  # replace with your model
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# TODO: training loop
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
