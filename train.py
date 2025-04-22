import torch
import torch.nn as nn
import torch.optim as optim
from model import MedicalNet
from dataset import get_data_loaders

def train():
    model = MedicalNet()
    dataloader = get_data_loaders('data/train')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for imgs, labels in dataloader:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), 'model.pth')

if _name_ == "_main_":
    train()
