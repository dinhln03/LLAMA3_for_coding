import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, out_size):
        super(MLP, self).__init__()
        self.sizes = [input_size] + hidden_layers + [out_size]
        self.linears = [nn.Linear(in_dim, out_dim, True) for in_dim, out_dim in zip(self.sizes[: -1], self.sizes[1:])]
        self.linears = nn.ModuleList(self.linears)
        self.weight_init()

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = F.relu(layer(x))
        x = self.linears[-1](x)
        return x

    def weight_init(self):
        for layer in self.linears:
            torch.nn.init.xavier_uniform(layer.weight)
            torch.nn.init.zeros_(layer.bias)

class MLP_bn(nn.Module):
    def __init__(self, input_size, hidden_layers, out_size):
        super(MLP_bn, self).__init__()
        self.sizes = [input_size] + hidden_layers + [out_size]
        self.linears = [nn.Sequential(nn.Linear(in_dim, out_dim, True), nn.BatchNorm1d(out_dim)) for in_dim, out_dim in zip(self.sizes[: -1], self.sizes[1:])]
        self.linears = nn.ModuleList(self.linears)
        self.weight_init()

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = F.relu(layer(x))
        x = self.linears[-1][0](x)
        return x

    def weight_init(self):
        for layer in self.linears:
            torch.nn.init.xavier_uniform(layer[0].weight)
            torch.nn.init.zeros_(layer[0].bias)

class MLP_drop(nn.Module):
    def __init__(self, input_size, hidden_layers, out_size):
        super(MLP_drop, self).__init__()
        self.sizes = [input_size] + hidden_layers + [out_size]
        self.linears = [nn.Sequential(nn.Linear(in_dim, out_dim, True), nn.Dropout(0.5)) for in_dim, out_dim in zip(self.sizes[: -1], self.sizes[1:])]
        self.linears = nn.ModuleList(self.linears)
        self.weight_init()

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = F.relu(layer(x))
        x = self.linears[-1][0](x)
        return x

    def weight_init(self):
        for layer in self.linears:
            torch.nn.init.xavier_uniform(layer[0].weight)
            torch.nn.init.zeros_(layer[0].bias)

def train_nn(model, data, num_epoch=5000):
    train_dataset = TensorDataset(torch.Tensor(data.Xtrain), torch.Tensor(data.Ytrain))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    test_dataset = TensorDataset(torch.Tensor(data.Xtest), torch.Tensor(data.Ytest))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=128)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for epoch in range(num_epoch):
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        model.eval()
        loss = 0.
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            loss += criterion(outputs, targets).data
        losses.append(loss.data // len(test_dataloader))
        model.train()

    return losses
