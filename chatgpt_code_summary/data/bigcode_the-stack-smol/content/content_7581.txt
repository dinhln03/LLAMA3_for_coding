"""Trains a hypergraph machine on MNIST and generates Figure 1 panels b and c
of Discrete and continuous learning machines
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from hypergraph_machines.hypergraph_machine import HypergraphMachine
from hypergraph_machines.utils import train, test, visualise_graph
from hypergraph_machines.dataset_loader import load_dataset
from hypergraph_machines.utils import BestModelSaver, generate_timestamp, reg_loss
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="paper", style="white")
plt.ion()

device = torch.device("cuda")
timestamp = generate_timestamp()
batch_size, num_epochs = 128, 100
train_loader,\
test_loader,\
image_size = load_dataset("MNIST", batch_size, data_folder = "../../data")
model = HypergraphMachine((1,28,28), 10, number_of_classes = 10, tol = 1e-6,
                          limit_image_upsample = 2, prune=True).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr= 3e-3)
saver = BestModelSaver('./checkpoints' + timestamp)

for epoch in range(1, num_epochs + 1):
    print("starting epoch {} of {}".format(epoch, num_epochs))
    train(model, device, train_loader, optimizer, epoch,
          loss_func = reg_loss, loss_inputs = [model, F.nll_loss, 1])
    loss, acc = test(model, device, test_loader)
    saver.save(model, optimizer, epoch, loss, acc)
    if epoch % 10 == 1:
        f,ax = plt.subplots()
        visualise_graph(model, ax=ax)
        f.suptitle("epoch {}".format(epoch))
