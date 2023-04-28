import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
sys.path.append("../src")
from datasets import GaussianVelocityDataset
from networks import SimpleNet
import numpy as np
from sklearn.linear_model import LinearRegression

# Reproducibility
torch.manual_seed(1)


# Define network
network_size = (5, 100, 100, 2)
net = SimpleNet(network_size=network_size)
#print(net)


# (1) TRAIN FOR MANUAL CONTROL
dataset = GaussianVelocityDataset()
dataloader = DataLoader(dataset, batch_size=8)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

epochs = 150
for t in range(epochs):
    for X, y in dataloader:
        # Compute prediction and loss
        h0 = 0.1*torch.rand((1, 8, network_size[2]))  # set of initial conditions for motor cortex
        pred, _ = net(X, h0)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 100 == 0:
            print(f"Epoch {t}: loss = {loss.item():>10e}")


# (2) BUILD BCI DECODER
n_readouts = 10
with torch.no_grad():
    for X, y in dataloader:
        h0 = 0.1*torch.rand((1, 8, network_size[2]))  # set of initial conditions for motor cortex
        pred, h = net(X, h0)
h = torch.reshape(h, (-1, h.size(2)))
pred = torch.reshape(pred, (-1, pred.size(2)))

# perform regression
lin_reg = LinearRegression(fit_intercept=False)
lin_reg.fit(h[:, :n_readouts].numpy(), pred.numpy())
T = lin_reg.coef_

# define readout matrix
R = np.zeros((n_readouts, network_size[2]), dtype=float)
R[range(n_readouts), range(n_readouts)] = 1.

# attach decoder to network
net.readout_layer.weight.data = torch.from_numpy(T @ R)


# (3) TEST BCI CONTROL BEFORE LEARNING

