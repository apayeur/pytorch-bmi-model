import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
sys.path.append("../src")
from datasets import GaussianVelocityDataset
from networks import SimpleNet
import numpy as np
from seaborn import color_palette

# Reproducibility
torch.manual_seed(2)

# Define network
network_size = (5, 100, 100, 2)
net = SimpleNet(network_size=network_size)
print(net)

# Train
dataset = GaussianVelocityDataset()
dataloader = DataLoader(dataset, batch_size=8)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

epochs = 1500
for t in range(epochs):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        h0 = 0.1*torch.rand((1, 8, network_size[2]))
        pred = net(X, h0)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 100 == 0:
            print(f"Epoch {t}: loss = {loss.item():>10e}")

# Plotting solution
fig, ax = plt.subplots(ncols=1, figsize=(3,3))
colors = color_palette('colorblind', dataset.n_targets)
for i in range(dataset.n_targets):
    pred = net(dataset[i][0], 0.1*torch.rand((1, network_size[2])))
    pred = pred.detach().numpy()
    pred_traj = dataset.dt * np.cumsum(pred, axis=0)
    target_vel = dataset[i][1].detach().numpy()
    target_traj = dataset.dt * np.cumsum(target_vel, axis=0)

    # plot trajectories
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], color=colors[i])
    ax.plot(target_traj[:, 0], target_traj[:, 1], '--', color=colors[i], lw=0.5)
ax.axis('off')
ax.set_aspect('equal', 'box')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

