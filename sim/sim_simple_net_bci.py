import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
sys.path.append("../src")
from datasets import GaussianVelocityDataset
from networks import SimpleNet
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
from seaborn import color_palette, despine
from plot_utils import units_convert
plt.style.use('../plot_params.dms')

"""
Description:
-----------
Simple simulation of a BCI experiment, using the `SimpleNet` neural network.

Training objective: reproduce Gaussian velocities in the direction of the targets, according to GaussianVelocityDataset.

Simulation structure:
# (1) TRAIN FOR MANUAL CONTROL
# (2) BUILD BCI DECODER
# (3) TEST BCI CONTROL BEFORE LEARNING
# (4) TRAINING BCI CONTROL with or without "CLDA".

CLDA is in quotes because real CLDA uses the intended velocities to refit the decoder whereas in the present case 
only the output velocity is used. 
(There is no notion of intended velocity because the spatial position of the cursor is irrelevant for the task here.) 
"""
# Arguments parsing
parser = argparse.ArgumentParser()
parser.add_argument("-n_targets", type=int, help="number of targets", default=8)
parser.add_argument("-n_readouts", type=int, help="number of readout units", default=12)
parser.add_argument("-dt", type=float, help="integration time step", default=0.01)
parser.add_argument("-sigma", type=float, help="time spread of target velocity profile", default=0.1)
parser.add_argument("-total_duration", type=float, help="total duration of the reach, including holding times", default=1.5)
parser.add_argument("-hold_start", type=float, help="duration of the preparatory hold", default=0.25)
parser.add_argument("-hold_end", type=float, help="duration of the termination hold", default=0.25)
parser.add_argument("-noisy_ics", type=float, help="noise intensity for RNN hidden layer initial condition", default=0.1)
parser.add_argument("-seed", type=str, help="seed", default=1)
parser.add_argument("-size", type=tuple, help="size of the network (in, h1, h2, out)", default=(5, 100, 100, 2))
args = parser.parse_args()


# Function definitions
def plot_trajectories(net, dataset, network_size, outfile_name):
    fig, ax = plt.subplots(ncols=1, figsize=(45*units_convert['mm'], 45*units_convert['mm']))
    colors = color_palette('colorblind', dataset.n_targets)
    for i in range(dataset.n_targets):
        pred, _ = net(dataset[i][0], NOISE_FOR_HIDDEN_INIT * torch.rand((1, network_size[2])))
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
    fig.savefig(f"../results/{outfile_name}", format='png')


def build_bci_decoder(net, dataloader, network_size, n_readouts=10, clda=0.):
    with torch.no_grad():
        for X, y in dataloader:
            h0 = NOISE_FOR_HIDDEN_INIT * torch.rand(
                (1, 8, network_size[2]))  # set of initial conditions for motor cortex
            pred, h = net(X, h0)
            if clda > 0.:
                pred = y
    h = torch.reshape(h, (-1, h.size(2)))
    pred = torch.reshape(pred, (-1, pred.size(2)))

    # perform regression
    lin_reg = LinearRegression(fit_intercept=False)
    lin_reg.fit(h[:, :n_readouts].numpy(), pred.numpy())
    T = lin_reg.coef_

    # define readout matrix
    R = np.zeros((n_readouts, network_size[2]), dtype=np.float32)  # tensor will have to be a float, not a double
    R[range(n_readouts), range(n_readouts)] = 1.
    return T @ R


# ======================  MAIN CODE  ====================== #
# Reproducibility
SEED = args.seed
torch.manual_seed(SEED)

# Define network
network_size = args.size
net = SimpleNet(network_size=network_size, nonlinearity='relu')

# Parameters
NOISE_FOR_HIDDEN_INIT = args.noisy_ics
CLDA_FREQUENCY = 20     # frequency with which to perform CLDA, in number of epochs
CLDA_START = 0          # epoch ID to start CLDA at
CLDA = 0.0              # intensity of CLDA (= 1. - alpha_CLDA, according to my older notation)
N_READOUTS = args.n_readouts

# (1) TRAIN FOR MANUAL CONTROL
dataset = GaussianVelocityDataset(n_targets=args.n_targets, total_duration=args.total_duration, dt=args.dt,
                                  distance=0.07, hold_start=args.hold_start, hold_end=args.hold_end, sigma=args.sigma,
                                  context='arm')
dataloader = DataLoader(dataset, batch_size=args.n_targets)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

epochs = 1500
for t in range(epochs):
    for X, y in dataloader:
        # Compute prediction and loss
        h0 = NOISE_FOR_HIDDEN_INIT*torch.rand((1, 8, network_size[2]))  # set of initial conditions for motor cortex
        pred, _ = net(X, h0)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 100 == 0:
            print(f"Epoch {t}: loss = {loss.item():>10e}")

plot_trajectories(net, dataset, network_size, "ManualControlTrajectories.png")
plt.show()


# (2) BUILD BCI DECODER
D = build_bci_decoder(net, dataloader, network_size, n_readouts=N_READOUTS)
net.readout_layer.weight.data = torch.from_numpy(D)  # attach decoder to network


# (3) TEST BCI CONTROL BEFORE LEARNING
dataset = GaussianVelocityDataset(n_targets=args.n_targets, total_duration=args.total_duration, dt=args.dt,
                                  distance=0.07, hold_start=args.hold_start, hold_end=args.hold_end, sigma=args.sigma,
                                  context='bci')
dataloader = DataLoader(dataset, batch_size=args.n_targets)
with torch.no_grad():
    for X, y in dataloader:
        h0 = NOISE_FOR_HIDDEN_INIT*torch.rand((1, 8, network_size[2]))
        pred, _ = net(X, h0)

plot_trajectories(net, dataset, network_size, "BCIControlTrajectories_BeforeLearning.png")
plt.show()


# (4) TRAINING BCI CONTROL
epochs = 1500
optimizer = torch.optim.Adam(net.parameters(), lr=2e-4)
optimizer.zero_grad()
next_CLDA = CLDA_START
losses = []
for t in range(epochs):
    for X, y in dataloader:
        # Compute prediction and loss
        h0 = NOISE_FOR_HIDDEN_INIT*torch.rand((1, 8, network_size[2]))  # set of initial conditions for motor cortex
        pred, _ = net(X, h0)
        loss = loss_fn(pred, y)
        losses.append(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # CLDA
        if CLDA > 0:
            if t == next_CLDA:
                D_new = build_bci_decoder(net, dataloader, network_size, n_readouts=N_READOUTS, clda=CLDA)
                D = (1. - CLDA) * D_new + CLDA * D
                net.readout_layer.weight.data = torch.from_numpy(D)  # attach decoder to network
                next_CLDA += CLDA_FREQUENCY

        if t % 100 == 0:
            print(f"Epoch {t}: loss = {loss.item():>10e}")

plot_trajectories(net, dataset, network_size, "BCIControlTrajectories_AfterLearning.png")
plt.show()

# SAVE DATA
np.save(f"../data/loss_clda{CLDA}_seed{SEED}.npy", losses)


