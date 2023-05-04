import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
sys.path.append("../src")
from datasets import HoldsDataset, EndLossDataset
from effectors import PointMassArm
from networks import SimpleNet, NoisyRNN, NoisyNet
from objectives import HoldsLoss, EndLoss
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
from seaborn import color_palette, despine
from plot_utils import units_convert
plt.style.use('../plot_params.dms')

"""
Description:
-----------
Simulation of the center-out BCI task using a point-mass arm to establish the inductive bias.
Workspace dimension is 2. 
"""


# Arguments parsing
parser = argparse.ArgumentParser()
parser.add_argument("-n_targets", type=int, help="number of targets", default=8)
parser.add_argument("-n_readouts", type=int, help="number of readout units", default=12)
parser.add_argument("-dt", type=float, help="integration time step", default=0.01)
parser.add_argument("-sigma", type=float, help="time spread of target velocity profile", default=0.1)
parser.add_argument("-total_duration", type=float, help="total duration of the reach, including holding times", default=1.)
parser.add_argument("-hold_start", type=float, help="duration of the preparatory hold", default=0.25)
parser.add_argument("-hold_end", type=float, help="duration of the termination hold", default=0.25)
parser.add_argument("-noisy_ics", type=float, help="noise intensity for RNN hidden layer initial condition", default=0.1)
parser.add_argument("-seed", type=str, help="seed", default=2)
parser.add_argument("-size", type=tuple, help="size of the network (in, h1, h2, out)", default=(4, 100, 100, 2))
args = parser.parse_args()


# Function definitions
def plot_trajectories(net, dataset, network_size, outfile_name=None):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(2*45*units_convert['mm'], 2*45/1.25*units_convert['mm']))
    colors = color_palette('colorblind', dataset.n_targets)
    for i in range(dataset.n_targets):
        pred, _, controls = net(dataset[i][0].unsqueeze(0), NOISE_FOR_HIDDEN_INIT * torch.rand((1, 1, network_size[2])))
        pred = pred.detach().numpy()
        controls = controls.detach().numpy()
        pred_traj = pred[0, :, :2]
        target = dataset[i][1].detach().numpy()

        # plot trajectories
        axes[0, 0].plot(pred_traj[:, 0], pred_traj[:, 1], color=colors[i])
        axes[0, 0].plot(target[0], target[1], color=colors[i], marker='o', markersize=3, lw=0.)

        # plot velocities
        pred_vel = pred[0, :, 2:4]
        axes[0, 1].plot(args.dt * np.arange(len(pred_vel)), np.linalg.norm(pred_vel, axis=1), color=colors[i])

        # plot forces
        pred_f = pred[0, :, 4:]
        axes[1, 0].plot(args.dt * np.arange(len(pred_f)), np.linalg.norm(pred_f, axis=1), color=colors[i])

        # plot controls
        axes[1, 1].plot(args.dt * np.arange(len(pred_f)), np.linalg.norm(controls[0, :, :], axis=-1), color=colors[i])
    axes[0,0].axis('off')
    axes[0,0].set_aspect('equal', 'box')
    axes[0,0].set_xticks([])
    axes[0,0].set_yticks([])
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Speed (m/s)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Force magnitude (N)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Control magnitude (N)')
    for ax in axes.ravel()[1:]:
        despine(ax=ax)
    fig.tight_layout()
    if outfile_name is not None:
        fig.savefig(f"../results/{outfile_name}", format='png')


def build_bci_decoder(net, dataloader, network_size, n_readouts=10, clda=0.):
    assert clda < 1e-6, "CLDA not yet implemented"
    with torch.no_grad():
        for X, y in dataloader:
            h0 = NOISE_FOR_HIDDEN_INIT * torch.rand(
                (1, 8, network_size[2]))  # set of initial conditions for motor cortex
            pred, h, _ = net(X, h0)
            if clda > 0.:
                pass
    h = torch.reshape(h, (-1, h.size(2)))
    pred = pred[:, :, 2:4]  # only using the velocities
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
net = NoisyNet(network_size=network_size, nonlinearity='relu')

# Use point-mass arm as effector
net.effector = PointMassArm()

# Parameters
NOISE_FOR_HIDDEN_INIT = args.noisy_ics
CLDA_FREQUENCY = 20     # frequency with which to perform CLDA, in number of epochs
CLDA_START = 0          # epoch ID to start CLDA at
CLDA = 0.0              # intensity of CLDA (= 1. - alpha_CLDA, according to my older notation)
GAMMA_v = 0.25
GAMMA_f = 0.05
lambda_ctrl = 0.01
N_READOUTS = args.n_readouts

# (1) TRAIN FOR MANUAL CONTROL
#dataset = HoldsDataset(n_targets=args.n_targets, total_duration=args.total_duration, dt=args.dt,
#                         distance=0.07, hold_start=args.hold_start, hold_end=args.hold_end, sigma=args.sigma,
#                         context='arm')
dataset = EndLossDataset(n_targets=args.n_targets, total_duration=args.total_duration, dt=args.dt,
                         distance=0.07, context='arm')
dataloader = DataLoader(dataset, batch_size=args.n_targets)

#loss_fn = HoldsLoss(GAMMA_v, GAMMA_f, args.hold_start, args.hold_end, args.dt)  # torch.nn.MSELoss()
loss_fn = EndLoss(GAMMA_v, GAMMA_f, lambda_ctrl, args.dt)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

epochs = 10000
losses = []
for t in range(epochs):
    for X, y in dataloader:
        # Prediction
        h0 = NOISE_FOR_HIDDEN_INIT*torch.rand((1, 8, network_size[2]))  # set of initial conditions for motor cortex
        pred, _, controls = net(X, h0)

        # Loss
        loss, loss_ctrl = loss_fn(pred, y.unsqueeze(1), controls)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if t % 500 == 0:
            print(f"Epoch {t}: loss = {loss.item():>10e},  loss_Ctrl = {loss_ctrl.item():>10e}")
plt.semilogy(losses)
plt.show()
plot_trajectories(net, dataset, network_size, f"ManualControlTrajectoriesPMA_seed{SEED}_lambdactrl{lambda_ctrl}.png")
plt.show()

"""
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

"""
