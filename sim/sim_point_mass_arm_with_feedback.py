import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
sys.path.append("../src")
from datasets import HoldsDataset, EndLossDataset
from effectors import PointMassArm
from networks import SimpleNet, NoisyRNN, NoisyNet, NoisyNetWithFeedback
from objectives import HoldsLoss, EndLoss
import numpy as np
import os
import argparse
from seaborn import color_palette, despine
from plot_utils import units_convert
plt.style.use('../plot_params.dms')

"""
Description:
-----------
Simulation of the center-out task using a point-mass arm.
Workspace dimension is 2. 
"""

# Arguments parsing
parser = argparse.ArgumentParser()
parser.add_argument("--n_targets", type=int, help="number of targets", default=8)
parser.add_argument("--dt", type=float, help="integration time step", default=0.01)
parser.add_argument("--total_duration", type=float, help="total duration of the reach, including holding times", default=1.)
parser.add_argument("--noisy_ics", type=float, help="noise intensity for RNN hidden layer initial condition", default=1.)
parser.add_argument("--reset_radius", type=float, help="noise intensity for arm initial condition", default=0.005)
parser.add_argument("--seed", type=str, help="seed", default=4)
parser.add_argument("--nonlinearity", type=str, help="nonlinearity", default='relu')
parser.add_argument("--gamma_v", type=float, help="hyperparameter for end-velocity loss", default=0.25)
parser.add_argument("--gamma_f", type=float, help="hyperparameter for end-force loss", default=0.05)
parser.add_argument("--lambda_ctrl", type=float, help="hyperparameter for control loss", default=0.05)  # was 0.05
parser.add_argument("--lambda_rate", type=float, help="hyperparameter for rate loss", default=0.)
parser.add_argument("--delay", type=int, help="hyperparameter for control loss", default=10)
parser.add_argument("--lr", type=float, help="learning rate", default=5e-4)
parser.add_argument("--size", type=tuple, help="size of the network (in, h1, h2, out)", default=(6, 100, 100, 2))
args = parser.parse_args()


# Function definitions
def plot_trajectories(net, dataset, outfile_name=None, n_reals=5):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(2*45*units_convert['mm'], 2*45/1.25*units_convert['mm']))
    colors = color_palette('colorblind', dataset.n_targets)
    with torch.no_grad():
        if n_reals > 1 and args.reset_radius > 0:
            for _ in range(n_reals):
                for i in range(dataset.n_targets):
                    pred, h, controls = net(dataset[i][0].unsqueeze(0), args.noisy_ics * torch.rand((1, 1, net.network_size[2])))
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
        fig.savefig(outfile_name, format='png')


def plot_loss(l, outfile_name=None):
    plt.figure(figsize=(45 * units_convert['mm'], 45 / 1.25 * units_convert['mm']))
    plt.semilogy(l)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    despine()
    plt.tight_layout()
    if outfile_name is not None:
        plt.savefig(os.path.join(RESULTDIR, f"Loss_seed{seed}.png"))


# ======================  MAIN CODE  ====================== #
# Paths to save data and results
DATADIR = "../data/point-mass-arm-with-feedback-noisy-reset-delay"
RESULTDIR = "../results/point-mass-arm-with-feedback-noisy-reset-delay"
if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)
if not os.path.exists(RESULTDIR):
    os.makedirs(RESULTDIR)

# Reproducibility
seed = args.seed
torch.manual_seed(seed)

# Define network
network_size = args.size
net = NoisyNetWithFeedback(network_size=network_size, nonlinearity=args.nonlinearity,
                           delay=args.delay, feedback_type='position_only')

# Use point-mass arm as effector
net.effector = PointMassArm(reset_radius=args.reset_radius)

# Define dataset and dataloader
# dataset = HoldsDataset(n_targets=args.n_targets, total_duration=args.total_duration, dt=args.dt,
#                         distance=0.07, hold_start=args.hold_start, hold_end=args.hold_end, sigma=args.sigma,
#                         context='arm')
dataset = EndLossDataset(n_targets=args.n_targets, total_duration=args.total_duration, dt=args.dt,
                         distance=0.07, context='arm')
dataloader = DataLoader(dataset, batch_size=args.n_targets)

# Define loss function TODO: make sure that `dataset` matches with `loss_fn`
# loss_fn = HoldsLoss(gamma_v, gamma_f, args.hold_start, args.hold_end, args.dt)  # torch.nn.MSELoss()
loss_fn = EndLoss(args.gamma_v, args.gamma_f, args.lambda_ctrl, args.lambda_rate, args.dt)

# Training
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
epochs = 5000
plot_trajectories(net, dataset, os.path.join(RESULTDIR, f"ManualControlTrajectoriesBeforeLearning_seed{seed}.png"))
losses = []

for t in range(epochs):
    for X, y in dataloader:
        # Prediction
        v0 = args.noisy_ics*torch.rand((1, args.n_targets, network_size[2]))  # set of initial conditions for motor cortex
        pred, h, controls = net(X, v0)

        # Loss
        loss, loss_ctrl, loss_rate = loss_fn(pred, y.unsqueeze(1), controls, h)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if t % 500 == 0:
            print(f"Epoch {t}: loss = {loss.item():>10e},  loss_Ctrl = {loss_ctrl.item():>10e}, loss_rate = {loss_rate.item():>10e}")

with torch.no_grad():
    for X, y in dataloader:
        pred, h, ctrls = net(X, args.noisy_ics * torch.rand((1, dataloader.batch_size, net.network_size[2])))
        print(torch.max(h))

# Plot some results
plot_trajectories(net, dataset, os.path.join(RESULTDIR, f"ManualControlTrajectories_seed{seed}.png"))
plot_loss(losses, os.path.join(RESULTDIR, f"Loss_seed{seed}.png"))

# Saving
torch.save(net.state_dict(), os.path.join(DATADIR, f"model_seed{seed}.pth"))    # model parameters
np.save(os.path.join(DATADIR, f"loss_seed{seed}.npy"), losses)                  # loss
np.save(os.path.join(DATADIR, f"params_seed{seed}.npy"), vars(args))            # parameters


