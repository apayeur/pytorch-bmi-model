import torch
import os
import sys
import numpy as np
from torch.utils.data import DataLoader
sys.path.append("../src")
from datasets import EndLossDataset
from effectors import PointMassArm, VelocityIntegrator
import matplotlib.pyplot as plt
from plot_utils import units_convert, plot_trajectories, plot_single_loss
from networks import NoisyNetWithFeedback, NoisyNet
from decode import build_bci_decoder
from objectives import EndLoss
import argparse
plt.style.use('../plot_params.dms')

# Arguments parsing
parser = argparse.ArgumentParser()
parser.add_argument("--n_readouts", type=int, help="number of readout units", default=10)
parser.add_argument("--seed", type=str, help="seed", default=1)
parser.add_argument("--clda_frequency", type=int,
                    help="frequency with which to perform CLDA, in number of epochs", default=50)
parser.add_argument("--clda_start", type=int, help="epoch ID to start CLDA at", default=0)
parser.add_argument("--clda_stop", type=int, help="epoch ID to stop CLDA at", default=1e9)
parser.add_argument("--clda", type=float,
                    help="intensity of CLDA (= 1. - alpha_CLDA, according to my older notation)", default=0.1)
args = parser.parse_args()

# CLDA parameters
clda_frequency = args.clda_frequency
clda_start = args.clda_start
clda = args.clda

# Load data
seed = args.seed
MANUAL_DATADIR = "../data/point-mass-arm-with-feedback"
params = np.load(os.path.join(MANUAL_DATADIR, f"params_seed{seed}.npy"), allow_pickle=True).item()

net = NoisyNetWithFeedback(network_size=params['size'], nonlinearity=params['nonlinearity'], delay=params['delay'])
#net = NoisyNet(network_size=params['size'], nonlinearity=params['nonlinearity'])
net.effector = PointMassArm()  # must add point-mass arm before loading
net.load_state_dict(torch.load(os.path.join(MANUAL_DATADIR, f"model_seed{seed}.pth")))
net.eval()

# Reproducibility
torch.manual_seed(seed)

# ======================  MAIN CODE  ====================== #
# Paths to save data and results
DATADIR = "../data/bci-with-feedback-SGD-nonstop"
RESULTDIR = "../results/bci-with-feedback-SGD-nonstop"
if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)
if not os.path.exists(RESULTDIR):
    os.makedirs(RESULTDIR)

# Dataset and dataloader
dataset = EndLossDataset(n_targets=params['n_targets'], total_duration=params['total_duration'], dt=params['dt'],
                         distance=0.07, context='bci')
dataloader = DataLoader(dataset, batch_size=params['n_targets'])

# Build BCI decoder
D = build_bci_decoder(net, dataloader, params['noisy_ics'], n_readouts=args.n_readouts, clda=0.)  # clda = 0. always here
net.readout_layer.weight.data = torch.from_numpy(D)     # attach decoder to network
net.effector = VelocityIntegrator()                            # remove the arm

# Test BCI control before learning
_, ax = plt.subplots(figsize=(45*units_convert['mm'], 45*units_convert['mm']))
plot_trajectories(ax, net, dataset, params['noisy_ics'])
plt.savefig(os.path.join(RESULTDIR, f"BCITrajectoriesBeforeLearning_seed{seed}_clda{clda}.png"))


# Training under BCI control
epochs = 2000
lambda_ctrl = 0.
loss_fn = EndLoss(params['gamma_v'], params['gamma_f'], lambda_ctrl, params['dt'])
optimizer = torch.optim.SGD(net.parameters(), lr=2e-5)
optimizer.zero_grad()
next_CLDA = clda_start
losses = []
for t in range(epochs):
    for X, y in dataloader:
        # Prediction
        v0 = params['noisy_ics']* torch.rand(
            (1, params['n_targets'], net.network_size[2]))  # set of initial conditions for motor cortex
        pred, _, controls = net(X, v0)

        # Loss
        loss, loss_ctrl = loss_fn(pred, y.unsqueeze(1), controls)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # CLDA
        if clda > 0:
            if t == next_CLDA and t < args.clda_stop:
                D_new = build_bci_decoder(net, dataloader, params['noisy_ics'], n_readouts=args.n_readouts, clda=args.clda)
                D = clda * D_new + (1 - clda) * D
                net.readout_layer.weight.data = torch.from_numpy(D)
                next_CLDA += clda_frequency

        losses.append(loss.item())
        if t % 100 == 0:
            print(f"Epoch {t}: loss = {loss.item():>10e}")

plot_single_loss(losses, os.path.join(RESULTDIR, f"Loss_seed{seed}_clda{clda}.png"))

# Test BCI control after learning
_, ax = plt.subplots(figsize=(45*units_convert['mm'], 45*units_convert['mm']))
plot_trajectories(ax, net, dataset, params['noisy_ics'])
plt.savefig(os.path.join(RESULTDIR, f"BCITrajectoriesAfterLearning_seed{seed}_clda{clda}.png"))

# Saving
torch.save(net.state_dict(), os.path.join(DATADIR, f"model_seed{seed}_clda{clda}.pth"))    # model parameters
np.save(os.path.join(DATADIR, f"loss_seed{seed}_clda{clda}.npy"), losses)                  # loss
np.save(os.path.join(DATADIR, f"params_seed{seed}_clda{clda}.npy"), vars(args))            # parameters
np.save(os.path.join(DATADIR, f"params_from_manual_seed{seed}_clda{clda}.npy"), params)            # parameters inherited from manual control


#