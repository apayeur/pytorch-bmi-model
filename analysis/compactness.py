import torch
import numpy as np
import sys
import os
from networks import NoisyNetWithFeedback, NoisyNet
from effectors import VelocityIntegrator
from objectives import EndLoss, HoldsLoss
from datasets import EndLossDataset, HoldsDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from seaborn import despine
from plot_utils import units_convert, colors
import copy
sys.path.append("../data")
plt.style.use('../plot_params.dms')

"""
Description:
-----------
Analyze compactness in the model after learning.
TODO: before learning as well
"""

n_seeds = 5
seeds = list(range(1, 1+n_seeds))
cldas = [0.0, 0.1, 0.2, 0.5]

DIR = "bci-with-feedback-velocity-cldastart0-cldastopFalse-cldafreq100-adam"
BCI_DATADIR = os.path.join("../data/", DIR)
RESULTDIR = os.path.join("../results", DIR)
if not os.path.exists(RESULTDIR):
    os.makedirs(RESULTDIR)

single_unit_ranking = {clda: [] for clda in cldas}
single_unit_ranking_losses = {clda: [] for clda in cldas}
ranked_uac = {clda: [] for clda in cldas}

for clda in cldas:
    for seed in seeds:
        try:
            # Load model and parameters
            params_manual = np.load(os.path.join(BCI_DATADIR, f"params_from_manual_seed{seed}_clda{clda}.npy"), allow_pickle=True).item()
            params_bci = np.load(os.path.join(BCI_DATADIR, f"params_seed{seed}_clda{clda}.npy"), allow_pickle=True).item()
            sigma = params_manual['sigma'] if 'sigma' in params_manual.keys() else 5e-3
            net = NoisyNetWithFeedback(network_size=params_manual['size'], nonlinearity=params_manual['nonlinearity'],
                                       delay=params_manual['delay'], feedback_type='position_and_velocity', sigma=sigma)
            reset_radius = params_manual['reset_radius'] if 'reset_radius' in params_manual.keys() else 0.
            net.effector = VelocityIntegrator(reset_radius=reset_radius)
            net.load_state_dict(torch.load(os.path.join(BCI_DATADIR, f"model_seed{seed}_clda{clda}.pth")))
            net.eval()

            # Compute single-unit ranking
            n_readouts = params_bci['n_readouts']
            D = copy.deepcopy(net.readout_layer.weight)

            # Define dataset and dataloader
            hold_duration = params_manual['hold_duration'] if 'hold_duration' in params_manual.keys() else 0
            if hold_duration > 0:
                dataset = HoldsDataset(n_targets=params_manual['n_targets'], total_duration=params_manual['total_duration'],
                                       dt=params_manual['dt'],
                                       distance=0.07, hold_start=params_manual['hold_duration'],
                                       hold_end=params_manual['hold_duration'],
                                       context='bci')
            else:
                dataset = EndLossDataset(n_targets=params_manual['n_targets'], total_duration=params_manual['total_duration'],
                                         dt=params_manual['dt'],
                                         distance=0.07, context='bci')
            dataloader = DataLoader(dataset, batch_size=params_manual['n_targets'])

            # Define loss function
            lambda_ctrl = 0.
            if hold_duration > 0:
                loss_fn = HoldsLoss(params_manual['gamma_v'], 0., lambda_ctrl, 0., hold_duration, hold_duration, params_manual['dt'])
            else:
                loss_fn = EndLoss(params_manual['gamma_v'], 0., lambda_ctrl, 0., params_manual['dt'])

            test_loss = np.empty(n_readouts)

            # Single-unit ranking
            for readout_i in range(n_readouts):
                net.readout_layer.weight.data = copy.deepcopy(D.data)
                net.readout_layer.weight.data[:, :readout_i] = 0.
                net.readout_layer.weight.data[:, readout_i+1:] = 0.
                with torch.no_grad():
                    for X, y in dataloader:
                        h0 = params_manual['noisy_ics'] * torch.randn(
                            (1, dataloader.batch_size, net.network_size[2]))  # set of initial conditions for motor cortex
                        pred, h, controls = net(X, h0)
                        test_loss[readout_i], _, _ = loss_fn(pred, y.unsqueeze(1), controls, h)
            ranked_units = np.argsort(test_loss)
            single_unit_ranking[clda].append(ranked_units)
            single_unit_ranking_losses[clda].append(np.sort(test_loss))

            # Compute ranked unit-adding curve
            test_loss = np.empty(n_readouts)
            net.readout_layer.weight.data = torch.zeros_like(D.data)
            for i, ranked_readout_i in enumerate(ranked_units):
                net.readout_layer.weight.data[:, ranked_readout_i] = D.data[:, ranked_readout_i]
                with torch.no_grad():
                    for X, y in dataloader:
                        h0 = params_manual['noisy_ics'] * torch.randn(
                            (1, dataloader.batch_size, net.network_size[2]))
                        pred, h, controls = net(X, h0)
                        test_loss[i], _, _ = loss_fn(pred, y.unsqueeze(1), controls, h)
            ranked_uac[clda].append(test_loss)
        except FileNotFoundError:
            print("File not found")


# Plot ranked single-unit loss
plt.figure(figsize=(45 * units_convert['mm'], 45 / 1.25 * units_convert['mm']))
for j, clda in enumerate(cldas):
    m = np.mean(np.log10(single_unit_ranking_losses[clda]), axis=0)
    sem = np.std(np.log10(single_unit_ranking_losses[clda]), axis=0, ddof=1) / len(single_unit_ranking_losses[clda])**0.5
    np.arange(1, 1 + n_readouts)
    plt.errorbar(np.arange(1, 1 + n_readouts), m, yerr=sem,
                 color='black' if clda < 1e-6 else colors[j], label=f"CLDA = {clda}")
plt.ylabel("Log$_{10}$ loss")
plt.xlabel("Ranked readout unit")
plt.legend()
plt.tight_layout()
despine()
plt.savefig(os.path.join(RESULTDIR, 'RankedSingleUnitLoss.png'))

# Plot ranked UAC
plt.figure(figsize=(45 * units_convert['mm'], 45 / 1.25 * units_convert['mm']))
for j, clda in enumerate(cldas):
    m = np.mean(np.log10(ranked_uac[clda]), axis=0)
    sem = np.std(np.log10(ranked_uac[clda]), axis=0, ddof=1) / len(ranked_uac[clda])**0.5
    plt.errorbar(np.arange(1, 1 + n_readouts), m, yerr=sem,
                 color='black' if clda < 1e-6 else colors[j], label=f"CLDA = {clda}")
plt.ylabel("Log$_{10}$ loss")
plt.xlabel("Ranked readout unit")
plt.legend()
plt.tight_layout()
despine()
plt.savefig(os.path.join(RESULTDIR, 'RankedUAC.png'))
