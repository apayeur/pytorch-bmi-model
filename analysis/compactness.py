import torch
import numpy as np
import sys
import os
from networks import NoisyNetWithFeedback, NoisyNet
from effectors import VelocityIntegrator
from objectives import EndLoss
from datasets import EndLossDataset
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

seeds = list(range(1, 11))

DIR = "bci-with-feedback-SGD-nonstop"
BCI_DATADIR = os.path.join("../data/", DIR)
RESULTDIR = os.path.join("../results", DIR)
if not os.path.exists(RESULTDIR):
    os.makedirs(RESULTDIR)

cldas = [0.0, 0.1, 0.5]
single_unit_ranking = {clda: [] for clda in cldas}
single_unit_ranking_losses = {clda: [] for clda in cldas}
ranked_uac = {clda: [] for clda in cldas}

for clda in cldas:
    for seed in seeds:
        # Load model and parameters
        params_manual = np.load(os.path.join(BCI_DATADIR, f"params_from_manual_seed{seed}_clda{clda}.npy"), allow_pickle=True).item()
        params_bci = np.load(os.path.join(BCI_DATADIR, f"params_seed{seed}_clda{clda}.npy"), allow_pickle=True).item()
        net = NoisyNetWithFeedback(network_size=params_manual['size'], nonlinearity=params_manual['nonlinearity'], delay=params_manual['delay'])
        net.effector = VelocityIntegrator()
        net.load_state_dict(torch.load(os.path.join(BCI_DATADIR, f"model_seed{seed}_clda{clda}.pth")))
        net.eval()

        # Compute single-unit ranking
        n_readouts = params_bci['n_readouts']
        D = copy.deepcopy(net.readout_layer.weight)

        loss_fn = EndLoss(params_manual['gamma_v'], 0., 0., params_manual['dt'])
        dataset = EndLossDataset(n_targets=params_manual['n_targets'], total_duration=params_manual['total_duration'],
                                 dt=params_manual['dt'],
                                 distance=0.07, context='bci')
        dataloader = DataLoader(dataset, batch_size=params_manual['n_targets'])
        test_loss = np.empty(n_readouts)

        for readout_i in range(n_readouts):
            net.readout_layer.weight.data = copy.deepcopy(D.data)
            net.readout_layer.weight.data[:, :readout_i] = 0.
            net.readout_layer.weight.data[:, readout_i+1:] = 0.
            with torch.no_grad():
                for X, y in dataloader:
                    h0 = params_manual['noisy_ics'] * torch.rand(
                        (1, dataloader.batch_size, net.network_size[2]))  # set of initial conditions for motor cortex
                    pred, h, controls = net(X, h0)
                    print("min h", torch.min(h))
                    print("max h", torch.max(h))
                    test_loss[readout_i], _ = loss_fn(pred, y.unsqueeze(1), controls)
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
                    h0 = params_manual['noisy_ics'] * torch.rand(
                        (1, dataloader.batch_size, net.network_size[2]))
                    pred, _, controls = net(X, h0)
                    test_loss[i], _ = loss_fn(pred, y.unsqueeze(1), controls)
        ranked_uac[clda].append(test_loss)


# Plot ranked single-unit loss
plt.figure(figsize=(45 * units_convert['mm'], 45 / 1.25 * units_convert['mm']))
for j, clda in enumerate(cldas):
    m = np.mean(np.log10(single_unit_ranking_losses[clda]), axis=0)
    sem = np.std(np.log10(single_unit_ranking_losses[clda]), axis=0, ddof=1) / len(seeds)**0.5
    plt.errorbar(np.arange(1, 1 + n_readouts), m, yerr=sem,
                 color='black' if clda < 1e-6 else colors[j], label=f"CLDA = {clda}")
plt.ylabel("Log loss")
plt.xlabel("Ranked readout unit")
plt.legend()
plt.tight_layout()
despine()
plt.show()

# Plot ranked UAC
plt.figure(figsize=(45 * units_convert['mm'], 45 / 1.25 * units_convert['mm']))
for j, clda in enumerate(cldas):
    m = np.mean(np.log10(ranked_uac[clda]), axis=0)
    sem = np.std(np.log10(ranked_uac[clda]), axis=0, ddof=1) / len(seeds)**0.5
    plt.errorbar(np.arange(1, 1 + n_readouts), m, yerr=sem,
                 color='black' if clda < 1e-6 else colors[j], label=f"CLDA = {clda}")
plt.ylabel("Log loss")
plt.xlabel("Ranked readout unit")
plt.legend()
plt.tight_layout()
despine()
plt.show()