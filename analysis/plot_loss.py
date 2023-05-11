import matplotlib.pyplot as plt
import numpy as np
from plot_utils import units_convert, plot_loss, colors
from seaborn import despine
import os
plt.style.use('../plot_params.dms')


dir_name = "bci-with-feedback-noisy-reset-delay-cldastart0-cldastopFalse-cldafreq10-test"
cldas = [0.0, 0.1]
n_seeds = 1
list_of_losses = []
for seed in range(1, 1+n_seeds):
    for clda in cldas:
        try:
            loss = np.load(f"../data/{dir_name}/loss_seed{seed}_clda{clda}.npy", allow_pickle=True)
            list_of_losses.append({clda: loss})
        except FileNotFoundError:
            print(f"File ../data/{dir_name}/loss_seed{seed}_clda{clda}.npy not found.")

# Plot loss
plot_loss(list_of_losses, os.path.join('../results', dir_name, 'Loss.png'), subsampling=100)


# Plot integrated differences between losses
list_of_integrated_loss = []
for seed in range(1, 1+n_seeds):
    for clda in cldas[1:]:
        try:
            loss = np.load(f"../data/{dir_name}/loss_seed{seed}_clda{clda}.npy", allow_pickle=True)
            loss_no_clda = np.load(f"../data/{dir_name}/loss_seed{seed}_clda0.0.npy", allow_pickle=True)
            list_of_integrated_loss.append({clda: np.cumsum(np.log10(loss) - np.log10(loss_no_clda))})
        except FileNotFoundError:
            print(f"File not found.")

plt.figure(figsize=(45 * units_convert['mm'], 45 / 1.25 * units_convert['mm']))

# Gather all the CLDA values in the list
losses = {clda: [] for clda in cldas[1:]}
for ld in list_of_integrated_loss:
    for k, v in ld.items():
        losses[k].append(v)

# Plot mean +/- error bar for each CLDA value
for i, clda in enumerate(sorted(cldas[1:])):
    mean_loss = np.mean(losses[clda], axis=0)
    errorbar = np.std(losses[clda], axis=0, ddof=1) / len(losses[clda])**0.5

    plt.plot(np.arange(len(mean_loss)), mean_loss,
                 color='black' if clda < 1e-6 else colors[i], label=f"CLDA = {clda}", lw=0.5)
    plt.fill_between(np.arange(len(mean_loss)), mean_loss-errorbar, mean_loss+errorbar,
                     color='black' if clda < 1e-6 else colors[i], lw=0, alpha=0.5)
plt.plot(np.arange(len(mean_loss)), np.zeros(len(mean_loss)), ':', color='grey')
plt.xlabel("Epoch")
plt.ylabel("Integrated difference\nof log losses")
plt.legend()
despine()
plt.tight_layout()
plt.savefig(f'../results/{dir_name}/IntegratedLogLoss.png')
