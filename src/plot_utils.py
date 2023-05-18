import matplotlib.pyplot as plt
from seaborn import despine, color_palette
import numpy as np
import torch

units_convert = {'cm': 1 / 2.54, 'mm': 1 / 2.54 / 10}
colors = [(200. / 255, 0, 0),  # red
                 (0.9, 0.6, 0),  # orange
                 (0.95, 0.9, 0.25),  # yellow
                 (0, 158. / 255, 115. / 255),  # bluish green
                 (86. / 255, 180. / 255, 233. / 255),  # sky_blue
                 (0, 0.45, 0.7),  # blue
                 (75. / 255, 0., 146. / 255),  # purple
                 (0.8, 0.6, 0.7)]  # pink


def plot_loss(list_of_dicts, outfile_name=None, errorbar_type='sem', subsampling=1, normalize=False):
    """
    Plot loss.

    :param list_of_dicts: [{CLDA: loss}, ..., {CLDA: loss}]
    e.g. [{0.1: [...]}, {0.: [...]}]
    :return:
    """
    figsize = (3, 3 / 1.25) if outfile_name is None else (45 * units_convert['mm'], 45 / 1.25 * units_convert['mm'])
    plt.figure(figsize=figsize)

    # Gather all the CLDA values in the list
    clda_values = set()
    no_clda_data_exist = False
    for ld in list_of_dicts:
        for k in ld.keys():
            if k < 1e-6:
                no_clda_data_exist = True
            clda_values.add(k)
    clda_values = np.sort(list(clda_values))

    if not no_clda_data_exist and normalize:
        normalize = False
        print("No losses for CLDA=0, `normalize` set to `False`")

    losses = {clda: [] for clda in clda_values}
    for ld in list_of_dicts:
        for k, v in ld.items():
            losses[k].append(np.log10(v))

    if normalize:
        if len(losses[clda_values[0]]) == 1:
            min_, max_ = np.min(losses[clda_values[0]]), np.max(losses[clda_values[0]])
        else:
            min_, max_ = np.min(losses[clda_values[0]], axis=1, keepdims=True), np.max(losses[clda_values[0]], axis=1, keepdims=True)
        for clda in clda_values:
            if len(losses[clda]) == 1:
                losses[clda] = (np.array(losses[clda]) - min_)/(max_ - min_)
            else:
                losses[clda] = (np.array(losses[clda]) - min_) / (max_ - min_)

    # Plot mean +/- error bar for each CLDA value
    for i, clda in enumerate(sorted(clda_values)):
        mean_loss = np.mean(losses[clda], axis=0)[::subsampling]
        if errorbar_type == 'sem':
            errorbar = np.std(losses[clda], axis=0, ddof=1) / len(losses[clda])**0.5
            errorbar = errorbar[::subsampling]
        elif errorbar == 'std':
            errorbar = np.std(losses[clda], axis=0, ddof=1)
            errorbar[::subsampling]
        plt.plot(np.arange(len(mean_loss)), mean_loss,
                     color='black' if clda < 1e-6 else colors[i], label=f"CLDA = {clda}", lw=0.5)
        plt.fill_between(np.arange(len(mean_loss)), mean_loss-errorbar, mean_loss+errorbar,
                         color='black' if clda < 1e-6 else colors[i], lw=0, alpha=0.5)
    if subsampling > 1:
        plt.xlabel(rf"Epochs ($\times${subsampling})")
    else:
        plt.xlabel("Epoch")

    if normalize:
        plt.ylabel("Relative log$_{10}$ loss")
    else:
        plt.ylabel("Log$_{10}$ loss")
    plt.legend(loc="upper right")
    despine()
    plt.tight_layout()
    if outfile_name is not None:
        plt.savefig(outfile_name)
    else:
        plt.show()


def plot_trajectories(ax, net, dataset, noise_for_hidden_init, n_reals=5):
    colors = color_palette('colorblind', dataset.n_targets)
    with torch.no_grad():
        for _ in range(n_reals):
            for i in range(dataset.n_targets):
                pred, _, _ = net(dataset[i][0].unsqueeze(0), noise_for_hidden_init * torch.rand((1, 1, net.network_size[2])))
                pred = pred.detach().numpy()
                pred_traj = pred[0, :, :2]
                target = dataset[i][1].detach().numpy()

                ax.plot(pred_traj[:, 0], pred_traj[:, 1], color=colors[i])
                ax.plot(target[0], target[1], color=colors[i], marker='o', markersize=3, lw=0.)
    ax.axis('off')
    ax.set_aspect('equal', 'box')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_speeds(ax, net, dataset, noise_for_hidden_init, n_reals=5):
    colors = color_palette('colorblind', dataset.n_targets)
    with torch.no_grad():
        for _ in range(n_reals):
            for i in range(dataset.n_targets):
                pred, _, _ = net(dataset[i][0].unsqueeze(0), noise_for_hidden_init * torch.rand((1, 1, net.network_size[2])))
                speeds = torch.linalg.vector_norm(pred[0, :, 2:4], dim=-1)
                ax.plot(dataset.dt * np.arange(speeds.size(0)), speeds, color=colors[i])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    despine(ax=ax)



def plot_single_loss(l, outfile_name=None):
    plt.figure(figsize=(45 * units_convert['mm'], 45 / 1.25 * units_convert['mm']))
    plt.semilogy(l)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    despine()
    plt.tight_layout()
    if outfile_name is not None:
        plt.savefig(outfile_name)