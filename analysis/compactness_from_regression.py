import torch
import numpy as np
import sys
import os
from networks import NoisyNetWithFeedback, NoisyNet
from effectors import VelocityIntegrator
from objectives import EndLoss, HoldsLoss
from datasets import EndLossDataset, HoldsDataset
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import matplotlib.pyplot as plt
from pyts.approximation import PiecewiseAggregateApproximation
from seaborn import despine
from plot_utils import units_convert, colors
import copy
sys.path.append("../data")
plt.style.use('../plot_params.dms')

"""
Description:
-----------
Analyze compactness in the model after learning using target encoding.
"""


def piecewise_aggregate_approximation(activity, window_size=10):
    """
    The time series is divided into a number of segments
    and each segment is replaced by the average of its data points.

    Parameters:
    ----------
    activity : 3D array with shape (n_targets, n_timebins, n_units)
      Activity (firing rate) of all hidden neurons for a batch.
    window_size : int
      Size of the window to perform average

    Return:
    ------
    paa_activity : 3D array, shape (n_targets, n_timebins/window_size, n_units)
    """
    n_targets, n_timebins, n_units = activity.shape
    assert n_timebins % window_size == 0, "`n_timebins` must be an integer multiple of `window_size`"
    paa_activity = np.empty((n_targets, n_timebins//window_size, n_units))

    paa = PiecewiseAggregateApproximation(window_size=window_size)
    for i in range(n_targets):
        X_transformed = paa.transform(activity[i].T)
        paa_activity[i] = X_transformed.T
    return paa_activity


def convert_target_position_to_id(targets, nb_of_ids=8):
    """
    Convert targets in workspace coordinates to target ids.
    E.g.: t = [0,0,0,0,0,0] -> 0

    Parameters:
    ----------
    targets : 2D array-like, shape (n_targets, 3*workspace_dim)
      Targets in the format [p_x, p_y, v_x, v_y, f_x, f_y], for a 2D workspace.
    nb_of_ids : int
      Number of different targets

    Return:
    ------
    target_ids : 2D array-like, shape (n_targets, )
    """
    # Compute distance to peripheral target and normalize
    workspace_dim = targets.shape[1] // 3
    targets = targets[:, :workspace_dim]
    distance = (targets[0, 0]**2 + targets[0, 1]**2)**0.5
    targets /= distance

    # Reference target positions
    reference_targets = [[np.cos(2*np.pi*i/nb_of_ids), np.sin(2*np.pi*i/nb_of_ids)] for i in range(nb_of_ids)]

    target_ids = np.empty(targets.shape[0], dtype=int)
    for i, target in enumerate(targets):
        for r_id, r in enumerate(reference_targets):
            if np.all(np.isclose(target, r)):
                target_ids[i] = r_id
                break
    return target_ids


n_seeds = 1
seeds = list(range(1, 1+n_seeds))
cldas = [0.5]  # [0.0, 0.1, 0.2, 0.5]

n_reals = 7  # number of realizations to use
bin_width = 0.1

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

            # Populate arrays for logistic regression
            window_size = int(bin_width / params_manual['dt'])
            n_timebins = int(params_manual['total_duration'] / params_manual['dt'])
            with torch.no_grad():
                H = np.empty((n_reals, dataloader.batch_size, n_timebins//window_size,  n_readouts))
                T = np.empty((n_reals, dataloader.batch_size), dtype=int)
                for r in range(n_reals):
                    for X, y in dataloader:
                        h0 = params_manual['noisy_ics'] * torch.rand(
                            (1, dataloader.batch_size, net.network_size[2]))  # set of initial conditions for motor cortex
                        pred, h, controls = net(X, h0)
                        h = h.numpy()
                        H[r] = piecewise_aggregate_approximation(h[:, :, :n_readouts], window_size=window_size)
                        T[r] = convert_target_position_to_id(y.numpy(), params_manual['n_targets'])

            # Construct dataset for skfolds
            X_fold = np.empty((n_reals * dataloader.batch_size, n_timebins // window_size))
            Y_fold = np.empty(n_reals * dataloader.batch_size)
            sample_count = 0
            for r in range(n_reals):
                for t in range(dataloader.batch_size):
                    X_fold[sample_count] = H[r, t, :, 0]
                    Y_fold[sample_count] = T[r, t]
                    sample_count += 1

            skfolds = StratifiedKFold(n_splits=2)
            for train_index, test_index in skfolds.split(X_fold, Y_fold):
                # Single-unit ranking
                scores = []
                for readout_i in range(n_readouts):
                    X = np.empty((n_reals * dataloader.batch_size, n_timebins // window_size))
                    Y = np.empty(n_reals * dataloader.batch_size)
                    sample_count = 0
                    for r in range(n_reals):
                        for t in range(dataloader.batch_size):
                            X[sample_count] = H[r, t, :, readout_i]
                            Y[sample_count] = T[r, t]
                            sample_count += 1
                    X_train, Y_train = X[train_index], Y[train_index]
                    X_test, Y_test = X[test_index], Y[test_index]

                    log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
                    log_reg.fit(X_train, Y_train)
                    scores.append(log_reg.score(X_test, Y_test))

                ranked_units = np.argsort(scores)[::-1]
                print(ranked_units)
                single_unit_ranking[clda].append(ranked_units)
                single_unit_ranking_losses[clda].append(np.array(scores)[ranked_units])
                print(np.array(scores)[ranked_units])


            # (could use reshape, but I don't trust data will all be correctly aligned)
            X = np.empty((n_reals*dataloader.batch_size, n_timebins//window_size * n_readouts))
            Y = np.empty(n_reals*dataloader.batch_size)






            '''
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
                        pred, h, controls = net(X, h0)
                        test_loss[i], _, _ = loss_fn(pred, y.unsqueeze(1), controls, h)
            ranked_uac[clda].append(test_loss)
            '''
        except FileNotFoundError:
            print("File not found")

'''
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
'''