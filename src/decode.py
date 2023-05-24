import torch
from sklearn.linear_model import LinearRegression
import numpy as np
import copy


def build_bci_decoder(net, dataloader, noisy_ics, n_readouts=10, hold=0, clda=0.):
    dynamics_after_clda = None
    dynamics_before_clda = None
    with torch.no_grad():
        for X, y in dataloader:
            h0 = noisy_ics * torch.randn(
                (1, dataloader.batch_size, net.network_size[2]))  # set of initial conditions for motor cortex
            dyn, h, _ = net(X, h0)
            dynamics_before_clda = copy.deepcopy(dyn)
    if clda > 0.:
        dyn = rotate_velocities(dyn, y, hold=hold)
        dynamics_after_clda = dyn
    h = torch.reshape(h, (-1, h.size(2)))
    dyn = dyn[:, :, 2:4]  # only using the velocities
    dyn = torch.reshape(dyn, (-1, dyn.size(2)))

    # perform regression
    lin_reg = LinearRegression(fit_intercept=False)
    lin_reg.fit(h[:, :n_readouts].numpy(), dyn.numpy())
    T = lin_reg.coef_

    # define readout matrix
    R = np.zeros((n_readouts, net.network_size[2]), dtype=np.float32)  # tensor will have to be a float, not a double
    R[range(n_readouts), range(n_readouts)] = 1.

    # print max activity for each readout
    #if clda < 1e-6:
    #    print("Mean activity ", np.mean(R @ h.numpy().T, axis=1))
    #    print("Max activity ", np.max(R @ h.numpy().T, axis=1))
    #    print("Max activity = ", torch.max(h.detach()))
    return T @ R, dynamics_before_clda, dynamics_after_clda


def rotate_velocities(dynamics, targets, hold=0):
    workspace_dim = 2 if dynamics.shape[-1] == 6 else 3

    speed = torch.linalg.vector_norm(dynamics[:, :, workspace_dim:2 * workspace_dim], dim=-1, keepdim=True)
    #if hold > 0:
    #    speed[:, :hold] = 0.
    #    speed[:, -hold:] = 0.

    target_positions = targets[:, :workspace_dim]
    unit_vectors_towards_targets = (target_positions.unsqueeze(1) - dynamics[:, :, :workspace_dim]) / \
                                   torch.linalg.vector_norm(
                                       target_positions.unsqueeze(1) - dynamics[:, :, :workspace_dim], dim=-1,
                                       keepdim=True)

    if hold > 0:
        center_position = torch.zeros_like(target_positions)
        unit_vectors_towards_targets[:, :hold, :] = (center_position.unsqueeze(1) - dynamics[:, :hold, :workspace_dim]) / \
                                   torch.linalg.vector_norm(
                                       center_position.unsqueeze(1) - dynamics[:, :hold, :workspace_dim], dim=-1,
                                       keepdim=True)

    dynamics[:, :, workspace_dim:2 * workspace_dim] = speed * unit_vectors_towards_targets

    return dynamics
