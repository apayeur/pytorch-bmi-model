import torch.nn as nn
import torch
import math


class PointMassArm(nn.Module):
    def __init__(self, workspace_dim=2, tau_f=0.04, m=1., dt=0.01, reset_noise=0.):
        super().__init__()
        self.workspace_dim = workspace_dim  # dimension of the workspace
        self.reset_noise = reset_noise      # noise in the initial position of the arm

        self.lin = nn.Linear(workspace_dim + 3 * workspace_dim, 3 * workspace_dim, bias=False)

        if workspace_dim == 2:
            self.A = torch.Tensor([[1., 0., dt, 0., 0., 0.],
                                   [0., 1., 0., dt, 0., 0.],
                                   [0., 0., 1., 0., dt/m, 0.],
                                   [0., 0., 0., 1., 0., dt/m],
                                   [0., 0., 0., 0., math.exp(-dt/tau_f), 0.],
                                   [0., 0., 0., 0., 0., math.exp(-dt/tau_f)]])
            self.B = torch.Tensor([[0., 0.],
                                   [0., 0.],
                                   [0., 0.],
                                   [0., 0.],
                                   [1., 0.],
                                   [0., 1.]])
        elif workspace_dim == 3:
            self.A = torch.Tensor([[1., 0., 0., dt, 0., 0., 0., 0., 0.],
                                   [0., 1., 0., 0., dt, 0., 0., 0., 0.],
                                   [0., 0., 1., 0., 0., dt, 0., 0., 0.],
                                   [0., 0., 0., 1., 0., 0., dt / m, 0., 0.],
                                   [0., 0., 0., 0., 1., 0., 0., dt / m, 0.],
                                   [0., 0., 0., 0., 0., 1., 0., 0., dt / m],
                                   [0., 0., 0., 0., 0., 0., math.exp(-dt / tau_f), 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., math.exp(-dt / tau_f), 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., math.exp(-dt / tau_f)]])
            self.B = torch.Tensor([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [1., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 0., 1.]])

        # Initialize weights
        self.lin.weight = nn.Parameter(torch.cat((self.A, self.B), dim=1))
        self.lin.requires_grad_(False)

    def forward(self, controls):
        """
        Parameters:
        ----------
        controls: shape = (N, L, D), where N = nb of batches, L = sequence length, D = dimension of control
        """
        all_states = torch.empty((controls.size(0), controls.size(1), 3 * self.workspace_dim))
        state = self.reset_noise * torch.randn((controls.size(0), 3 * self.workspace_dim))
        all_states[:, 0, :] = state
        for t in range(1, controls.size(1)):
            combined = torch.cat((state, controls[:, t, :]), -1)
            state = self.lin(combined)
            all_states[:, t, :] = state
        return all_states



