import torch.nn as nn
import torch
import math


class NoEffector(nn.Module):
    def __init__(self, workspace_dim=2):
        super().__init__()
        self.workspace_dim = workspace_dim
        self.layer = nn.Identity()

    def forward(self, controls, state_prev=None):
        return self.layer(controls)


class PointMassArm(nn.Module):
    def __init__(self, workspace_dim=2, tau_f=0.04, m=1., dt=0.01, reset_radius=0.):
        super().__init__()
        self.workspace_dim = workspace_dim  # dimension of the workspace
        self.reset_radius = reset_radius      # noise in the initial position of the arm

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

    def forward(self, controls, states_prev=None):
        """
        Parameters:
        ----------
        controls: shape = (N, L, D), where N = nb of batches, L = sequence length, D = dimension of control
        """
        if controls.dim() == 3:
            all_states = torch.empty((controls.size(0), controls.size(1), 3 * self.workspace_dim))
            state = torch.zeros((controls.size(0), 3 * self.workspace_dim))
            if self.workspace_dim == 2:
                state[:, 0] = self.reset_radius * torch.cos(2 * math.pi * torch.rand(controls.size(0)))
                state[:, 1] = self.reset_radius * torch.sin(2 * math.pi * torch.rand(controls.size(0)))
            elif self.workspace_dim == 3:
                state[:, 0] = self.reset_radius * torch.sin(math.pi * torch.rand(controls.size(0))) * torch.cos(2 * math.pi * torch.rand(controls.size(0)))
                state[:, 1] = self.reset_radius * torch.sin(math.pi * torch.rand(controls.size(0))) * torch.sin(2 * math.pi * torch.rand(controls.size(0)))
                state[:, 2] = self.reset_radius * torch.cos(math.pi * torch.rand(controls.size(0)))
            all_states[:, 0, :] = state
            for t in range(1, controls.size(1)):
                combined = torch.cat((state, controls[:, t, :]), -1)  # TODO: should be t-1
                state = self.lin(combined)
                all_states[:, t, :] = state

        elif controls.dim() == 2:
            # controls is of shape (N, D) and state_prev is of shape (N, 3*`self.workspace_dim)
            assert states_prev.shape[0] == controls.shape[0] and states_prev.shape[1] == 3*self.workspace_dim, \
                f"`state_prev` has shape {states_prev.shape} but should be ({controls.shape[0]}, 3*self.workspace_dim)"
            combined = torch.cat((states_prev, controls), -1)
            all_states = self.lin(combined)
        return all_states


class VelocityIntegrator(nn.Module):
    def __init__(self, workspace_dim=2, dt=0.01, reset_radius=0.):
        super().__init__()
        self.workspace_dim = workspace_dim  # dimension of the workspace
        self.reset_radius = reset_radius  # noise in the initial position of the arm

        self.lin = nn.Linear(workspace_dim, 3 * workspace_dim, bias=False)

        if workspace_dim == 2:
            A = torch.Tensor([[1., 0., 0., 0., 0., 0.],
                              [0., 1., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.]])
            B = torch.Tensor([[dt, 0.],
                              [0., dt],
                              [1., 0.],
                              [0., 1.],
                              [0., 0.],
                              [0., 0.]])
        elif workspace_dim == 3:
            A = torch.Tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
            B = torch.Tensor([[dt, 0., 0.],
                              [0., dt, 0.],
                              [0., 0., dt],
                              [1., 0., 0.],
                              [0., 1., 0.],
                              [0., 0., 1.],
                              [0., 0., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]])

        # Initialize weights
        self.lin.weight = nn.Parameter(torch.cat((A, B), dim=1))
        self.lin.requires_grad_(False)

    def forward(self, velocities, states_prev=None):
        """
        Parameters:
        ----------
        controls: shape = (N, L, D), where N = nb of batches, L = sequence length, D = dimension of control
        """
        if velocities.dim() == 3:
            all_states = torch.empty((velocities.size(0), velocities.size(1), 3 * self.workspace_dim))
            state = torch.zeros((velocities.size(0), 3 * self.workspace_dim))
            if self.workspace_dim == 2:
                state[:, 0] = self.reset_radius * torch.cos(2 * math.pi * torch.rand(velocities.size(0)))
                state[:, 1] = self.reset_radius * torch.sin(2 * math.pi * torch.rand(velocities.size(0)))
            elif self.workspace_dim == 3:
                state[:, 0] = self.reset_radius * torch.sin(math.pi * torch.rand(velocities.size(0))) * torch.cos(2 * math.pi * torch.rand(velocities.size(0)))
                state[:, 1] = self.reset_radius * torch.sin(math.pi * torch.rand(velocities.size(0))) * torch.sin(2 * math.pi * torch.rand(velocities.size(0)))
                state[:, 2] = self.reset_radius * torch.cos(math.pi * torch.rand(velocities.size(0)))
            all_states[:, 0, :] = state
            for t in range(1, velocities.size(1)):
                combined = torch.cat((state, velocities[:, t, :]), -1)
                state = self.lin(combined)
                all_states[:, t, :] = state

        elif velocities.dim() == 2:
            assert states_prev.shape[0] == velocities.shape[0] and states_prev.shape[1] == 3 * self.workspace_dim, \
                f"`state_prev` has shape {states_prev.shape} but should be ({velocities.shape[0]}, 3*self.workspace_dim)"
            combined = torch.cat((states_prev, velocities), -1)
            all_states = self.lin(combined)
        return all_states



