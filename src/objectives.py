import torch
import torch.nn as nn


class HoldsLoss(nn.Module):
    def __init__(self, hyperparam_v, hyperparam_f, hold_start, hold_end, dt):
        super().__init__()
        self.hyperparam_v, self.hyperparam_f = hyperparam_v, hyperparam_f
        self.hold_start, self.hold_end, self.dt = hold_start, hold_end, dt
        self.non_dimensionalizer = torch.Tensor(
            [[[0.01, 0.01, 0.02, 0.02, 0.08, 0.08]]])  # to rescale velocity and acceleration components of loss
        self.hyperparam_vec = torch.Tensor([[[1., 1.,
                                              hyperparam_v**0.5, hyperparam_v**0.5,
                                              hyperparam_f**0.5, hyperparam_f**0.5]]])

    def forward(self, prediction, target):
        loss = 0.
        prediction = prediction / self.non_dimensionalizer * self.hyperparam_vec
        target = target / self.non_dimensionalizer * self.hyperparam_vec

        # Hold at the center
        initial_target = torch.zeros_like(prediction[:, :int(self.hold_start/self.dt), :])
        loss += torch.mean((prediction[:, :int(self.hold_start/self.dt), :] - initial_target)**2)

        # Hold at the periphery
        loss += torch.mean((prediction[:, -int(self.hold_end/self.dt):, :] - target)**2)

        return loss


class EndLoss(nn.Module):
    r"""
    Objective function:

    :math:`L = \|x_T - d\|^2/\delta_p^2 + \gamma_v \|\dot{x}_T\|^2/\delta_v^2
    + \gamma_f \|f_T\|^2/\delta_f^2 + r \sum_{t=0}^{T-1} \|u_t\|^2/\delta_f^2`

    where

    :math:`x_T, \dot{x}_T, f_T` : final position, velocity and force/acceleration

    :math:`d`: target position

    :math:`\delta_p, \delta_v, \delta_f`: factors to non-dimensionalize and rescale the loss components


    """
    def __init__(self, hyperparam_v, hyperparam_f, hyperparam_ctrl, hyperparam_rate, dt, workspace_dim=2):
        super().__init__()
        self.hyperparam_v, self.hyperparam_f, self.dt = hyperparam_v, hyperparam_f, dt
        self.hyperparam_rate = hyperparam_rate
        self.hyperparam_ctrl = hyperparam_ctrl
        self.workspace_dim = workspace_dim
        if workspace_dim == 2:
            self.non_dimensionalizer = torch.Tensor(
                [[[0.01, 0.01, 0.02, 0.02, 0.08, 0.08]]])  # to rescale velocity and acceleration components of loss
            self.hyperparam_vec = torch.Tensor([[[1., 1.,
                                                  hyperparam_v**0.5, hyperparam_v**0.5,
                                                  hyperparam_f**0.5, hyperparam_f**0.5]]])
        elif workspace_dim == 3:
            self.non_dimensionalizer = torch.Tensor(
                [[[0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.08, 0.08, 0.08]]])
            self.hyperparam_vec = torch.Tensor([[[1., 1., 1.,
                                                  hyperparam_v ** 0.5, hyperparam_v ** 0.5, hyperparam_v ** 0.5,
                                                  hyperparam_f ** 0.5, hyperparam_f ** 0.5, hyperparam_f ** 0.5]]])

    def forward(self, prediction, target, controls, activities):
        loss = 0.
        loss_ctrl = self.hyperparam_ctrl * torch.mean(controls**2) / self.non_dimensionalizer[0, 0, -1]**2
        loss += loss_ctrl

        loss_rate = self.hyperparam_rate * torch.mean(activities**2)
        loss += loss_rate

        prediction = prediction / self.non_dimensionalizer * self.hyperparam_vec
        target = target / self.non_dimensionalizer * self.hyperparam_vec

        loss += torch.mean((prediction[:, -1, :] - target.squeeze())**2)
        #loss += torch.mean((prediction[:, -1, :self.workspace_dim] - target.squeeze()[:, :self.workspace_dim])**2)
        #loss += torch.mean((prediction[:, -2, self.workspace_dim:2*self.workspace_dim]
        #                    - target.squeeze()[:, self.workspace_dim:2*self.workspace_dim]) ** 2)
        #loss += torch.mean((prediction[:, -3, 2*self.workspace_dim:] - target.squeeze()[:, 2*self.workspace_dim:]) ** 2)

        return loss, loss_ctrl, loss_rate