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
    def __init__(self, hyperparam_v, hyperparam_f, dt):
        super().__init__()
        self.hyperparam_v, self.hyperparam_f, self.dt = hyperparam_v, hyperparam_f, dt
        self.non_dimensionalizer = torch.Tensor(
            [[[0.01, 0.01, 0.02, 0.02, 0.08, 0.08]]])  # to rescale velocity and acceleration components of loss
        self.hyperparam_vec = torch.Tensor([[[1., 1.,
                                              hyperparam_v**0.5, hyperparam_v**0.5,
                                              hyperparam_f**0.5, hyperparam_f**0.5]]])

    def forward(self, prediction, target):
        prediction = prediction / self.non_dimensionalizer * self.hyperparam_vec
        target = target / self.non_dimensionalizer * self.hyperparam_vec

        loss = torch.mean((prediction[:, -1, :] - target.squeeze())**2)

        return loss
