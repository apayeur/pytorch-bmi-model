import torch
from torch.utils.data import Dataset
import numpy as np


class GaussianVelocityDataset(Dataset):
    """
    Inputs are 5-dimensional:
        - 2 dimensions for the target information
        - 2 dimensions for the context
        - 1 dimension for the hold signal
    Targets are 2-dimensional, each dimension being Gaussian velocity profile (for x and y directions)
    """
    def __init__(self, n_targets=8, total_duration=1.5, dt=0.01, distance=0.07,
                hold_start=0.25, hold_end=0.25, sigma=0.1, context='arm'):
        self.n_targets = n_targets
        self.total_duration = total_duration  # in seconds
        self.dt = dt  # in seconds
        self.distance = distance  # in meters
        self.hold_start = hold_start  # in seconds
        self.hold_end = hold_end  # in seconds
        self.sigma = sigma  # in seconds
        self.peak_velocity = self.distance / (2. * np.pi)**0.5 / sigma
        self.context = context

    def __len__(self):
        return self.n_targets

    def construct_input(self, idx):
        L = int(self.total_duration / self.dt)
        x = torch.zeros((L, 5),
                        dtype=torch.float32)  # 2 cols for context, 2 cols for target position, 1 col for hold signal
        x[:, 0] = np.cos(2 * np.pi * idx / self.n_targets)  # self.distance * np.cos(2*np.pi*idx/self.n_targets)
        x[:, 1] = np.sin(2 * np.pi * idx / self.n_targets)  # self.distance * np.sin(2*np.pi*idx/self.n_targets)
        x[:, 2] = 1. if self.context == 'arm' else 0.
        x[:, 3] = 0. if self.context == 'arm' else 1.
        x[:, 4] = 0.
        x[:int(self.hold_start / self.dt), 4] = -1.
        x[-int(self.hold_end / self.dt), 4] = -1.
        return x

    def __getitem__(self, idx):
        # input
        x = self.construct_input(idx)

        # output
        L = int(self.total_duration / self.dt)
        v = torch.zeros((L, 2), dtype=torch.float32)
        t = self.dt * torch.arange(0, L)
        v[:, 0] = self.peak_velocity * np.cos(2*np.pi*idx/self.n_targets) * torch.exp(-(t - self.total_duration/2)**2/2/self.sigma**2)
        v[:, 1] = self.peak_velocity * np.sin(2*np.pi*idx/self.n_targets) * torch.exp(-(t - self.total_duration/2)**2/2/self.sigma**2)
        v[:int(self.hold_start / self.dt), :] = 0.
        v[-int(self.hold_end / self.dt), :] = 0.

        return x, v


class HoldsDataset(GaussianVelocityDataset):
    def __init__(self, n_targets=8, total_duration=1.5, dt=0.01, distance=0.07,
                hold_start=0.25, hold_end=0.25, sigma=0.1, context='arm'):
        super().__init__(n_targets=n_targets, total_duration=total_duration, dt=dt, distance=distance,
                         hold_start=hold_start, hold_end=hold_end, sigma=sigma, context=context)

    def __len__(self):
        return self.n_targets

    def __getitem__(self, idx):
        # input
        x = super().construct_input(idx)

        # output
        target = torch.zeros((3*2), dtype=torch.float32)
        target[0], target[1] = self.distance * np.cos(2*np.pi*idx/self.n_targets), self.distance * np.sin(2*np.pi*idx/self.n_targets)
        target[2:] = 0.
        return x, target


class EndLossDataset(Dataset):
    def __init__(self, n_targets=8, total_duration=1.5, dt=0.01, distance=0.07, context='arm'):
        self.n_targets = n_targets
        self.total_duration = total_duration  # in seconds
        self.dt = dt  # in seconds
        self.distance = distance  # in meters
        self.context = context

    def __len__(self):
        return self.n_targets

    def __getitem__(self, idx):
        # input
        L = int(self.total_duration / self.dt)
        x = torch.zeros((L, 4),
                        dtype=torch.float32)  # 2 cols for context, 2 cols for target position
        x[:, 0] = np.cos(2 * np.pi * idx / self.n_targets)  # self.distance * np.cos(2*np.pi*idx/self.n_targets)
        x[:, 1] = np.sin(2 * np.pi * idx / self.n_targets)  # self.distance * np.sin(2*np.pi*idx/self.n_targets)
        x[:, 2] = 1. if self.context == 'arm' else 0.
        x[:, 3] = 0. if self.context == 'arm' else 1.

        # output
        target = torch.zeros((3*2), dtype=torch.float32)
        target[0], target[1] = self.distance * np.cos(2*np.pi*idx/self.n_targets), self.distance * np.sin(2*np.pi*idx/self.n_targets)
        target[2:] = 0.
        return x, target