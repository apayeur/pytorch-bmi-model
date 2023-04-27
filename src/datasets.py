import torch
from torch.utils.data import Dataset
import numpy as np


class GaussianVelocityDataset(Dataset):
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

    def __getitem__(self, idx):
        # input
        L = int(self.total_duration / self.dt)
        x = torch.zeros((L, 5), dtype=torch.float32)  # 2 cols for context, 2 cols for target position, 1 col for hold signal
        x[:, 0] = np.cos(2*np.pi*idx/self.n_targets)  # self.distance * np.cos(2*np.pi*idx/self.n_targets)
        x[:, 1] = np.sin(2*np.pi*idx/self.n_targets)  # self.distance * np.sin(2*np.pi*idx/self.n_targets)
        x[:, 2] = 1. if self.context == 'arm' else 0.
        x[:, 3] = 0. if self.context == 'arm' else 1.
        x[:, 4] = 0.
        x[:int(self.hold_start / self.dt), 4] = -1.
        x[-int(self.hold_end / self.dt), 4] = -1.

        # output
        v = torch.zeros((L, 2), dtype=torch.float32)
        t = self.dt * torch.arange(0, L)
        v[:, 0] = self.peak_velocity * np.cos(2*np.pi*idx/self.n_targets) * torch.exp(-(t - self.total_duration/2)**2/2/self.sigma**2)
        v[:, 1] = self.peak_velocity * np.sin(2*np.pi*idx/self.n_targets) * torch.exp(-(t - self.total_duration/2)**2/2/self.sigma**2)
        v[:int(self.hold_start / self.dt), :] = 0.
        v[-int(self.hold_end / self.dt), :] = 0.

        return x, v

