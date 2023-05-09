import matplotlib.pyplot as plt
import numpy as np
from plot_utils import units_convert, plot_loss
plt.style.use('../plot_params.dms')

# Import data
dir_name = "bci-with-feedback-SGD-nonstop"
list_of_losses = []
for seed in range(1, 11):
    for clda in [0.0, 0.1, 0.5]:
        loss = np.load(f"../data/{dir_name}/loss_seed{seed}_clda{clda}.npy", allow_pickle=True)
        list_of_losses.append({clda: loss})

plot_loss(list_of_losses, f'../results/{dir_name}/Loss.png')