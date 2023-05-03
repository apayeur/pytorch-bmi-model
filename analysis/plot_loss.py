import matplotlib.pyplot as plt
import numpy as np
from plot_utils import units_convert, plot_loss
plt.style.use('../plot_params.dms')

# Import data
list_of_losses = []
for clda in [0.0, 0.1, 0.5, 1.0]:
    loss = np.load(f"../data/loss_clda{clda}_seed1.npy", allow_pickle=True)
    list_of_losses.append({clda: loss})

plot_loss(list_of_losses, '../results/Loss.png')