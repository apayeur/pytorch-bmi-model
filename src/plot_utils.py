import matplotlib.pyplot as plt
from seaborn import despine

units_convert = {'cm': 1 / 2.54, 'mm': 1 / 2.54 / 10}
colors = [(200. / 255, 0, 0),  # red
                 (0.9, 0.6, 0),  # orange
                 (0.95, 0.9, 0.25),  # yellow
                 (0, 158. / 255, 115. / 255),  # bluish green
                 (86. / 255, 180. / 255, 233. / 255),  # sky_blue
                 (0, 0.45, 0.7),  # blue
                 (75. / 255, 0., 146. / 255),  # purple
                 (0.8, 0.6, 0.7)]  # pink


def plot_loss(list_of_dicts, outfile_name=None):
    """
    Plot loss.

    :param list_of_dicts: [{CLDA: loss}, ..., {CLDA: loss}]
    e.g. [{0.1: [...]}, {0.: [...]}]
    :return:
    """
    figsize = (3, 3 / 1.25) if outfile_name is None else (45 * units_convert['mm'], 45 / 1.25 * units_convert['mm'])
    plt.figure(figsize=figsize)
    for i, loss in enumerate(list_of_dicts):
        clda = list(loss.keys())[0]
        val = list(loss.values())[0]
        plt.semilogy(val, color='black' if clda < 1e-6 else colors[i], label=f"CLDA = {clda}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    despine()
    plt.tight_layout()
    if outfile_name is not None:
        plt.savefig(outfile_name)
    else:
        plt.show()