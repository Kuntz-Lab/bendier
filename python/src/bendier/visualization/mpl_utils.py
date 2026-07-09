import os

import matplotlib.pyplot as plt


def setup_plt(width=3.0, height=5.0, grid=False):
    os.makedirs("output/figures", exist_ok=True)
    plt.rcParams.update({
        "figure.figsize": (width, height),
        "font.family": "STIXGeneral",
        "font.size": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": grid,
        "grid.alpha": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
        "mathtext.rm": "stix",
        "lines.markersize": 4,
    })
