"""Shared matplotlib style, model identity maps, and save helper."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

# One fixed color per model (Okabe-Ito, colorblind-safe). QNN is black (primary).
MODEL_COLORS = {
    "QNN": "#000000",
    "ANN": "#E69F00",
    "ANN-Reg": "#56B4E9",
    "SVR": "#009E73",
    "DTR": "#D55E00",
    "XGBoost": "#CC79A7",
    "RF": "#0072B2",
}

# One fixed marker per model.
MODEL_MARKERS = {
    "QNN": "o",
    "ANN": "s",
    "ANN-Reg": "^",
    "SVR": "D",
    "DTR": "v",
    "XGBoost": "P",
    "RF": "X",
}

MODEL_ORDER = ["QNN", "ANN", "ANN-Reg", "SVR", "DTR", "XGBoost", "RF"]


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
        "legend.frameon": False,
    })


def save(fig, name):
    """Save vector PDF plus a 300-dpi PNG preview; return the PDF path."""
    pdf = OUTPUT_DIR / f"{name}.pdf"
    png = OUTPUT_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(f"saved {pdf}")
    print(f"saved {png}")
    return pdf


def print_color_map():
    for m in MODEL_ORDER:
        print(f"  {m:8s} color={MODEL_COLORS[m]} marker={MODEL_MARKERS[m]}")
