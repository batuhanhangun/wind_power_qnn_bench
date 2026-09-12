"""Fig 4: actual vs predicted power at N=3200, one panel per model."""
import math

import matplotlib.pyplot as plt
import numpy as np

from figures.load_data import load_all_per_seed, load_predictions_3200
from figures.style import MODEL_COLORS, MODEL_ORDER, apply_style, save


def main():
    apply_style()
    per_seed = load_all_per_seed()
    pooled, missing = load_predictions_3200()
    if missing:
        absent_models = [m for m in MODEL_ORDER if m not in pooled]
        print("MISSING INPUTS for Fig 4 — drawing without these models:",
              absent_models)
        for f in missing:
            print("  absent:", f)

    models = [m for m in MODEL_ORDER if m in pooled]
    ncols = 4
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 1.75 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    lo = min(min(d["Actual"].min(), d["Predicted"].min()) for d in pooled.values())
    hi = max(max(d["Actual"].max(), d["Predicted"].max()) for d in pooled.values())
    pad = 0.04 * (hi - lo)
    lims = (lo - pad, hi + pad)

    for ax, m in zip(axes, models):
        d = pooled[m]
        ax.scatter(d["Actual"], d["Predicted"], s=2, alpha=0.12,
                   color=MODEL_COLORS[m], edgecolors="none", rasterized=True)
        ax.plot(lims, lims, ls="--", lw=0.7, color="#555555", zorder=4)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.text(0.05, 0.94, m, transform=ax.transAxes, ha="left", va="top",
                fontsize=8)
        sub = per_seed[(per_seed["model"] == m) & (per_seed["train_size"] == 3200)]
        mu, sd = sub["test_r2"].mean(), sub["test_r2"].std(ddof=1)
        ax.text(0.96, 0.06, f"$R^2$ = {mu:.3f} $\\pm$ {sd:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    for ax in axes[len(models):]:
        ax.set_visible(False)

    fig.supxlabel("Actual power (kW)", fontsize=8)
    fig.supylabel("Predicted power (kW)", fontsize=8)
    save(fig, "fig4_actual_vs_predicted_scatter")
    plt.close(fig)


if __name__ == "__main__":
    main()
