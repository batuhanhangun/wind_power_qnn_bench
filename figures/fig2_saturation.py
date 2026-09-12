"""Fig 2: mean test R^2 vs training size, all seven models, +/- 1 sd."""
import matplotlib.pyplot as plt
import numpy as np

from figures.load_data import SIZES, load_all_per_seed
from figures.style import MODEL_COLORS, MODEL_MARKERS, MODEL_ORDER, apply_style, save


def main():
    apply_style()
    df = load_all_per_seed()

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    xpos = np.arange(len(SIZES))
    offsets = np.linspace(-0.09, 0.09, len(MODEL_ORDER))  # avoid marker pileup

    for off, model in zip(offsets, MODEL_ORDER):
        sub = df[df["model"] == model]
        means = [sub[sub["train_size"] == n]["test_r2"].mean() for n in SIZES]
        stds = [sub[sub["train_size"] == n]["test_r2"].std(ddof=1) for n in SIZES]
        primary = model == "QNN"
        if model == "ANN":
            yerr = None  # hide ANN error bars, since they are huge
        else:
            yerr = stds
        ax.errorbar(
            xpos + off, means, yerr=yerr,
            color=MODEL_COLORS[model], marker=MODEL_MARKERS[model],
            markersize=4, linewidth=1.6 if primary else 1.0,
            elinewidth=0.8, capsize=2, capthick=0.8,
            zorder=5 if primary else 3, label=model, clip_on=True,
        )

    ax.set_xticks(xpos)
    ax.set_xticklabels([str(n) for n in SIZES])
    ax.set_xlim(-0.35, len(SIZES) - 0.65)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Test $R^2$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4,
              columnspacing=0.9, handlelength=1.4, handletextpad=0.4)
    save(fig, "fig2_saturation_plot")
    plt.close(fig)


if __name__ == "__main__":
    main()
