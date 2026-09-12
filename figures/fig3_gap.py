"""Fig 3: generalization gap — grouped train/test R^2 bars, pooled 40 runs."""
import matplotlib.pyplot as plt
import numpy as np

from figures.load_data import load_all_per_seed
from figures.style import MODEL_COLORS, apply_style, save


def main():
    apply_style()
    df = load_all_per_seed()

    stats = (df.groupby("model")
               .agg(train=("train_r2", "mean"), test=("test_r2", "mean"),
                    gap=("generalization_gap", "mean"))
               .sort_values("gap"))
    models = list(stats.index)
    print("model order by gap:", models)

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = np.arange(len(models))
    w = 0.36
    y0 = 0.65  # must sit below ANN's pooled test R^2 (~0.69) so its bars show
    for i, m in enumerate(models):
        c = MODEL_COLORS[m]
        ax.bar(x[i] - w / 2, stats.loc[m, "train"] - y0, w, bottom=y0,
               color=c, hatch="///", edgecolor="white", linewidth=0.4, alpha=0.75)
        ax.bar(x[i] + w / 2, stats.loc[m, "test"] - y0, w, bottom=y0,
               color=c, edgecolor="white", linewidth=0.4)
        top = max(stats.loc[m, "train"], stats.loc[m, "test"])
        ax.text(x[i], min(top, 1.0) + 0.006, f"{stats.loc[m, 'gap']:.3f}",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(y0, 1.02)
    ax.set_yticks(np.arange(0.65, 1.001, 0.05))
    ax.set_ylabel("$R^2$")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#999999", hatch="///", edgecolor="white", label="Train"),
               Patch(facecolor="#999999", label="Test")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, handlelength=1.2, columnspacing=1.2)
    save(fig, "fig3_generalization_gap")
    plt.close(fig)


if __name__ == "__main__":
    main()
