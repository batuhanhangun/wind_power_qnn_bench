"""Fig 5: QNN test R^2 vs depolarizing noise level (freeze-and-evaluate)."""
import matplotlib.pyplot as plt
import numpy as np

from figures.load_data import load_noise_summary
from figures.style import MODEL_COLORS, MODEL_MARKERS, apply_style, save


def main():
    apply_style()
    df = load_noise_summary()
    x = np.arange(len(df))
    mean = df["mean_test_r2"].to_numpy()
    std = df["std_test_r2"].to_numpy()

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    # Feasible envelope p <= 0.005 (first three categorical positions).
    ax.axvspan(-0.5, 2.5, color="#B7E4C7", alpha=0.35, zorder=0, linewidth=0)
    ax.fill_between(x, mean - std, mean + std, color=MODEL_COLORS["QNN"],
                    alpha=0.15, linewidth=0)
    ax.plot(x, mean, color=MODEL_COLORS["QNN"], marker=MODEL_MARKERS["QNN"],
            markersize=4, linewidth=1.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:g}" for p in df["noise_p"]])
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_ylim(-0.6, 1.0)
    ax.set_xlabel("Depolarizing noise level $p$")
    ax.set_ylabel("Test $R^2$")
    save(fig, "fig5_noise_degradation")
    plt.close(fig)


if __name__ == "__main__":
    main()
