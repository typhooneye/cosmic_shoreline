from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLANETS = [
    "55 Cnc e", "LHS 1140 b", "HD 3167 b", "L 98-59 d", "TOI-1685 b",
    "TOI-561 b", "TOI-1468 b", "LP 890-9 c", "LTT 1445 A b", "L 98-59 c",
    "LP 890-9 b", "LHS 3844 b", "LHS 1478 b", "GJ 486 b", "LHS 1140 c",
    "GJ 3473 b", "GJ 357 b", "GJ 1132 b", "LTT 1445 A c", "TRAPPIST-1 b",
    "GJ 3929 b", "TRAPPIST-1 c", "LHS 475 b", "TRAPPIST-1 e", "GJ 341 b",
    "TRAPPIST-1 d", "GJ 367 b",
]
REFERENCES = ["Earth", "Venus"]
UNDER_DENSE = {"TOI-561 b", "55 Cnc e", "LHS 1140 b", "HD 3167 b", "L 98-59 d"}
COMPOSITIONS = {
    44: (r"CO$_2$", "#377eb8"),
    16: (r"CH$_4$", "#e68613"),
    28: (r"N$_2$/O$_2$ (CP24)", "#4daf4a"),
}
DATA_DIR = Path("data-montecarlo/exoplanets_montecarlo")
OUT = Path("figures/cumulative_atmospheric_loss_by_planet.png")
EARTH_MASS_KG = 5.972e24


def result_file(name, mmw):
    stem = name.replace(" ", "_")
    files = list(DATA_DIR.glob(f"df_{stem}_MMW_{mmw}*.csv"))
    if not files:
        raise FileNotFoundError(f"No MMW={mmw} result for {name}")
    return max(files, key=lambda path: ("CP24" in path.name, path.stat().st_mtime))


def loss_fraction_quantiles(name, mmw):
    data = pd.read_csv(result_file(name, mmw))
    fraction = data["C_loss"] / (data["pl_mass"] * EARTH_MASS_KG)
    assert np.isfinite(fraction).all() and (fraction > 0).all()
    return np.quantile(fraction, [0.16, 0.50, 0.84])


def main():
    ranked_planets = sorted(PLANETS, key=lambda name: loss_fraction_quantiles(name, 44)[1], reverse=True)
    names = ranked_planets + REFERENCES
    x = np.arange(len(names))
    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

    for ax, (mmw, (label, color)) in zip(axes, COMPOSITIONS.items()):
        q = np.array([loss_fraction_quantiles(name, mmw) for name in names])
        ax.axvspan(len(ranked_planets) - 0.5, len(names) - 0.5, color="0.94", zorder=0)
        ax.axvline(len(ranked_planets) - 0.5, color="0.55", linewidth=1)
        ax.errorbar(
            x, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
            fmt="o", ms=4.5, capsize=2, elinewidth=1, color=color, zorder=3,
        )
        ax.set_yscale("log")
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.grid(axis="y", which="both", alpha=0.22)
        ax.set_title(label, color=color, loc="left", fontweight="bold")

    axes[0].text(len(ranked_planets) + 0.5, 1.02, "Solar System references",
                 transform=axes[0].get_xaxis_transform(), ha="center", color="0.35")
    axes[-1].set_xticks(x, names, rotation=70, ha="right")
    for tick, name in zip(axes[-1].get_xticklabels(), names):
        if name in UNDER_DENSE:
            tick.set_color("green")
    axes[-1].set_xlabel(r"")
    fig.supylabel(r"Cumulative \n atmospheric loss/Mp")
    fig.suptitle("Normalized cumulative atmospheric loss")
    fig.tight_layout(rect=(0.02, 0, 1, 0.98), h_pad=0.5)
    fig.savefig(OUT, dpi=250, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
