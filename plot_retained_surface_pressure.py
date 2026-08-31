from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from script_monte_carlo_exoplanets import PLANETS

COMPOSITIONS = {
    44: (r"CO$_2$", "#377eb8"),
    16: (r"CH$_4$", "#e68613"),
    28: (r"N$_2$/O$_2$ (CP24)", "#4daf4a"),
}
DATA_DIR = Path("data-montecarlo/exoplanets_montecarlo")
OUT = Path("figures/retained_surface_pressure.png")
EARTH_MASS_KG = 5.972e24
PRESSURE_BAR_PER_MASS_FRACTION = 1.149754e6


def result_file(name, mmw):
    suffix = "_CP24" if mmw == 28 else ""
    path = DATA_DIR / f"df_{name.replace(' ', '_')}_MMW_{mmw}{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run script_monte_carlo_exoplanets.py first: missing {path}")
    return path


def loss_quantiles(name, mmw):
    data = pd.read_csv(result_file(name, mmw))
    if "atmospheric_loss" not in data:
        raise ValueError(f"Rerun {name}, MMW={mmw} with the revised Monte Carlo script")
    mass = data["pl_mass"].median()
    loss_fraction = data["atmospheric_loss"] / (data["pl_mass"] * EARTH_MASS_KG)
    if not np.isfinite(loss_fraction).all() or (loss_fraction < 0).any():
        raise ValueError(f"Invalid loss samples for {name}, MMW={mmw}")
    return mass, np.quantile(loss_fraction, [0.16, 0.50, 0.84])


def surface_pressure_bar(atmosphere_fraction, planet_mass):
    return PRESSURE_BAR_PER_MASS_FRACTION * atmosphere_fraction * planet_mass ** 0.88


def retained_pressure_bounds(initial_fraction, loss_q16_q50_q84, planet_mass):
    retained = np.maximum(initial_fraction[:, None] - loss_q16_q50_q84[[2, 1, 0]], 0)
    return surface_pressure_bar(retained, planet_mass).T


def self_check():
    assert len(PLANETS) == 27
    low, _, high = retained_pressure_bounds(
        np.array([1e-2]), np.array([1e-3, 5e-3, 2e-2]), 1,
    )[:, 0]
    assert low == 0
    assert np.isclose(high, surface_pressure_bar(9e-3, 1))


def main():
    self_check()
    initial_fraction = np.logspace(-5, -2, 250)
    colors = plt.colormaps["turbo"](np.linspace(0, 1, len(PLANETS)))
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)

    for ax, (mmw, (composition, composition_color)) in zip(axes, COMPOSITIONS.items()):
        for color, name in zip(colors, PLANETS):
            mass, loss_quantile = loss_quantiles(name, mmw)
            low, median, high = retained_pressure_bounds(initial_fraction, loss_quantile, mass)
            #ax.fill_between(initial_fraction, low, high, color=color, alpha=0.14)
            ax.plot(initial_fraction, median, color=color, linewidth=1.5, label=name)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.grid(alpha=0.2)
        ax.set_title(composition, color=composition_color, loc="left", fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96),
               ncol=6, frameon=False, fontsize=7)
    axes[-1].set_xlabel(r"Initial atmospheric mass fraction, $M_{\rm atm,0}/M_p$")
    fig.supylabel(r"Retained surface pressure, $P_{\rm retained}$ (bar)")
    fig.suptitle("Retained atmosphere after cumulative escape", y=0.995)
    fig.tight_layout(rect=(0.02, 0, 1, 0.86), h_pad=0.5)
    fig.savefig(OUT, dpi=250, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
