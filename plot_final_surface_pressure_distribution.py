from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_retained_surface_pressure import (
    COMPOSITIONS,
    EARTH_MASS_KG,
    PLANETS,
    result_file,
    surface_pressure_bar,
)


OUT = Path("figures/final_surface_pressure_distribution.png")
SEED = 42


def loss_samples(name, mmw):
    data = pd.read_csv(result_file(name, mmw))
    required = {"atmospheric_loss", "pl_mass"}
    if not required <= set(data):
        raise ValueError(f"Rerun {name}, MMW={mmw} with the revised Monte Carlo script")
    mass = data["pl_mass"].to_numpy()
    loss = data["atmospheric_loss"].to_numpy() / (mass * EARTH_MASS_KG)
    if not np.isfinite(mass).all() or not np.isfinite(loss).all() or (mass <= 0).any() or (loss < 0).any():
        raise ValueError(f"Invalid samples for {name}, MMW={mmw}")
    return mass, loss


def final_pressures(name, rng):
    samples = {mmw: loss_samples(name, mmw) for mmw in COMPOSITIONS}
    sizes = {len(loss) for _, loss in samples.values()}
    if len(sizes) != 1:
        raise ValueError(f"Composition trial counts differ for {name}")
    initial_fraction = 10 ** rng.uniform(-6, -2, sizes.pop())
    return {
        mmw: surface_pressure_bar(np.maximum(initial_fraction - loss, 0), mass)
        for mmw, (mass, loss) in samples.items()
    }


def self_check():
    draws = 10 ** np.random.default_rng(0).uniform(-6, -2, 100)
    assert (draws >= 1e-6).all() and (draws <= 1e-2).all()
    pressure = surface_pressure_bar(np.maximum(np.array([1e-3, 1e-2]) - 2e-3, 0), 1)
    assert pressure[0] == 0 and pressure[1] > 0


def main():
    self_check()
    rng = np.random.default_rng(SEED)
    results = {name: final_pressures(name, rng) for name in PLANETS}
    positive = np.concatenate([
        pressure[pressure > 0]
        for planet in results.values()
        for pressure in planet.values()
    ])
    if not len(positive):
        raise ValueError("Every simulated atmosphere was lost")
    bins = np.logspace(np.floor(np.log10(positive.min())),
                       np.ceil(np.log10(positive.max())), 35)

    fig, axes = plt.subplots(6, 5, figsize=(10,10), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, name in zip(axes, PLANETS):
        for line, (mmw, (label, color)) in enumerate(COMPOSITIONS.items()):
            pressure = results[name][mmw]
            retained = pressure > 0
            ax.hist(pressure[retained], bins=bins,
                    weights=np.full(retained.sum(), 1 / len(pressure)),
                    histtype="step", linewidth=1.4, color=color, label=label)
            ax.text(0.03, 0.85 - 0.09 * line,
                    f"{label}: $P=$ {1 - retained.mean():.0%}",
                    color=color, transform=ax.transAxes, va="top", fontsize=7)
        ax.set_title(name, fontsize=10)
        ax.set_xscale("log")
        ax.grid(alpha=0.2)
        ax.text(0.03, 0.95, 'Bare Rock Probability', transform=ax.transAxes, va="top", fontsize=7)

    for ax in axes[len(PLANETS):]:
        fig.delaxes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.62, 0.99),
               ncol=3, frameon=False)
    fig.supxlabel(r"Final atmospheric surface pressure, $P_{\rm retained}$ (bar)")
    fig.supylabel("Probability per logarithmic bin")
    fig.suptitle(r"$M_{\rm atm,0}/M_p = 10^{U(-6,-2)}$",
                 x=0.02, y=0.99, ha="left")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=250, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
