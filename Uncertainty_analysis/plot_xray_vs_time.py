#!/usr/bin/env python3
"""Plot X-ray luminosity versus time for several stellar masses."""

import os
from pathlib import Path

import numpy as np

cache_dir = Path(__file__).resolve().parent / ".cache"
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib.pyplot as plt

from xray_evolution import XrayEvolution


def main():
    xray = XrayEvolution()
    ages_yr = np.logspace(6, 10, 300)
    stellar_masses = [0.1, 0.2, 0.5, 1.0]
    models = ["Jackson/Guinan", "Selsis"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in models:
        linestyle = "-" if model == "Jackson/Guinan" else "--"
        for stellar_mass in stellar_masses:
            lx_w = xray.lx_w(stellar_mass, ages_yr, model=model)
            ax.plot(
                ages_yr / 1e9,
                lx_w,
                linestyle=linestyle,
                label=f"{model}, {stellar_mass:.1f} Msun",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Age (Gyr)")
    ax.set_ylabel("X-ray luminosity (W)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "xray_vs_time.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
