#!/usr/bin/env python3
"""Bolometric stellar evolution utilities."""

from pathlib import Path

import numpy as np
import scipy.interpolate


L_SUN_W = 3.846e26
R_SUN_M = 6.96e8


class BolometricEvolution:
    """Baraffe et al. 2015 bolometric luminosity and stellar radius."""

    def __init__(self, data_path=None):
        root = Path(__file__).resolve().parents[1]
        data_dir = Path(data_path) if data_path is not None else root / "data-interpolation"

        luminosity_grid = np.load(data_dir / "L_B15.npy")
        age_grid_gyr = np.load(data_dir / "tB15_Gyr.npy")
        mass_grid = np.load(data_dir / "Mstar_B15.npy")
        radius_grid = np.load(data_dir / "Rs_B15.npy")

        log_age_grid_yr = np.log10(age_grid_gyr * 1e9)
        self._log_lbol_interp = scipy.interpolate.RegularGridInterpolator(
            (mass_grid, log_age_grid_yr),
            luminosity_grid.T,
            bounds_error=False,
            fill_value=np.nan,
        )
        self._radius_interp = scipy.interpolate.RegularGridInterpolator(
            (mass_grid, log_age_grid_yr),
            radius_grid.T,
            bounds_error=False,
            fill_value=np.nan,
        )

    def lbol_lsun(self, stellar_mass, age_yr):
        """Return bolometric luminosity in solar luminosities."""
        return 10 ** self._log_lbol_interp((stellar_mass, np.log10(age_yr)))

    def lbol_w(self, stellar_mass, age_yr):
        """Return bolometric luminosity in W."""
        return L_SUN_W * self.lbol_lsun(stellar_mass, age_yr)

    def radius_rsun(self, stellar_mass, age_yr):
        """Return stellar radius in solar radii."""
        return self._radius_interp((stellar_mass, np.log10(age_yr)))
