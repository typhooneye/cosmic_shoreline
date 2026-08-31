#!/usr/bin/env python3
"""X-ray stellar evolution utilities."""

from pathlib import Path

import numpy as np
import scipy.interpolate

from bolometric_evolution import BolometricEvolution


class XrayEvolution:
    """X-ray luminosity options used by the atmospheric-loss framework."""

    def __init__(self, data_path=None, bolometric=None):
        root = Path(__file__).resolve().parents[1]
        self.data_dir = Path(data_path) if data_path is not None else root / "data-interpolation"
        self.bolometric = bolometric or BolometricEvolution(self.data_dir)

        self.selsis_ages = np.load(self.data_dir / "y_ages_selsis07.npy")
        self.selsis_masses = np.load(self.data_dir / "x_starmasses_selsis07.npy")
        self.selsis_lx_lbol = np.load(self.data_dir / "Fx_over_Fbol_selsis07.npy")

        self.jackson_masses = np.load(self.data_dir / "j12_starmasses.npy")
        self.jackson_ages = np.load(self.data_dir / "j12_ages.npy")
        self.jackson_lx_lbol = np.load(self.data_dir / "j12_LXUV_over_Lbol.npy")

        self.guinan_masses = np.load(self.data_dir / "guinan16_mass_range.npy")
        self.guinan_ages = np.load(self.data_dir / "guinan16_ages.npy")
        self.guinan_lx_lbol = np.load(self.data_dir / "guinan16_Lx_over_Lbol.npy")

    def lx_lbol(self, stellar_mass, age_yr, model="Jackson/Guinan"):
        """Return L_X / L_bol for the selected X-ray model."""
        if model == "Selsis":
            return self._selsis_lx_lbol(stellar_mass, age_yr)
        if model in ("Jackson", "Jackson/Guinan"):
            return self._jackson_guinan_lx_lbol(stellar_mass, age_yr)
        raise ValueError("model must be 'Jackson/Guinan' or 'Selsis'")

    def lx_w(self, stellar_mass, age_yr, model="Jackson/Guinan"):
        """Return X-ray luminosity in W."""
        return self.lx_lbol(stellar_mass, age_yr, model=model) * self.bolometric.lbol_w(stellar_mass, age_yr)

    def _selsis_lx_lbol(self, stellar_mass, age_yr):
        idx = np.argmin(np.abs(self.selsis_masses - stellar_mass))
        return 10 ** scipy.interpolate.interp1d(
            np.log10(self.selsis_ages),
            np.log10(self.selsis_lx_lbol[:, idx]),
            bounds_error=False,
            fill_value="extrapolate",
        )(np.log10(age_yr))

    def _jackson_guinan_lx_lbol(self, stellar_mass, age_yr):
        if stellar_mass >= 0.5:
            idx = np.argmin(np.abs(self.jackson_masses - stellar_mass))
            ages = self.jackson_ages
            values = self.jackson_lx_lbol[idx, :]
        else:
            idx = np.argmin(np.abs(self.guinan_masses - stellar_mass))
            ages = self.guinan_ages
            values = self.guinan_lx_lbol[idx, :]

        return 10 ** scipy.interpolate.interp1d(
            np.log10(ages),
            np.log10(values),
            bounds_error=False,
            fill_value="extrapolate",
        )(np.log10(age_yr))
