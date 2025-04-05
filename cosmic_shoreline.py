#!/usr/bin/env python3
"""
cosmic_shoreline.py

This module provides the CosmicShoreline class for modeling atmospheric escape
and related planetary processes (e.g. XUV flux calculations, carbon loss, equilibrium
temperature, and more).

Usage:
    from cosmic_shoreline import CosmicShoreline
    cs = CosmicShoreline(data_path="/path/to/your/data/")
    L_XUV = cs.calculate_L_XUV(star_mass=1.0, time_XUV=1e9)
    total_loss = cs.integrate_carbon_loss(MMW=44, pl_orbsmax=1, pl_masse=1, st_mass=1.0, t1=1e6, dt=1e8)
"""

import numpy as np
import scipy
import scipy.interpolate
import scipy.integrate
import pandas as pd

# -----------------------------------------------------------------------------
# Helper class for extended linear ND interpolation with fallback to nearest
# -----------------------------------------------------------------------------
class LinearNDInterpolatorExt:
    """
    Extended Linear ND Interpolator that first uses linear interpolation and,
    if the result is NaN, falls back to a nearest-neighbor interpolation.
    """
    def __init__(self, points, values):
        self.funcinterp = scipy.interpolate.LinearNDInterpolator(points, values)
        self.funcnearest = scipy.interpolate.NearestNDInterpolator(points, values)
    
    def __call__(self, *args):
        z = self.funcinterp(*args)
        if np.any(np.isnan(z)):
            return np.where(np.isnan(z), self.funcnearest(*args), z)
        else:
            return z

# -----------------------------------------------------------------------------
# CosmicShoreline Class Definition
# -----------------------------------------------------------------------------
class CosmicShoreline:
    """
    CosmicShoreline class provides methods to calculate XUV fluxes,
    atmospheric escape rates, mass–radius relations, equilibrium temperatures,
    carbon loss rates, and more. All necessary data are loaded during initialization.
    """

    def __init__(self, data_path="/project/abbot/xuanji/2.Cosmic-Shoreline/models/data-interpolation/"):
        """
        Initialize the CosmicShoreline class.

        Parameters:
            data_path (str): Base directory where all the data (NumPy files, text files)
                             for interpolation are stored.
        """
        self.earth_mass = 5.972e24  # kg
        self.earth_radius = 6.371e6  # m

        # ---------------------------
        # Fx/Fbol interpolation data
        # ---------------------------
        # Load data for Selsis method (Selsis et al. 2007)
        self.S07_ages_y = np.load(f'{data_path}y_ages_selsis07.npy')
        self.S07_starmasses_x = np.load(f'{data_path}x_starmasses_selsis07.npy')
        self.S07_Fx_over_Fbol = np.load(f'{data_path}Fx_over_Fbol_selsis07.npy')
        
        # Load data for Jackson method (Jackson et al. 2012)
        self.j12_starmasses_x = np.load(f'{data_path}j12_starmasses.npy')
        self.j12_ages_y = np.load(f'{data_path}j12_ages.npy')
        self.j12_Lx_over_Lbol = np.load(f'{data_path}j12_LXUV_over_Lbol.npy')
        self.guinan16_Lx = np.load(f'{data_path}guinan16_Lx.npy')

        # ----------------------------------------
        # Load data for L_bol and R_star interpolation (B15) (Baraffe et al. 2015)
        # ----------------------------------------
        L_B15 = np.load(f'{data_path}L_B15.npy')  # Shape: (num_ages, num_masses)
        tB15_Gyr = np.load(f'{data_path}tB15_Gyr.npy')  # Shape: (num_ages,)
        starmass_B15 = np.load(f'{data_path}Mstar_B15.npy')  # Shape: (num_masses,)
        starradius_B15 = np.load(f'{data_path}Rs_B15.npy')  # Shape: (num_ages, num_masses)

        # Create interpolators for L_bol and R_star
        # Transpose the data arrays to match the grid dimensions
        # The grid is (mass, log(age)), so data arrays should have shape (num_masses, num_ages)
        self.L_bol_interpolator_B15 = scipy.interpolate.RegularGridInterpolator(
            (starmass_B15, np.log10(tB15_Gyr * 1e9)), L_B15.T,
            bounds_error=False, fill_value=np.nan
        )
        self.R_star_interpolator_B15 = scipy.interpolate.RegularGridInterpolator(
            (starmass_B15, np.log10(tB15_Gyr * 1e9)), starradius_B15.T,
            bounds_error=False, fill_value=np.nan
        )

        """
        CO2 atmospheric loss interpolation (Tian et al. 2009 & Tian 2009)
        """
        
        x_mass_log = np.load(f'{data_path}tian2009_log_masses.npy')
        y_xuvflux_log = np.load(f'{data_path}tian2009_log_fluxes.npy')
        z_escape_log = np.load(f'{data_path}tian2009_log_escape.npy')

        X_log, Y_log = np.meshgrid(x_mass_log, y_xuvflux_log)

        # Create interpolation function for atmospheric loss
        self.CO2_Mol_dot_logfit_interpolator = LinearNDInterpolatorExt((np.log10(X_log.flatten()), 
                                                            np.log10(Y_log.flatten())), 
                                                            np.log10(z_escape_log.flatten()))

       
        x_mass = np.load(f'{data_path}tian2009_masses.npy')
        y_xuvflux = np.load(f'{data_path}tian2009_fluxes.npy')
        z_escape = np.load(f'{data_path}tian2009_escape.npy')

        X, Y = np.meshgrid(x_mass, y_xuvflux)

        # Create interpolation function for atmospheric loss
        self.CO2_Mol_dot_linearfit_interpolator = LinearNDInterpolatorExt((X.flatten(), 
                                                            np.log10(Y.flatten())), 
                                                            np.log10(z_escape.flatten()))
        
        x_GP_GP = np.load(f'{data_path}tian2009_GP_GP.npy')
        y_xuvflux_GP = np.load(f'{data_path}tian2009_GP_fluxes.npy')
        z_escape_GP = np.load(f'{data_path}tian2009_GP_escape.npy')

        X_GP, Y_GP = np.meshgrid(x_GP_GP, y_xuvflux_GP)

        # Create interpolation function for atmospheric loss
        self.CO2_Mol_dot_GP_interpolator = LinearNDInterpolatorExt((X_GP.flatten(),
                                                            np.log10(Y_GP.flatten())), 
                                                            z_escape_GP.flatten())
        # ---------------------------
        # N2/O2 and H2O escape rate interpolators
        # ---------------------------
        # N2O2 loss interpolation using Nakauchi et al. 2022
        x_F_XUV_to_earth_N2O2 = np.array([1,1.999456012001378,3.011087696068116,3.997824343928456,
                                            5.036486330526271,10.070232872933508,20.134987660040828,
                                            30.00569202577696,50.18892605590459,100.35054993837102,
                                            200.64651038192062,299.00874528544716,500.1360340000868,
                                            1000])
        F_xuv_earth = 0.00464
        x_XUV_N2O2 = F_xuv_earth*x_F_XUV_to_earth_N2O2

        y_time_scale_N2O2 = np.array([
            208.18930618300956,208.18930618300956,220.19080978248428,236.17022977547424,271.6921428536773,
            330.5748251334223,202.43613289325174,52.736285090565076,7.416219167256278,2.28573758287157,
            2.0433597178569416,2.1310832013045498,2.0433597178569416,1.986892828343377])
        y_mass_rate_kg_sm2_N2O2 = 5e18/(4*np.pi*(6.637e6)**2)/(y_time_scale_N2O2*1e9*3.15e7)

        self.log_N2O2_loss_kg_sm2_interpolator = scipy.interpolate.interp1d(
            np.log10(x_F_XUV_to_earth_N2O2), np.log10(y_mass_rate_kg_sm2_N2O2), bounds_error=False, fill_value='extrapolate'
        )

        # Create interpolators for H2O loss (Jhonstone et al. 2020)
        x_H2O_Fxuv_erg_s_cm2 = np.array([18.4+98.7, 22.9+124.5, 30.0+166.0, 42.3+241.2, 68.5+409.8,
                                       77.5+470.2, 89.0+548.2, 104.0+651.8, 124.2+794.7, 152.8+1001.7,
                                       195.7+1321.9, 266.3+1866.0, 399.4+2933.9, 719.0+5632.0])
        x_H2O_Fx_erg_s_cm2 = np.array([18.4, 22.9, 30.0, 42.3, 68.5, 77.5, 89.0, 104.0, 124.2, 152.8, 195.7, 266.3, 399.4, 719.0])

        y_H2O_1em10_g_s_cm2 = np.array([
            0.0027213326776891395,0.0036842826076093887,0.00529046732581549,0.008333898193473611,
            0.01452459920195443,0.0166187716404126,0.019175597595385636,0.022501968929153852,
            0.026853604116249177,0.03231848884251108,0.03793157757750501,0.04924701589802671,
            0.07073565671613313,0.11336574365517495
        ])
        self.H2O_loss_g_s_cm2_itpltor = scipy.interpolate.interp1d(np.log10(x_H2O_Fxuv_erg_s_cm2), -10+np.log10(y_H2O_1em10_g_s_cm2),
                                                                    bounds_error=False, fill_value='extrapolate')
        
        self.H2O_loss_g_s_cm2_itpltor_Fx = scipy.interpolate.interp1d(np.log10(x_H2O_Fx_erg_s_cm2), -10+np.log10(y_H2O_1em10_g_s_cm2),
                                                                    bounds_error=False, fill_value='extrapolate')

        """
        Non-thermal escape rate for CO2 (mole/s) (Chin et al. 2024)
        """

        R_arr = np.array([0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.25])*0.95
        CO2_flux = np.array([0.081,0.268,0.148,0.066,0.102,0.056,0.049,0.037])

        self.CO2_nonthermal_1e26mole_per_sec = scipy.interpolate.interp1d(R_arr, CO2_flux, 
                                                                          bounds_error=False, fill_value=np.nan)


        
        """
        Load data for M_R interpolation
        Based on Zeng et al. 2019
        # http://www.astrozeng.com/
        """
        x_data_Earth = np.load(f'{data_path}x_data_Earth.npy')
        y_data_Earth = np.load(f'{data_path}y_data_Earth.npy')

        x_data_iron = np.load(f'{data_path}x_data_iron.npy')
        y_data_iron = np.load(f'{data_path}y_data_iron.npy')

        x_data_rock = np.load(f'{data_path}x_data_rock.npy')
        y_data_rock = np.load(f'{data_path}y_data_rock.npy')

        self.M_R_interpolator_Earth_xM = scipy.interpolate.interp1d(x_data_Earth, y_data_Earth, fill_value='extrapolate')
        self.M_R_interpolator_iron_xM = scipy.interpolate.interp1d(x_data_iron, y_data_iron, fill_value='extrapolate')
        self.M_R_interpolator_rock_xM = scipy.interpolate.interp1d(x_data_rock, y_data_rock, fill_value='extrapolate')

        self.M_R_interpolator_Earth_xR = scipy.interpolate.interp1d(y_data_Earth, x_data_Earth, fill_value='extrapolate')
        self.M_R_interpolator_iron_xR = scipy.interpolate.interp1d(y_data_iron, x_data_iron, fill_value='extrapolate')
        self.M_R_interpolator_rock_xR = scipy.interpolate.interp1d(y_data_rock, x_data_rock, fill_value='extrapolate')

        # ---------------------------
        # CR24 N2O2 atmosphere interpolation (Chatterjee and Pierrehumbert 2024)
        # ---------------------------
        earth_radius = self.earth_radius
        F_xuv_to_Earth_range = np.logspace(0,5,100)

        Fxuv_Mars = 50
        M_dot_Mars = 3e8/1e3
        M_dot_Mars_per_area = M_dot_Mars/(4*np.pi*(self.M_R_interpolator_Earth_xM(0.1)*earth_radius)**2)
        M_dot_Mars_per_area_arr = M_dot_Mars_per_area * (F_xuv_to_Earth_range/Fxuv_Mars)    
        unc_dex_Mars_low = 1
        log_Fxuv_Mars_range_low = np.log10(Fxuv_Mars) + np.array([-unc_dex_Mars_low, unc_dex_Mars_low])

        unc_dex_Mars_high_1 = -1
        unc_dex_Mars_high_2 = 2
        log_Fxuv_Mars_high_range = np.log10(Fxuv_Mars) + np.array([unc_dex_Mars_high_1, unc_dex_Mars_high_2])


        Fxuv_Earth = 400
        M_dot_Earth = 1e9/1e3
        M_dot_Earth_per_area = M_dot_Earth/(4*np.pi*(self.M_R_interpolator_Earth_xM(1)*earth_radius)**2)
        M_dot_Earth_per_area_arr = M_dot_Earth_per_area * (F_xuv_to_Earth_range/Fxuv_Earth)
        unc_dex_Earth_low = 0.5
        log_Fxuv_Earth_range_low = np.log10(Fxuv_Earth) + np.array([-unc_dex_Earth_low, unc_dex_Earth_low])

        unc_dex_Earth_high_1 = -0.5
        unc_dex_Earth_high_2 = 1.5
        log_Fxuv_Earth_high_range = np.log10(Fxuv_Earth) + np.array([unc_dex_Earth_high_1, unc_dex_Earth_high_2])



        Fxuv_1d76Me = 1000
        M_dot_1d76Me = 2.2e8/1e3
        M_dot_1d76Me_per_area = M_dot_1d76Me/(4*np.pi*(self.M_R_interpolator_Earth_xM(1.76)*earth_radius)**2)
        M_dot_1d76Me_per_area_arr = M_dot_1d76Me_per_area * (F_xuv_to_Earth_range/Fxuv_1d76Me)
        unc_dex_1d76Me_low = 0.3
        log_Fxuv_1d76Me_range_low = np.log10(Fxuv_1d76Me) + np.array([-unc_dex_1d76Me_low, unc_dex_1d76Me_low])

        unc_dex_1d76Me_high_1 = -0.3
        unc_dex_1d76Me_high_2 = 1.3
        log_Fxuv_1d76Me_high_range = np.log10(Fxuv_1d76Me) + np.array([unc_dex_1d76Me_high_1, unc_dex_1d76Me_high_2])

        Fxuv_5d9M = 2000
        M_dot_5d9M = 3.4e8/1e3
        M_dot_5d9M_per_area = M_dot_5d9M/(4*np.pi*(self.M_R_interpolator_Earth_xM(5.9)*earth_radius)**2)
        M_dot_5d9M_per_area_arr = M_dot_5d9M_per_area * (F_xuv_to_Earth_range/Fxuv_5d9M)
        unc_dex_5d9M_low = 0.01
        log_Fxuv_5d9M_range_low = np.log10(Fxuv_5d9M) + np.array([-unc_dex_5d9M_low, unc_dex_5d9M_low])

        unc_dex_5d9M_high_1 = -0.01
        unc_dex_5d9M_high_2 = 1.01
        log_Fxuv_5d9M_high_range = np.log10(Fxuv_5d9M) + np.array([unc_dex_5d9M_high_1, unc_dex_5d9M_high_2])

        F_xuv_data = np.zeros(400)
        M_dot_data = np.zeros(400)
        M_dot_per_area_data = np.zeros(400)
        pl_masse_data = np.zeros(400)

        for i in range(4):
            F_xuv_data[i*100:(i+1)*100] = F_xuv_to_Earth_range
            M_dot_per_area_data[i*100:(i+1)*100] = [M_dot_Mars_per_area_arr, M_dot_Earth_per_area_arr, 
                                                    M_dot_1d76Me_per_area_arr, M_dot_5d9M_per_area_arr][i]
            pl_masse_data[i*100:(i+1)*100] = [0.1,1,1.76,5.9][i]


        pl_masse_arr = np.array([0.1,1,1.76,5.9])
        log_fxuv_low_limit_low = np.array([log_Fxuv_Mars_range_low[0], log_Fxuv_Earth_range_low[0], log_Fxuv_1d76Me_range_low[0], log_Fxuv_5d9M_range_low[0]])
        log_fxuv_high_limit_low = np.array([log_Fxuv_Mars_range_low[1], log_Fxuv_Earth_range_low[1], log_Fxuv_1d76Me_range_low[1], log_Fxuv_5d9M_range_low[1]])

        log_fxuv_low_limit_high = np.array([log_Fxuv_Mars_high_range[0], log_Fxuv_Earth_high_range[0], log_Fxuv_1d76Me_high_range[0], log_Fxuv_5d9M_high_range[0]])
        log_fxuv_high_limit_high = np.array([log_Fxuv_Mars_high_range[1], log_Fxuv_Earth_high_range[1], log_Fxuv_1d76Me_high_range[1], log_Fxuv_5d9M_high_range[1]])

        self.CR24_N2O2_EL_interp = LinearNDInterpolatorExt((np.log10(F_xuv_data), pl_masse_data), np.log10(M_dot_per_area_data))

        self.CR24_N2O2_low_cut_interp_low = scipy.interpolate.interp1d(pl_masse_arr, log_fxuv_low_limit_low, fill_value=(log_fxuv_low_limit_low[0], log_fxuv_low_limit_low[-1]), bounds_error=False)
        self.CR24_N2O2_high_cut_interp_low = scipy.interpolate.interp1d(pl_masse_arr, log_fxuv_high_limit_low, fill_value=(log_fxuv_high_limit_low[0], log_fxuv_high_limit_low[-1]), bounds_error=False)

        self.CR24_N2O2_low_cut_interp_high = scipy.interpolate.interp1d(pl_masse_arr, log_fxuv_low_limit_high, fill_value=(log_fxuv_low_limit_high[0], log_fxuv_low_limit_high[-1]), bounds_error=False)
        self.CR24_N2O2_high_cut_interp_high = scipy.interpolate.interp1d(pl_masse_arr, log_fxuv_high_limit_high, fill_value=(log_fxuv_high_limit_high[0], log_fxuv_high_limit_high[-1]), bounds_error=False)

        # ---------------------------
        # Heat Capacity Ratio (gamma) interpolation
        # ---------------------------
        # Read the file into a pandas DataFrame
        file_path = f'{data_path}gamma_CO2_CEA.txt'
        data = pd.read_csv(file_path, sep='\s+', header=None, names=['t', 'p', 'gam'],skiprows=1)

        self.interp_gamma_T_logP = LinearNDInterpolatorExt(list(zip(data['t'], np.log10(data['p']*1e5))), data['gam'])

        # ---------------------------
        # Magma Depth Initialization
        # ---------------------------
        # Load total melt mass data and create a list of interpolators over temperature
        Tp_list = np.arange(3e3, 2.1e3, -2e2)
        Tp_list = np.append(Tp_list, 2000)
        pl_masses = np.arange(0.5, 10.01, 0.1)

        X, Y = np.meshgrid(Tp_list, pl_masses)
        z = np.load(f'{data_path}total_melt_mass_shallower_than_2900km.npy')
        
        magma_itpltor = []
        for n_y,pl_masse in enumerate(pl_masses):
            magma_itpltor.append(scipy.interpolate.interp1d(Tp_list, z[n_y,:], bounds_error=False, fill_value='extrapolate'))
        # Create interpolator for magma depth
        self.M_magma_interpolators_0d5_to_10_Me_d0d1 = magma_itpltor

    # -------------------------------------------------------------------------
    # Below follow the methods for mass–radius fits, luminosity, escape rates,
    # equilibrium temperature, carbon loss integration, and atmospheric profiles.
    # Each method includes a docstring and inline comments.
    # -------------------------------------------------------------------------
    
    def M_R_fit(self, x, type='Earth', x_M_or_R='M'):
        """
        Fit the mass–radius relation for a planet.
        (Zeng et al. 2019)
        Parameters:
            x: Input value (mass or radius) for interpolation.
            type: Composition type ('Earth', 'iron', or 'rock').
            x_M_or_R: Specify whether x is mass ('M') or radius ('R').

        Returns:
            Interpolated mass or radius.
        """
        if type == 'Earth':
            if x_M_or_R == 'M':
                return self.M_R_interpolator_Earth_xM(x)
            else:
                return self.M_R_interpolator_Earth_xR(x)
        elif type == 'iron':
            if x_M_or_R == 'M':
                return self.M_R_interpolator_iron_xM(x)
            else:
                return self.M_R_interpolator_iron_xR(x)
        elif type == 'rock':
            if x_M_or_R == 'M':
                return self.M_R_interpolator_rock_xM(x)
            else:
                return self.M_R_interpolator_rock_xR(x)
    
    def calculate_L_X_to_bol(self, Ms, time_XUV, method='Jackson'):
        """
        Calculate the XUV-to-bolometric luminosity ratio based on stellar mass and age.
        Parameters:
            Ms: Stellar mass (scalar).
            time_XUV: Stellar age in years (scalar or array).
            method: 'Selsis' or 'Jackson' interpolation method. (default: 'Jackson')
            (Selsis et al. 2007 or Jackson et al. 2012)

        Returns:
            L_XUV / L_bol ratio.
        """
        if method not in ['Selsis', 'Jackson']:
            raise ValueError("Method must be either 'Selsis' or 'Jackson'")
        if method == 'Selsis':
            idx = np.argmin(np.abs(self.S07_starmasses_x - Ms))
            L_X_to_bol = 10 ** scipy.interpolate.interp1d(
                np.log10(self.S07_ages_y),
                np.log10(self.S07_Fx_over_Fbol[:, idx]),
                bounds_error=False, fill_value='extrapolate'
            )(np.log10(time_XUV))
        else:  # Jackson method
            if Ms >= 0.5:
                idx = np.argmin(np.abs(self.j12_starmasses_x - Ms))
                L_X_to_bol = 10 ** scipy.interpolate.interp1d(
                    np.log10(self.j12_ages_y),
                    np.log10(self.j12_Lx_over_Lbol[idx, :]),
                    bounds_error=False, fill_value='extrapolate'
                )(np.log10(time_XUV))
            else:
                L_X = 10 ** scipy.interpolate.interp1d(
                    np.log10(self.j12_ages_y),
                    np.log10(self.guinan16_Lx),
                    bounds_error=False, fill_value='extrapolate'
                )(np.log10(time_XUV))
                L_bol = 3.827e26 * 10 ** self.L_bol_interpolator_B15((Ms, np.log10(time_XUV)))
                L_X_to_bol = L_X / L_bol
        return L_X_to_bol
    
    def calculate_L_XUV(self, Ms, time_XUV, method='Jackson', output='single',
                        beta=0, gamma=0, beta1=116, beta2=3040, gamma1=-0.35, gamma2=-0.76):
        """
        Extrapolate the XUV luminosity based on X-ray using either the King and Wheatley (2020)
        method or an alternative fitting method.
        (King & Wheatley 2020)

        Parameters:
            Ms: Stellar mass (scalar).
            time_XUV: Stellar age in years (scalar or array).
            method: Interpolation method ('Selsis' or 'Jackson').
            beta, gamma: Parameters for an alternative fitting method.
            beta1, beta2, gamma1, gamma2: Parameters for the King & Wheatley (2020) method.
            output: 'single' for total L_XUV, 'multi-band' to return individual components.

        Returns:
            XUV luminosity (W/m²) or a tuple with multiple bands and stellar radius.
        """
        L_X_to_bol = self.calculate_L_X_to_bol(Ms, time_XUV, method)
        L_bol = 3.846e26 * 10 ** self.L_bol_interpolator_B15((Ms, np.log10(time_XUV)))
        Rs = self.R_star_interpolator_B15((Ms, np.log10(time_XUV)))
        As = 4 * np.pi * (Rs * 6.96e8) ** 2 * 1e4  # Stellar surface area in cm²

        if beta1 != 0 and beta2 != 0:
            L_EUV1_to_bol = beta1 * (L_bol * 1e7/As) ** (gamma1) * (L_X_to_bol) ** (gamma1+1)
            L_EUV2_to_bol = beta2 * (L_bol * 1e7/As) ** (gamma2) * (L_X_to_bol) ** (gamma2+1)
            L_XUV_to_bol = L_X_to_bol + L_EUV1_to_bol + L_EUV2_to_bol
            L_XUV = L_XUV_to_bol * L_bol
        else:
            L_EUV_to_bol = beta * (L_bol * 1e7/As) ** gamma * (L_X_to_bol) ** (gamma+1)
            L_XUV_to_bol = L_X_to_bol + L_EUV_to_bol
            L_XUV = L_XUV_to_bol * L_bol

        if output == 'multi-band':
            if beta1 != 0 and beta2 != 0:
                return L_XUV, L_X_to_bol * L_bol, L_EUV1_to_bol * L_bol, L_EUV2_to_bol * L_bol, Rs
            else:
                return L_XUV, L_X_to_bol * L_bol, L_EUV_to_bol * L_bol, Rs
        else:
            return L_XUV
        
    def M_dot_stellar_wind(self, Lx, R_star, idx=0.77):
        '''
        Calculate the mass loss rate per unit area of a star.
        (Wood et al. 2021)

        Parameters:
        - Lx: The X-ray luminosity in erg/s
        - R_star: The stellar radius in Solar radii
        - idx: The power-law index for the mass loss rate scaling. 0.77 from B. E. Wood et al. (2021)
        Returns:
        - M_dot_per_area: The mass loss rate per unit area in M_sun/s/R_sun^2
        '''
        M_dot_surf_0 = 0.5/0.47**2
        R_sun_cm = 6.957e10
        Fx_surf_0 = 10**27.03/(4*np.pi*(0.47*R_sun_cm)**2)
        Fx_surf = Lx*1e7/(4*np.pi*(R_star*R_sun_cm)**2)
        M_dot_per_area = Fx_surf**idx/(Fx_surf_0**0.77)*M_dot_surf_0
        M_dot = M_dot_per_area*(R_star)**2
        return M_dot
    
    def M_C_dot_non_thermal(self, Lx, R_star, R_planet, a, idx=0.77):
        '''
        Calculate the carbon mass loss rate per unit area of a star.
        (Chin et al. 2024)

        Parameters:
        - Lx: The X-ray luminosity in erg/s
        - R_star: The stellar radius in Solar radii
        - R_planet: The planet radius in Earth radii
        - a: The semi-major axis of the planet in AU
        - idx: The power-law index for the mass loss rate scaling
        Returns:
        - M_dot_per_area: The mass loss rate per unit area in M_sun/s/R_sun^2
        '''    
        M_dot_arr = self.M_dot_stellar_wind(Lx, R_star, idx)
        # M_dot_sun_Msun_yr = 2e-14
        # M_dot_0 = 8.5e-13/M_dot_sun_Msun_yr

        Lx_to_bol_youngsun = self.calculate_L_X_to_bol(1, 6e8)
        L_bol_youngsun = 3.846e26*10**self.L_bol_interpolator_B15((1, np.log10(6e8)))
        # Lx_youngsun = Lx_to_bol_youngsun*L_bol_youngsun
        # R_star_youngsun = self.R_star_interpolator_B15((1, np.log10(6e8)))

        # M_dot_0 = self.M_dot_stellar_wind(Lx_youngsun, R_star_youngsun, idx=0.77)
        M_dot_sun_Msun_yr = 2e-14
        M_dot_0 =  9.29e-13/M_dot_sun_Msun_yr

        C_mole_dot_0 = self.CO2_nonthermal_1e26mole_per_sec(R_planet)
        C_mole_dot = C_mole_dot_0/a**2*M_dot_arr/M_dot_0
        C_dot = C_mole_dot*1e26/6e23*0.012
        return C_dot

    def _calculate_F_xuv_to_earth(self, L_XUV, pl_orbsmax):
        """
        Calculate the XUV flux relative to Earth's.

        This method computes the ratio of the XUV flux received by the planet
        to the XUV flux received by Earth. It uses the stellar XUV luminosity
        and the planet's orbital distance to determine this ratio.

        Parameters:
        L_XUV (float): The XUV luminosity of the star in W
        pl_orbsmax (float): The semi-major axis of the planet's orbit in AU

        Returns:
        float: The XUV flux relative to Earth's XUV flux
        """
        F_xuv_earth = 0.00464 # Ribas et al. 2005
        return L_XUV/(4*np.pi*(pl_orbsmax*1.496e11)**2)/F_xuv_earth

    def y_Mol_cm_s_CO2_x_XUV(self, pl_masse, F_xuv_to_earth, CO2_fit='linear',Energy_limit=False,pl_radiuse=""):
        """
        Calculate the CO2 escape rate in molecules/cm²/s for a given planet.

        Parameters:
        - pl_masse: Planet mass in Earth masses
        - F_xuv_to_earth: XUV flux relative to Earth's
        - CO2_fit: CO2 escape model to use ('log', 'linear', or 'GP')
        - Energy_limit: If True, return the energy-limited escape rate
        - pl_radiuse: Planet radius in Earth radii
        Returns:
        - CO2 escape rate in molecules/cm²/s
        """

        F_xuv_earth = 0.00464 # W/m2 Ribas et al. 2005
        K_tide = 1
        if pl_radiuse=="":
            pl_radiuse = self.M_R_fit(pl_masse)
        Mol_dot_Energy =  6e23/0.012/1e4/4 * (0.1 * F_xuv_to_earth * F_xuv_earth * (pl_radiuse * 6.37e6)) / (6.67e-11 * pl_masse * 5.97e24 * K_tide)

        # print(np.log10(F_xuv_to_earth))
        if CO2_fit == 'log':
            Mol_dot = self.CO2_Mol_dot_logfit_interpolator(np.log10(pl_masse), np.log10(F_xuv_to_earth))
            Mol_dot = 10**Mol_dot
            # if np.isscalar(Mol_dot):
                # if Mol_dot < 1e7:
                    # Mol_dot = 0
            # else:
                # Mol_dot[Mol_dot < 1e7] = 0
        elif CO2_fit == 'linear':
            Mol_dot = self.CO2_Mol_dot_linearfit_interpolator(pl_masse, np.log10(F_xuv_to_earth))
            Mol_dot = 10**Mol_dot
        elif CO2_fit == 'GP':
            pl_GP = (pl_masse/pl_radiuse)**(-1)
            Mol_dot = self.CO2_Mol_dot_GP_interpolator(pl_GP, np.log10(F_xuv_to_earth))
            Mol_dot = Mol_dot
        if Energy_limit == True:
            Mol_dot = Mol_dot_Energy
        else:
            Mol_dot = np.minimum(Mol_dot, Mol_dot_Energy)
        return Mol_dot
    
    def M_C_dot_CO2(self, F_xuv_to_earth, pl_masse, CO2_fit='linear',Energy_limit=False, pl_radiuse=""):
        """
        Calculate the CO2 escape rate in kg/s for a given planet.
        (Tian et al. 2009 & Tian 2009)

        Parameters:
        - F_xuv_to_earth: XUV flux relative to Earth's
        - pl_masse: Planet mass in Earth masses
        - CO2_fit: CO2 escape model to use ('log', 'linear', or 'GP')
        - Energy_limit: If True, return the energy-limited escape rate
        - pl_radiuse: Planet radius in Earth radii
        Returns:

        - CO2 escape rate in kg/s
        """

        if pl_radiuse=="":
            pl_radiuse = self.M_R_fit(pl_masse)

        if np.isscalar(F_xuv_to_earth):
            F_xuv_to_earth = max(1e-10, F_xuv_to_earth)
        else:
            F_xuv_to_earth = np.maximum(F_xuv_to_earth, 1e-10)

        Mol_dot = self.y_Mol_cm_s_CO2_x_XUV(pl_masse, F_xuv_to_earth, CO2_fit=CO2_fit, Energy_limit=Energy_limit,pl_radiuse=pl_radiuse)

        M_dot = Mol_dot / 6e23 * 0.012 * (1e4 * 4 * np.pi * (pl_radiuse * 6.37e6)**2) 
        return M_dot
    
    def M_C_dot_N2O2(self, F_xuv_to_earth, pl_masse, pl_radiuse="", model = 'N22',frac=0.5):
        """
        Calculate the N2O2 escape rate in kg/s for a given planet.
        (Nakayama et al 2022 (default) & Chatterjee and Pierrehumbert 2024)
        
        Parameters:
        - F_xuv_to_earth: XUV flux relative to Earth's
        - pl_masse: Planet mass in Earth masses
        - pl_radiuse: Planet radius in Earth radii
        - model: N2O2 escape model to use ('N22' or 'CR24')
        - frac: If 'CR24' model is used, the fraction of the range between the low and high cut values to use
        """
        if model == 'N22':
            if pl_masse !=1:
                raise ValueError("N2O2 loss only applies to Earth-mass planets")
            if pl_radiuse=="":
                pl_radiuse = 1
            if np.isscalar(F_xuv_to_earth):
                F_xuv_to_earth = max(1e-10, F_xuv_to_earth)
            else:
                F_xuv_to_earth = np.maximum(F_xuv_to_earth, 1e-10)
            
            loss_rate_kg_sm2 = 10**(self.log_N2O2_loss_kg_sm2_interpolator(np.log10(F_xuv_to_earth)))
            M_dot = loss_rate_kg_sm2*(4*np.pi*(6.637e6)**2)
        elif model == 'CR24':
            if np.isscalar(F_xuv_to_earth):
                F_xuv_to_earth = max(1e-10, F_xuv_to_earth)
                Fxuv = np.array([F_xuv_to_earth])
                pl_masse = np.array([pl_masse])
            elif np.isscalar(pl_masse):
                F_xuv_to_earth = np.maximum(F_xuv_to_earth, 1e-10)
                Fxuv = F_xuv_to_earth
                pl_masse = pl_masse*np.ones_like(F_xuv_to_earth)
            else:
                F_xuv_to_earth = np.maximum(F_xuv_to_earth, 1e-10)
                Fxuv = F_xuv_to_earth

            if pl_radiuse=="":
                pl_radiuse = self.M_R_fit(pl_masse)
                
            low_cut_low_log = self.CR24_N2O2_low_cut_interp_low(5.9)*np.ones_like(pl_masse)
            low_cut_low_log[pl_masse < 5.9] = self.CR24_N2O2_low_cut_interp_low(pl_masse[pl_masse < 5.9])

            high_cut_low_log = self.CR24_N2O2_high_cut_interp_low(5.9)*np.ones_like(pl_masse)
            high_cut_low_log[pl_masse < 5.9] = self.CR24_N2O2_high_cut_interp_low(pl_masse[pl_masse < 5.9])

            loss_rate_EL = 10**self.CR24_N2O2_EL_interp(np.log10(Fxuv), pl_masse)

            low_cut_high_log = self.CR24_N2O2_low_cut_interp_high(5.9)*np.ones_like(pl_masse)
            low_cut_high_log[pl_masse < 5.9] = self.CR24_N2O2_low_cut_interp_high(pl_masse[pl_masse < 5.9])

            high_cut_high_log = self.CR24_N2O2_high_cut_interp_high(5.9)*np.ones_like(pl_masse)
            high_cut_high_log[pl_masse < 5.9] = self.CR24_N2O2_high_cut_interp_high(pl_masse[pl_masse < 5.9])


            low_cut= 10**(low_cut_low_log + frac * (low_cut_high_log - low_cut_low_log))
            high_cut = 10**(high_cut_low_log + frac * (high_cut_high_log - high_cut_low_log))

            loss_rate_low_logmax = self.CR24_N2O2_EL_interp(np.log10(high_cut[Fxuv > high_cut]), pl_masse[Fxuv > high_cut])
            loss_rate_high_logmax = self.CR24_N2O2_EL_interp(np.log10(high_cut[Fxuv > high_cut]), pl_masse[Fxuv > high_cut])

            loss_rate = loss_rate_EL
            loss_rate[Fxuv < low_cut] = 1e-17
            loss_rate[Fxuv > high_cut] = 10**(loss_rate_low_logmax + frac * (loss_rate_high_logmax - loss_rate_low_logmax))
            
            M_dot = loss_rate*(4*np.pi*(pl_radiuse*6.637e6)**2)
            if len(M_dot) == 1:
                M_dot = M_dot[0]
        return M_dot
    
    def M_C_dot_H2O(self, F_xuv_to_earth, pl_masse, pl_radiuse="",Fx_Wm2=""):
        """
        Calculate the H2O escape rate in kg/s for a given planet.
        (Johnstone et al. 2020)

        Parameters:
        - F_xuv_to_earth: XUV flux relative to Earth's
        - pl_masse: Planet mass in Earth masses
        - pl_radiuse: Planet radius in Earth radii
        - Fx_Wm2: X-ray flux in W/m² (optional, if input, F_xuv_to_earth is ignored)
        Returns:
        - H2O escape rate in kg/s
        """

        if pl_masse !=1:
            raise ValueError("N2O2 loss only applies to Earth-mass planets")
        if pl_radiuse=="":
            pl_radiuse = 1
        if np.isscalar(F_xuv_to_earth):
            F_xuv_to_earth = max(1e-10, F_xuv_to_earth)
        else:
            F_xuv_to_earth = np.maximum(F_xuv_to_earth, 1e-10)
        if Fx_Wm2 == "":
            Wm2_to_ergscm2 = 1e3
            F_xuv_earth = 0.00464 # Ribas et al. 2005
            Fxuv = F_xuv_to_earth*F_xuv_earth*Wm2_to_ergscm2
            mass_rate_g_s_cm2_H2O = 10**self.H2O_loss_g_s_cm2_itpltor(np.log10(Fxuv))
            M_dot = (4*np.pi*(6.37e6+1e8)**2*1e4) /1e3 * mass_rate_g_s_cm2_H2O
            return M_dot
        else:
            Wm2_to_ergscm2 = 1e3
            Fx_erg_s_cm2 = Fx_Wm2*Wm2_to_ergscm2
            mass_rate_g_s_cm2_H2O = 10**self.H2O_loss_g_s_cm2_itpltor_Fx(np.log10(Fx_erg_s_cm2))
            M_dot = (4*np.pi*(6.37e6+1e8)**2*1e4) /1e3 * mass_rate_g_s_cm2_H2O
            return M_dot


    def M_C_dot_CH4(self, F_xuv_to_earth, pl_masse, pl_radiuse="",
                    epsilon=0.1, K_tide=1, R_xuv_ratio=1):
        """
        Calculate the CH4 escape rate in kg/s for a given planet using energy-limited estimate.)

        Parameters:
        - F_xuv_to_earth: XUV flux relative to Earth's
        - pl_masse: Planet mass in Earth masses
        - pl_radiuse: Planet radius in Earth radii
        - epsilon: The efficiency factor for CH4 escape
        - K_tide:
        - R_xuv_ratio: The ratio of the XUV absorption radius to the planet radius
        Returns:
        - CH4 escape rate in kg/s
        """


        if pl_radiuse=="":
            pl_radiuse = self.M_R_fit(pl_masse)
        if np.isscalar(F_xuv_to_earth):
            F_xuv_to_earth = max(1e-10, F_xuv_to_earth)
        else:
            F_xuv_to_earth = np.maximum(F_xuv_to_earth, 1e-10)

        # e_lyman = 6.626e-34 * 2.47e15
        # flux_lyman_0 = 6.19e-3  # W/m2 # Ribas et al. 2005
        # flux_lyman = flux_lyman_0 * F_xuv_to_earth
        
        # n_CH4_total = flux_lyman * np.pi * (pl_radiuse * 6.37e6)**2 / e_lyman
        # M_C_dot_lyman = n_CH4_total / 6e23 * 0.012  # kg
        
        F_xuv_earth = 0.00464 # W/m2 Ribas et al. 2005
        R_xuv = R_xuv_ratio * pl_radiuse

        M_C_dot_energy = 12/16 * (epsilon * np.pi * F_xuv_to_earth * F_xuv_earth * (R_xuv *  6.37e6)**3) / (6.67e-11 * pl_masse * 5.97e24 * K_tide)
        # if lyman_or_energy == 1:
        #     return M_C_dot_lyman
        # elif lyman_or_energy == 2:
        #     return M_C_dot_energy
        # elif lyman_or_energy == 0:
        #     return np.minimum(M_C_dot_lyman, M_C_dot_energy)
        return M_C_dot_energy
    def M_C_dot_Energy(self, F_xuv_to_earth, pl_masse, pl_radiuse=""
                       , epsilon=0.1, K_tide=1, R_xuv_ratio=1):
        if pl_radiuse=="":
            pl_radiuse = self.M_R_fit(pl_masse)
        if np.isscalar(F_xuv_to_earth):
            F_xuv_to_earth = max(1e-10, F_xuv_to_earth)
        else:
            F_xuv_to_earth = np.maximum(F_xuv_to_earth, 1e-10)

        R_xuv = R_xuv_ratio * pl_radiuse

        F_xuv_earth = 0.00464 # W/m2 Ribas et al. 2005
        return epsilon * np.pi * F_xuv_to_earth * F_xuv_earth * (R_xuv * 6.37e6)**3 / (6.67e-11 * pl_masse * 5.97e24 * K_tide)
    

    def M_C_dot_loss(self, MMW, pl_orbsmax, pl_masse, L_XUV, 
                     pl_radiuse="",CO2_fit='linear',Energy_limit=False, N2O2_model='N22',
                     epsilon=0.1, K_tide=1, R_xuv_ratio=1,L_X=0, nonthermal = False,R_star=0, idx_SolarWind=0.77,
                     frac=0.5):
        """
        Calculate the carbon mass loss rate for a given atmospheric composition.

        This method computes the rate of carbon mass loss from a planet's atmosphere
        due to XUV radiation-driven escape, based on the planet's orbital and physical
        properties, as well as the star's XUV flux and atmospheric composition.

        if nonthermal == True:
            R_star, L_X, idx_SolarWind are required
        Parameters:
        MMW (int): The mean molecular weight of the atmosphere (44 for CO2, 16 for CH4, 18 for H2O, 28, 29 for N2/O2)
        pl_orbsmax (float): The semi-major axis of the planet's orbit in AU
        pl_masse (float): The mass of the planet in Earth masses
        L_XUV (float): The XUV luminosity of the star in W
        pl_radiuse (float): The radius of the planet in Earth radii
        CO2_fit (str): The fitting method to use for CO2 escape rates ('linear' or 'log')
        Energy_limit (bool): Whether to limit the escape rate to the energy-limited regime
        epsilon (float): The efficiency factor for CH4 escape
        K_tide (float): The tidal heating factor for CH4 escape
        R_xuv_ratio (float): The ratio of the XUV absorption radius to the planet radius
        L_X (float): The X-ray luminosity of the star in W
        nonthermal (bool): Whether to calculate non-thermal escape
        R_star (float): The radius of the star in Solar radii
        idx_SolarWind (float): The power-law index for the mass loss rate scaling
        low_or_high (bool): If 'CR24' model is used, whether to use the low or high cut values
        frac (float): If 'CR24' model is used, the fraction of the range between the low and high cut values to use
        """

        
        F_xuv_to_earth= self._calculate_F_xuv_to_earth(L_XUV, pl_orbsmax)

        if L_X>0:
            Fx_Wm2 = self._calculate_F_xuv_to_earth(L_X, pl_orbsmax)*0.00464
        else:
            Fx_Wm2 = ""
        if nonthermal == True:
            return self.M_C_dot_non_thermal(L_X, R_star, pl_radiuse, pl_orbsmax, idx_SolarWind)
        else:
            if MMW == 44: #CO2
                return self.M_C_dot_CO2(F_xuv_to_earth, pl_masse, CO2_fit = CO2_fit, Energy_limit=Energy_limit, pl_radiuse=pl_radiuse)
            elif MMW == 28 or MMW == 29: #N2O2
                return self.M_C_dot_N2O2(F_xuv_to_earth, pl_masse, pl_radiuse=pl_radiuse, 
                                         model=N2O2_model,frac=frac)
            elif MMW == 18: #H2O
                return self.M_C_dot_H2O(F_xuv_to_earth, pl_masse, pl_radiuse=pl_radiuse,Fx_Wm2=Fx_Wm2)
            elif MMW == 16: #CH4
                return self.M_C_dot_CH4(F_xuv_to_earth, pl_masse, pl_radiuse=pl_radiuse, 
                                        epsilon=epsilon, K_tide=K_tide, R_xuv_ratio=R_xuv_ratio)
            elif MMW == 0: #Energy
                return self.M_C_dot_Energy(F_xuv_to_earth, pl_masse, 
                                        pl_radiuse=pl_radiuse, epsilon=epsilon, K_tide=K_tide, R_xuv_ratio=R_xuv_ratio)
        
    def integrate_carbon_loss(self, MMW, pl_orbsmax, pl_masse, st_mass, t1, dt, num_steps=2000, 
                              method="Jackson", pl_radiuse="", N2O2_model='N22',
                              CO2_fit='linear',Energy_limit=False, 
                                epsilon=0.1, K_tide=1, R_xuv_ratio=1,L_X=0,output='single',
                                beta1 = 116, beta2 = 3040, gamma1 = -0.35, gamma2 = -0.76,
                                frac=0.5):
        """
        Integrate the carbon loss from t1 to t1+dt to get the total carbon loss.

        Parameters:
        MMW (int): The mean molecular weight of the atmosphere (44 for CO2, 16 for CH4)
        pl_orbsmax (float): The semi-major axis of the planet's orbit in AU
        pl_masse (float): The mass of the planet in Earth masses
        pl_radiuse (float): The radius of the planet in Earth radii
        st_mass (float): The mass of the star in solar masses
        t1 (float): The starting time in years
        dt (float): The time interval to integrate over in years
        num_steps (int): The number of steps to use in the integration (default: 1000)

        Returns:
        float: The total carbon loss in kg
        """
        if pl_radiuse=="":
            pl_radiuse = self.M_R_fit(pl_masse)

        time_array = np.linspace(t1, t1 + dt, num_steps)
        L_XUV = self.calculate_L_XUV(st_mass, time_array, method, beta1=beta1, beta2=beta2, gamma1=gamma1, gamma2=gamma2)
        carbon_loss_rates = self.M_C_dot_loss(MMW, pl_orbsmax, pl_masse, L_XUV, 
                                              pl_radiuse=pl_radiuse,CO2_fit=CO2_fit,Energy_limit=Energy_limit, 
                                              epsilon=epsilon, K_tide=K_tide, R_xuv_ratio=R_xuv_ratio,L_X=L_X, N2O2_model=N2O2_model,
                                              frac=frac)
        # Calculate the total carbon loss using vectorized operations
        # print(carbon_loss_rates)
        total_carbon_loss = scipy.integrate.trapz(carbon_loss_rates,time_array) * 365.25 * 24 * 3600  # Convert years to seconds
        if output == 'single':
            return total_carbon_loss
        else:
            return time_array, carbon_loss_rates, total_carbon_loss
        
    def calculate_pl_teq(self, pl_orbsmax, st_mass, time=5e9, albedo=0.0):
        """
        Calculate the Equilibrium temperature of a planet based on its orbital distance.

        Parameters:
        pl_orbsmax (float): The semi-major axis of the planet's orbit in AU
        st_mass (float): The mass of the star in solar masses
        time (float): The age of the star in years (default: 1e9)
        albedo (float): The planet's albedo (default: 0.3)

        Returns:
        float: The Equilibrium temperature of the planet in K
        """
        sigma = 5.67e-8  # Stefan-Boltzmann constant in W/m^2/K^4
        S0 = 1361  # Solar constant in W/m^2
        L_bol_to_sun = 10**self.L_bol_interpolator_B15((st_mass, np.log10(time)))
        # Calculate the planet's Equilibrium temperature
        pl_teq = ((1/4) * L_bol_to_sun * (1-albedo) * S0 / (pl_orbsmax**2 * sigma))**(1/4)
        
        return pl_teq

    def calculate_pl_orbsmax(self, pl_teq, st_mass, time=5e9, albedo=0.0):
        """
        Calculate the orbital distance of a planet based on its Equilibrium temperature.

        Parameters:
        pl_teff (float): The Equilibrium temperature of the planet in K
        st_mass (float): The mass of the star in solar masses
        time (float): The age of the star in years (default: 1e9)
        albedo (float): The planet's albedo (default: 0.3)

        Returns:
        float: The semi-major axis of the planet's orbit in AU
        """
        sigma = 5.67e-8  # Stefan-Boltzmann constant in W/m^2/K^4
        S0 = 1361  # Solar constant in W/m^2
        L_bol_to_sun = 10**self.L_bol_interpolator_B15((st_mass, np.log10(time)))
        # Calculate the planet's orbital distance
        pl_orbsmax = ((1/4) * L_bol_to_sun * (1-albedo) * S0 / (pl_teq**4 * sigma))**(1/2)
        
        return pl_orbsmax
    
    def Ts_from_ps(self, pl_teq, pl_psurf, pl_masse, MMR, st_mass, pl_radiuse = "",p_RCB_cal=1,
        inversion=False,
        gamma=None,
        kappa0=0.1,    # Optical of CO2 at p0
        p0=1e4,   
        p_RCB=2.5e4,
        albedo=0.3):
        """
        Calculate the surface temperature for a given surface pressure using different methods.

        Parameters:
        pl_teq: Equilibrium temperature of the planet in K
        pl_psurf: surface pressure in Pa
        pl_masse: mass of the planet in Earth masses
        MMR: mean molecular weight of the atmosphere
        st_mass: mass of the star in solar masses
        pl_radiuse: radius of the planet in Earth radii
        p_RCB_cal: method to calculate the radiative-convective boundary pressure (default: 1)
            1: p_rcb is fixed
            2: p_rcb is calculated from opacity
        inversion: whether to consider thermal inversion for Ts calculation: isothermal layer below 100 bar (default: False)
        gamma: adiabatic index of the atmosphere (default: None)
                If None, the adiabatic index is calculated from the temperature-pressure profile
        kappa0: opacity at p0 in m^2/kg (only used if p_RCB_cal=2) (default: 0.1)
        p0: reference pressure in Pa where kappa0 is defined (default: 1e4)
        p_RCB: radiative-convective boundary pressure in Pa (default: 2.5e4)
        albedo: planet's albedo (default: 0.3)

        Returns:
        Ts: surface temperature in K
        """
        if pl_radiuse=="":
            pl_radiuse = self.M_R_fit(pl_masse)
        
        g = 9.8 * (pl_masse) / (pl_radiuse)**2  # Calculate g from pl_masse
        T_rcb = (1-albedo)**(0.25)*pl_teq
        if p_RCB_cal == 1:
            if gamma is None:
                if pl_psurf<p_RCB:
                    return T_rcb
                def integrate_temperature(p_rcb, T_rcb, p_surf):
                    def dry_adiabat_equation(p, T, gamma_interp):
                        """
                        dT/dp = (gamma - 1)/gamma * (T / p)
                        """
                        gamma_val = gamma_interp(T,np.log10(p))  # T is a 1-element array
                        return ((gamma_val - 1)/gamma_val) * (T/p)
                    def odefun(p, T):
                        return dry_adiabat_equation(p, T, self.interp_gamma_T_logP)
                    
                    sol = scipy.integrate.solve_ivp(
                        fun=odefun,
                        t_span=(p_rcb, p_surf), 
                        y0=[T_rcb],
                        method='RK45',  # or 'Radau' if you prefer
                        max_step=(p_surf - p_rcb)/10  # adjust if you want bigger steps
                    )
                    T_end = sol.y[0, -1]
                    return T_end
                if inversion:
                    if np.isscalar(pl_psurf):
                        Ts = integrate_temperature(p_RCB, T_rcb, min(pl_psurf,1e7))
                    else:
                        Ts = np.zeros_like(pl_psurf)
                        for n_psurf,pl_psurf_i in enumerate(pl_psurf):
                            Ts[n_psurf] = integrate_temperature(p_RCB, T_rcb, min(pl_psurf_i,1e7))
                else:
                    if np.isscalar(pl_psurf):
                        Ts = integrate_temperature(p_RCB, T_rcb, pl_psurf)
                    else:
                        Ts = np.zeros_like(pl_psurf)
                        for n_psurf,pl_psurf_i in enumerate(pl_psurf):
                            Ts[n_psurf] = integrate_temperature(p_RCB, T_rcb, pl_psurf_i)
            else: 
                k0 = (gamma - 1) / gamma
                if inversion:
                    Ts = np.where(
                            pl_psurf > 1e7,
                            (1e7 / p_RCB) ** k0 * T_rcb,
                            np.where(pl_psurf > p_RCB, (pl_psurf / p_RCB) ** k0 * T_rcb, T_rcb)
                        )
                else:
                    Ts = (pl_psurf / p_RCB) ** k0 * T_rcb if pl_psurf > p_RCB else T_rcb
        elif p_RCB_cal == 2:
            if MMR == 44:
                kappa0 = 0.1
            elif MMR == 16:
                kappa0 = 0.02
            p_RCB = np.sqrt(2 * g * p0 / kappa0)
            k0 = (gamma - 1) / gamma
            Ts = (pl_psurf / p_RCB) ** k0 * T_rcb if pl_psurf > p_RCB else T_rcb
        elif p_RCB_cal == 0:
            Ts = T_rcb
        else:
            raise ValueError('P_RCB calculation method not recognized')
        return Ts
    
    def calculate_z_rad(self,pl_rade, pl_masse, P_rad, P_surf, pl_teq, MMW, T_adjust = True, g_adjust = True,
                         inversion = False, P_rcb = 0.25*1e5, gamma = None, albedo=0.3):
        '''
        Calculate the transit radius of the planet based on a certain pressure.
        Parameters:
        pl_rade: radius of the planet in Earth radii
        pl_masse: mass of the planet in Earth masses
        P_rad: radiative pressure in Pa (scalar)
        P_surf: surface pressure in Pa (scalar or array)
        pl_teq: Equilibrium temperature of the planet in K (albedo = 0)
        MMW: mean molecular weight of the atmosphere
        T_adjust: whether to adjust the temperature for the calculation (default: True)
        g_adjust: whether to adjust the gravity for the calculation (default: True)
        inversion: whether to consider thermal inversion for Ts calculation: isothermal layer below 100 bar (default: False)
        P_rcb: radiative-convective boundary pressure in Pa (default: 2.5e4) (scalar)
        gamma: adiabatic index of the atmosphere (default: None)
               If None, the adiabatic index is calculated from the temperature-pressure profile
        albedo: planet's bond albedo (default: 0.3)
        Returns:
        z_rad: transit radius of the planet in m
        '''
        T_rcb = (1-albedo)**(0.25)*pl_teq
        if gamma is None:
            G=6.67430e-11
            R_gas = 8.314/(MMW*1e-3)

            def integrate_isothermal_analytic(one_r_low, p_low, p_high, T_rcb, R_gas, pl_masse):
                """
                Returns the 1/r_high for an isothermal layer
                spanning from p_low to p_high
                """
                # Delta z = (R_s*T_iso / g) * ln(p_start/p_end).
                return np.log(p_high / p_low)/(G*pl_masse*self.earth_mass/(R_gas*T_rcb))+one_r_low
            
            def dry_adiabat_convective_ode(p, y, gamma_fn, R_gas, pl_masse):
                """
                ODE for convective (dry adiabatic) region.
                p : float (independent variable)
                y : [T, z]
                T = temperature at pressure p
                z = height at pressure p
                returns d[T, 1/r]/dp
                """
                T, one_r = y
                gamma_val = gamma_fn(T,np.log10(p)) # Evaluate gamma at (p, T)
                # (Check for NaN or out-of-bounds if needed.)
                
                dTdp = (gamma_val - 1)/gamma_val * T/p
                d1_rdp = (R_gas * T) / (p * G * pl_masse * self.earth_mass)
                return [dTdp, d1_rdp]
            

            def ode_wrap(p, y):
                return dry_adiabat_convective_ode(p, y, self.interp_gamma_T_logP, R_gas, pl_masse)
            
            def get_z_atm(pl_P_surf):
                if (inversion == True) and pl_P_surf>1e7:
                    # calculate the temperature, and (1/r_base-1/_r_rcb) at the base of convective layer
                    # Solve the ODE
                    sol_ada = scipy.integrate.solve_ivp(
                        fun=ode_wrap,
                        t_span=(P_rcb, 1e7),
                        y0=[T_rcb, 0],
                        method='RK45',
                        # You can tweak tolerances for speed/accuracy:
                        # rtol=1e-5, atol=1e-7,
                        dense_output=True
                    )
                    d_1_r_rcb_to_base = sol_ada.y[1, -1]
                    T_surf = sol_ada.y[0, -1]

                    # Calculate the 1/r_base
                    one_r_base = integrate_isothermal_analytic(1/(pl_rade*self.earth_radius), pl_P_surf, 1e7, T_surf, R_gas, pl_masse)
                    one_r_rcb = one_r_base - d_1_r_rcb_to_base
                    one_r_rad = integrate_isothermal_analytic(one_r_rcb , P_rcb, P_rad, T_rcb, R_gas, pl_masse)
                    return 1/one_r_rad
                else:
                    # calculate the temperature, and (1/r_surf-1/_r_rcb) at the base of convective layer
                    # Solve the ODE
                    sol_ada = scipy.integrate.solve_ivp(
                        fun=ode_wrap,
                        t_span=(P_rcb, pl_P_surf),
                        y0=[T_rcb, 0],
                        method='RK45',
                        # You can tweak tolerances for speed/accuracy:
                        # rtol=1e-5, atol=1e-7,
                        dense_output=True
                    )
                    one_r_rcb = 1/(pl_rade*self.earth_radius) - sol_ada.y[1, -1]
                    T_surf = sol_ada.y[0, -1]
                    one_r_rad = integrate_isothermal_analytic(one_r_rcb , P_rcb, P_rad, T_rcb, R_gas, pl_masse)

                    return 1/one_r_rad 
            if np.isscalar(P_surf):
                if P_rcb > P_surf:
                    result = pl_rade*self.earth_radius
                else:
                    result = get_z_atm(P_surf)
            else:
                result = np.zeros_like(P_surf)
                for n_P_surf, P_surf_i in enumerate(P_surf):
                    if P_rcb > P_surf_i:
                        result[n_P_surf] = pl_rade*self.earth_radius
                    else:
                        result[n_P_surf] = get_z_atm(P_surf_i)
        else:
            G=6.67430e-11
            M = pl_masse * self.earth_mass
            R = pl_rade * self.earth_radius
            kappa = (gamma-1)/gamma
            R_gas = 8.314/(MMW*1e-3)
            
            # Convert inputs to NumPy arrays to handle both scalar and array cases
            P_surf = np.asarray(P_surf)

            # Initialize result array with R where P_rad > P_surf
            result = np.where(P_rad > P_surf, R, np.nan)

            # For elements where P_rad <= P_surf, perform the calculation
            mask = P_rad <= P_surf

            if T_adjust:
                if inversion:
                    term1 = np.log(P_rad / P_rcb)
                    term2 = (1 / kappa) * (1 - (1e7 / P_rcb) ** kappa)
                    term3 = np.log(1e7 / P_surf[mask])

                    numerator = (1 / R) + (R_gas * T_rcb) / (G * M) * (term1 + term2 + term3)
                    r_rad = 1 / numerator
                    result[mask] = r_rad
                else:
                    term1 = np.log(P_rad / P_rcb)
                    term2 = (1 / kappa) * (1 - (P_surf[mask] / P_rcb) ** kappa)
                    numerator = (1 / R) + (R_gas * T_rcb) / (G * M) * (term1 + term2)
                    r_rad = 1 / numerator
                    result[mask] = r_rad
            else:
                if g_adjust:
                    numerator = (1 / R) + (R_gas * T_rcb) / (G * M) * np.log(P_rad / P_surf[mask])
                    r_rad = 1 / numerator
                    result[mask] = r_rad
                elif g_adjust == False:
                    g = 9.8 * pl_masse / (pl_rade ** 2)
                    z_rad = R_gas * T_rcb / g * np.log(P_surf[mask] / P_rad)
                    result[mask] = z_rad + R
        return result
    
    def cal_total_C_in_pl_mass_from_f_c(self, f_c_magma, pl_teq,
                                         pl_masse, MMR, st_mass, pl_radiuse="",
                                         albedo=0.3,gamma=None, inversion=False, 
                                         p_RCB=0.25*1e5):
        """
        Calculate the total carbon mass in a planet's mass from the solubility of carbon in the magma.


        Parameters:
        f_c_magma: The fraction of carbon in the magma
        pl_teq: The equilibrium temperature of the planet in K
        pl_masse: The mass of the planet in Earth masses
        MMR: The mean molecular weight of the atmosphere (44 for CO2, 16 for CH4)
        st_mass: The mass of the star
        pl_radiuse: The radius of the planet in Earth radii
        albedo: The albedo of the planet
        gamma: The adiabatic index of the atmosphere
               If None, the gamma is calculated from the temperature and pressure
        """
        if pl_radiuse=="":
            pl_radiuse = self.M_R_fit(pl_masse)

        pl_g = 9.8*(pl_masse)/(pl_radiuse)**2

        if MMR == 44:
            alpha = 1.937e-9 / 1e6
            beta = 0.714
        elif MMR == 16:
            alpha = 9.937e-8 / 1e6
            beta = 1
        elif MMR == 29:
            alpha = 7e-5 / 1e6
            beta = 1.800
        elif MMR == 18:
            alpha = 1.033 / 1e6
            beta = 1.747
        else:
            raise ValueError("MMR must be CO2 CH4 N2O2 or H2O")

        psurf = (f_c_magma / alpha) ** beta
        Tsurf = self.Ts_from_ps(pl_teq, psurf, pl_masse, MMR, st_mass,gamma=gamma,
                                inversion=inversion, albedo=albedo, p_RCB=p_RCB)
        
        M_magma = self.cal_m_magma(Tsurf, pl_masse)
        
        C_magma = f_c_magma * M_magma
        C_atm = 12 / MMR * psurf / pl_g * 4 * np.pi * (pl_radiuse * 6.37e6) ** 2
        if MMR == 29 or 28:
            C_atm = 28 / MMR * psurf / pl_g * 4 * np.pi * (pl_radiuse * 6.37e6) ** 2
        if MMR == 18:
            C_atm = psurf / pl_g * 4 * np.pi * (pl_radiuse * 6.37e6) ** 2
        C_tot = C_magma + C_atm

        return C_tot / (pl_masse * self.earth_mass)
    
    def cal_m_magma(self, Tsurf, pl_masse):
        """
        Calculate the mass of the magma for a given surface temperature and planet mass.

        Parameters:
        Tsurf: The surface temperature of the planet in K
        pl_masse: The mass of the planet in Earth masses

        Returns:
        The mass of the magma in kg
        """
        pl_masses = np.arange(0.5, 10.01, 0.1)
        arg = np.argmin(np.abs(pl_masses - pl_masse))
        magma_itpltor = self.M_magma_interpolators_0d5_to_10_Me_d0d1[arg]
        M_magma = magma_itpltor(Tsurf)

        M_magma = np.where(M_magma<0,0, M_magma)
        M_magma = np.minimum(M_magma, 2/3*pl_masse*self.earth_mass)
        return M_magma
    
    def Ps_Ts_from_f_c(self,f_c_magma,pl_teq,pl_masse,MMR,st_mass,f_C,pl_radiuse="",p_RCB = 0.25*1e5, albedo=0.3,gamma=None,inversion=False):
        """
        Calculate the surface pressure and temperature for a given fraction of carbon in the magma
        and the total carbon mass in the planet's mass.

        Parameters:
        f_c_magma: carbon concentration in magma. Carbon mass in magma/magma mass
        pl_teq: The equilibrium temperature of the planet in K
        pl_masse: The mass of the planet in Earth masses
        MMR: The mean molecular weight of the atmosphere (44 for CO2, 16 for CH4)
        st_mass: The mass of the star
        f_C: The fraction of the planet's mass that is carbon
        pl_radiuse: The radius of the planet in Earth radii
        p_RCB: The radiative-convective boundary pressure in Pa
        albedo: The albedo of the planet
        gamma: The adiabatic index of the atmosphere
               If None, the gamma is calculated from the temperature and pressure
        inversion: whether to consider thermal inversion for Ts calculation: isothermal layer below 100 bar (default: False)

        Returns:
        psurf: The surface pressure of the planet in Pa
        Tsurf: The surface temperature of the planet in K
        """
        #f_C is the fraction of the planet's mass that is C
        if pl_radiuse=="":
            pl_radiuse = self.M_R_fit(pl_masse)

        pl_g = 9.8*(pl_masse)/(pl_radiuse)**2

        if MMR==44:
            # C_tot: kg
            alpha=1.937e-9/1e6
            beta=0.714

            
        elif MMR==16:
            # C_tot: kg
            alpha=9.937e-8/1e6
            beta=1
            
            
        elif MMR==29:
            # C_tot: kg
            alpha=7e-5/1e6
            beta=1.8
            
            
        elif MMR==18:
            # C_tot: kg
            alpha=1.033/1e6
            beta=1.747
            
        psurf = (f_c_magma/alpha)**beta
        Tsurf = self.Ts_from_ps(pl_teq, psurf, pl_masse, MMR, st_mass, 
                                p_RCB=p_RCB,gamma=gamma,inversion=inversion,albedo=albedo)
        
        M_magma = self.cal_m_magma(Tsurf, pl_masse)
        if M_magma<=0:
            m_atm = f_C*pl_masse*self.earth_mass*MMR/12
            psurf  = m_atm*pl_g/(4*np.pi*(pl_radiuse*6.37e6)**2)
            Tsurf = self.Ts_from_ps(pl_teq, psurf, pl_masse, MMR, st_mass, 
                                p_RCB=p_RCB,gamma=gamma,inversion=inversion,albedo=albedo)
        
        # if f_c_magma>0.99:
        #     m_atm = f_C*pl_masse*self.earth_mass-f_c_magma*2/3*pl_masse*self.earth_mass
        #     psurf = m_atm*pl_g/(4*np.pi*(pl_radiuse*6.37e6)**2)
        #     # psurf = min(psurf,1e9)
        #     # pl_teff = min(pl_teff,2500)
        #     Tsurf = self.Ts_from_ps(pl_teff, psurf, pl_masse, MMR, st_mass)
        return psurf,Tsurf
    
    def get_TPz_profile(self, pl_rade, pl_masse, P_rad, P_surf, pl_teq, MMW, inversion=False, 
                    P_rcb=0.25*1e5, gamma=None, albedo=0.3):
        """
        Calculate the temperature-pressure profile for a given surface pressure.

        Parameters:
        pl_rade: radius of the planet in Earth radii
        pl_masse: mass of the planet in Earth masses
        P_rad: radiative pressure in Pa
        P_surf: surface pressure in Pa
        pl_teq: equilibrium temperature of the planet in K
        MMW: mean molecular weight of the atmosphere
        inversion: whether to consider thermal inversion for Ts calculation: isothermal layer below 100 bar (default: False)
        P_rcb: radiative-convective boundary pressure in Pa (default: 2.5e4)
        gamma: adiabatic index of the atmosphere (default: None)
                If None, the adiabatic index is calculated from the temperature-pressure profile

        Returns:
        T_output: temperature profile in K
        z_output: altitude profile in m
        """

        G = 6.67430e-11
        R_gas = 8.314/(MMW*1e-3)
        T_rcb = (1-albedo)**(0.25)*pl_teq
        P_output = np.logspace(np.log10(P_rad), np.log10(P_surf), 100)
        T_output = np.zeros_like(P_output)
        z_output = np.zeros_like(P_output)

        def integrate_isothermal_analytic(one_r_low, p_low, p_high, T_eff, R_gas, pl_masse):
            """
            Returns the 1/r_high for an isothermal layer
            spanning from p_low to p_high
            """
            # Delta z = (R_s*T_iso / g) * ln(p_start/p_end).
            return np.log(p_high / p_low)/(G*pl_masse*self.earth_mass/(R_gas*T_eff))+one_r_low
        
        def dry_adiabat_convective_ode(p, y, gamma_fn, R_gas, pl_masse):
            """
            ODE for convective (dry adiabatic) region.
            p : float (independent variable)
            y : [T, z]
            T = temperature at pressure p
            z = height at pressure p
            returns d[T, 1/r]/dp
            """
            T, one_r = y
            gamma_val = gamma_fn(T,np.log10(p)) # Evaluate gamma at (p, T)
            # (Check for NaN or out-of-bounds if needed.)

            dTdp = (gamma_val - 1)/gamma_val * T/p
            d1_rdp = (R_gas * T) / (p * G * pl_masse * self.earth_mass)
            return [dTdp, d1_rdp]

        def ode_wrap(p, y):
            return dry_adiabat_convective_ode(p, y, self.interp_gamma_T_logP, R_gas, pl_masse)

        if (inversion == True) and P_surf>1e7:
            # calculate the temperature, and (1/r_base-1/_r_rcb) at the base of convective layer
            # Solve the ODE
            sol_ada = scipy.integrate.solve_ivp(
                fun=ode_wrap,
                t_span=(P_rcb, 1e7),
                y0=[T_rcb, 0],
                t_eval=P_output[(P_output<=1e7)&(P_output>=P_rcb)],
                method='RK45',
                # You can tweak tolerances for speed/accuracy:
                # rtol=1e-5, atol=1e-7,
                dense_output=True
            )
            T_output[(P_output<=1e7)&(P_output>=P_rcb)] = sol_ada.y[0, :]
            d_1_r_rcb_to_base = sol_ada.y[1, ::-1]

            T_surf = sol_ada.y[0, -1]
            one_r_base = integrate_isothermal_analytic(1/(pl_rade*self.earth_radius), P_surf, 1e7, T_surf, R_gas, pl_masse)
            one_r_rcb = one_r_base - sol_ada.y[1, -1]
            z_output[(P_output<=1e7)&(P_output>=P_rcb)] = 1/(one_r_base - d_1_r_rcb_to_base)-pl_rade*self.earth_radius
             


            T_output[P_output>1e7] = T_surf
            z_output[P_output>1e7] = 1/integrate_isothermal_analytic(1/(pl_rade*self.earth_radius), 
                                                                    P_surf, P_output[P_output>1e7], 
                                                                    T_surf, R_gas, pl_masse)-pl_rade*self.earth_radius
            T_output[P_output<P_rcb] = T_rcb
            z_output[P_output<P_rcb] = 1/integrate_isothermal_analytic(one_r_rcb,
                                                                    P_rcb, P_output[P_output<P_rcb],
                                                                        T_rcb, R_gas, pl_masse)-pl_rade*self.earth_radius
        else:
            sol_ada = scipy.integrate.solve_ivp(
                fun=ode_wrap,
                t_span=(P_rcb, P_surf),
                y0=[T_rcb, 0],
                t_eval=P_output[(P_output<=P_surf)&(P_output>=P_rcb)],
                method='RK45',
                # You can tweak tolerances for speed/accuracy:
                # rtol=1e-5, atol=1e-7,
                dense_output=True
            )
            T_output[(P_output>=P_rcb)] = sol_ada.y[0, :]
            d_1_r_rcb_to_base = sol_ada.y[1, ::-1]

            z_output[(P_output>=P_rcb)] = 1/(1/(pl_rade*self.earth_radius) - d_1_r_rcb_to_base) - pl_rade*self.earth_radius
            
            one_r_rcb = 1/(pl_rade*self.earth_radius) - sol_ada.y[1,-1]

            T_output[P_output<P_rcb] = T_rcb
            z_output[P_output<P_rcb] = 1/integrate_isothermal_analytic(one_r_rcb,
                                                                    P_rcb, P_output[P_output<P_rcb],
                                                                        T_rcb, R_gas, pl_masse)-pl_rade*self.earth_radius
        return P_output, T_output, z_output
    

# -----------------------------------------------------------------------------
# Example usage when run as a script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Create an instance of CosmicShoreline (update the data_path if necessary)
    cs = CosmicShoreline()
    
    # Example: Calculate the stellar XUV luminosity for a solar-mass star at 1 Gyr
    star_mass = 1.0
    time_XUV = 1e9  # years
    L_XUV = cs.calculate_L_XUV(star_mass, time_XUV)
    print("Calculated L_XUV:", L_XUV)
    
    # You can add further examples to test other methods.
