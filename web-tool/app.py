#!/usr/bin/env python3
"""
Flask backend for the Cosmic Shoreline atmospheric loss calculator.
Run with: python app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from cosmic_shoreline import CosmicShoreline

app = Flask(__name__, static_folder='.')
CORS(app)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE, 'data-interpolation/')
CSV_PATH   = os.path.join(BASE, 'exoplanets_data/TSM/NASAExoArchive_2025-02-28_aggregate.csv')

cs = CosmicShoreline(data_path=DATA_PATH)

# ── Pre-build df_rocks (matches notebook §2.2) ──────────────────────────────
def build_df_rocks():
    df = pd.read_csv(CSV_PATH, keep_default_na=True, low_memory=False)
    df['pl_masse']     = np.where(df['pl_bmasselim'] == 1, np.nan, df['pl_bmasse'])
    df['pl_masseerr1'] = df['pl_bmasseerr1']
    df['pl_masseerr2'] = df['pl_bmasseerr2']
    df = df[(df['pl_masse'] < 10) | (df['pl_masse'].isna())]

    masse_calc = cs.M_R_fit(df['pl_rade'].to_numpy(), x_M_or_R='R')
    df['pl_masse_calc'] = np.where(df['pl_masse'].isna(), masse_calc, df['pl_masse'])
    df['pl_rho_ratio']  = cs.M_R_fit(df['pl_masse'].to_numpy(), x_M_or_R='M')**3 / df['pl_rade']**3

    mask_low = (
        (df['st_mass'] < 0.6) &
        (((df['pl_rade'] < 1.6) & df['pl_masse'].isna()) |
         ((df['pl_rho_ratio'] > 0.6) & (df['pl_masse'] < 10)))
    )
    mask_mid = (
        (df['st_mass'] >= 0.6) & (df['st_mass'] < 1.4) &
        (np.log10(df['pl_rade']) < (-0.11 * np.log10(df['pl_orbper']) + 0.37)) &
        (((df['pl_rade'] < 1.6) & df['pl_masse'].isna()) | ~df['pl_masse'].isna())
    )
    rocks = df[mask_low | mask_mid].copy()
    G = 6.67430e-11; earth_mass = 5.972e24; earth_radius = 6.371e6
    rocks['pl_vesc'] = (
        np.sqrt(2 * G * earth_mass * rocks['pl_masse_calc'] /
                (rocks['pl_rade'] * earth_radius)) / 1e3
    )
    return df, rocks   # df = full archive (mass-filtered), rocks = rocky subset


print("Loading NASA Exoplanet Archive…", flush=True)
df_archive, df_rocks = build_df_rocks()
print(f"  archive: {len(df_archive)} rows  |  df_rocks: {len(df_rocks)} rows", flush=True)


# ── Atmosphere config ────────────────────────────────────────────────────────
ATM_CONFIG = {
    'CO2':       {'MMW': 44},
    'CH4':       {'MMW': 16},
    'H2O':       {'MMW': 18},
    'N2O2_N22':  {'MMW': 28, 'N2O2_model': 'N22'},
    'N2O2_CP24': {'MMW': 28, 'N2O2_model': 'CP24'},
}


def _lxuv_from_custom(st_mass, time_array, xray_params,
                      beta1=116, beta2=3040, gamma1=-0.35, gamma2=-0.76):
    """
    Compute L_XUV (W) from a custom piecewise power-law X-ray prescription
    (Lammer et al. 2009 style) plus King & Wheatley 2020 EUV scaling.

    L_X(t) = k_sat   * L0 * (t/Gyr)^idx_sat    for t < t_sat
    L_X(t) = k_unsat * L0 * (t/Gyr)^idx_unsat  for t >= t_sat
    """
    k_sat     = xray_params['k_sat']
    k_unsat   = xray_params['k_unsat']
    L0_ergs   = 10 ** xray_params['log10_L0']   # erg/s
    idx_sat   = xray_params['idx_sat']
    idx_unsat = xray_params['idx_unsat']
    t_sat_Gyr = xray_params['t_sat']

    t_Gyr = time_array / 1e9
    L_X_ergs = np.where(
        t_Gyr < t_sat_Gyr,
        k_sat   * L0_ergs * t_Gyr ** idx_sat,
        k_unsat * L0_ergs * t_Gyr ** idx_unsat,
    )
    L_X_W = L_X_ergs / 1e7   # erg/s → W

    # Bolometric luminosity and stellar radius from Baraffe+2015 grid
    L_bol = 3.846e26 * 10 ** cs.L_bol_interpolator_B15(
        (st_mass, np.log10(time_array))
    )
    Rs = cs.R_star_interpolator_B15((st_mass, np.log10(time_array)))
    As = 4 * np.pi * (Rs * 6.96e8) ** 2 * 1e4   # cm²

    L_X_to_bol = L_X_W / L_bol
    # King & Wheatley 2020 EUV scaling
    L_EUV1 = beta1 * (L_bol * 1e7 / As) ** gamma1 * L_X_to_bol ** (gamma1 + 1) * L_bol
    L_EUV2 = beta2 * (L_bol * 1e7 / As) ** gamma2 * L_X_to_bol ** (gamma2 + 1) * L_bol
    return L_X_W + L_EUV1 + L_EUV2


def compute_one(atm_key, pl_orbsmax, pl_masse, st_mass, t1, dt,
                pl_radiuse='', num_steps=200, want_series=True,
                xray_model='Jackson', xray_params=None):
    """Return dict with total_loss + optionally downsampled time series."""
    import scipy.integrate as sci_int
    cfg = ATM_CONFIG[atm_key]
    loss_kw = {}
    if pl_radiuse not in ('', None):
        loss_kw['pl_radiuse'] = float(pl_radiuse)
    if 'N2O2_model' in cfg:
        loss_kw['N2O2_model'] = cfg['N2O2_model']

    if xray_model == 'custom' and xray_params:
        time_array = np.linspace(t1, t1 + dt, num_steps)
        L_XUV = _lxuv_from_custom(st_mass, time_array, xray_params)
        rates = cs.M_C_dot_loss(cfg['MMW'], pl_orbsmax, pl_masse, L_XUV, **loss_kw)
        total = sci_int.trapezoid(rates, time_array) * 365.25 * 24 * 3600
        t_arr = time_array
    else:
        method = 'Selsis' if xray_model == 'Selsis' else 'Jackson'
        kwargs = dict(
            MMW=cfg['MMW'], pl_orbsmax=pl_orbsmax, pl_masse=pl_masse,
            st_mass=st_mass, t1=t1, dt=dt, num_steps=num_steps,
            output='multi', method=method, **loss_kw,
        )
        t_arr, rates, total = cs.integrate_carbon_loss(**kwargs)

    result = {
        'total_loss_kg':        float(total),
        'total_loss_earth_atm': float(total / 5.1e18),
    }
    if want_series:
        step = max(1, len(t_arr) // 200)
        result['time_array'] = t_arr[::step].tolist()
        result['loss_rates'] = rates[::step].tolist()
    return result


def parse_array(raw):
    """Parse a string like '1.0' or '1.0, 2.0, 3.0' into a list of floats."""
    if isinstance(raw, (int, float)):
        return [float(raw)]
    parts = [x.strip() for x in str(raw).replace(';', ',').split(',')]
    return [float(p) for p in parts if p]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/planet_names')
def planet_names():
    """All planet names available in the archive (for autocomplete)."""
    names = sorted(df_archive['pl_name'].dropna().unique().tolist())
    return jsonify(names)


@app.route('/df_rocks_meta')
def df_rocks_meta():
    """Metadata for every planet in df_rocks (for the scatter plot)."""
    valid = df_rocks[
        df_rocks['pl_orbsmax'].notna() &
        df_rocks['pl_masse_calc'].notna() &
        df_rocks['st_mass'].notna() &
        df_rocks['pl_vesc'].notna()
    ].copy()
    cols = ['pl_name', 'pl_orbsmax', 'pl_masse_calc', 'pl_rade', 'st_mass', 'pl_vesc']
    records = valid[cols].rename(columns={'pl_masse_calc': 'pl_masse'}).to_dict(orient='records')
    return jsonify(records)


@app.route('/lookup_planets', methods=['POST'])
def lookup_planets():
    """Look up planet parameters by name from the full archive."""
    names = request.get_json().get('names', [])
    results = []
    for name in names:
        row = df_archive[df_archive['pl_name'] == name]
        if row.empty:
            results.append({'pl_name': name, 'error': 'Not found in archive'})
            continue
        row = row.iloc[0]
        pl_masse = row['pl_masse_calc'] if 'pl_masse_calc' in row.index else (
            row['pl_bmasse'] if pd.notna(row.get('pl_bmasse')) else
            cs.M_R_fit(float(row['pl_rade']), x_M_or_R='R') if pd.notna(row.get('pl_rade')) else None
        )
        if pl_masse is None or pd.isna(pl_masse):
            results.append({'pl_name': name, 'error': 'Could not determine planet mass'})
            continue
        if pd.isna(row.get('pl_orbsmax')):
            results.append({'pl_name': name, 'error': 'Missing orbital distance (pl_orbsmax)'})
            continue
        results.append({
            'pl_name':    name,
            'pl_orbsmax': float(row['pl_orbsmax']),
            'pl_masse':   float(pl_masse),
            'pl_rade':    float(row['pl_rade']) if pd.notna(row.get('pl_rade')) else None,
            'st_mass':    float(row['st_mass'])  if pd.notna(row.get('st_mass')) else None,
        })
    return jsonify(results)


@app.route('/calculate', methods=['POST'])
def calculate():
    """
    Unified calculation endpoint.
    Body:
      {
        planets: [{ label, pl_orbsmax, pl_masse, st_mass, pl_radiuse? }],
        atmospheres: ['CO2', ...],
        t1: float,
        dt: float,
        want_series: bool  (false for df_rocks scatter mode)
      }
    """
    data = request.get_json()
    try:
        planets    = data['planets']          # list of planet dicts
        atmospheres = data.get('atmospheres', ['CO2'])
        t1         = float(data.get('t1', 1e6))
        dt         = float(data.get('dt', 5e9))
        want_series = data.get('want_series', True)
        num_steps  = 100 if not want_series else 200

        out = []
        for planet in planets:
            pl_orbsmax = float(planet['pl_orbsmax'])
            pl_masse   = float(planet['pl_masse'])
            st_mass    = float(planet['st_mass'])
            pl_radiuse = planet.get('pl_radiuse') or ''
            label      = planet.get('label', 'Planet')

            xray_model  = planet.get('xray_model', 'Jackson')
            xray_params = planet.get('xray_params') or None

            atm_results = {}
            atm_errors  = {}
            for atm in atmospheres:
                if atm not in ATM_CONFIG:
                    atm_errors[atm] = f'Unknown: {atm}'
                    continue
                try:
                    atm_results[atm] = compute_one(
                        atm, pl_orbsmax, pl_masse, st_mass, t1, dt,
                        pl_radiuse=pl_radiuse, num_steps=num_steps,
                        want_series=want_series,
                        xray_model=xray_model, xray_params=xray_params,
                    )
                except Exception as e:
                    atm_errors[atm] = str(e)

            out.append({
                'label': label,
                'pl_vesc': planet.get('pl_vesc'),  # pass through for scatter
                'atmospheres': atm_results,
                'errors': atm_errors,
            })

        return jsonify({'status': 'ok', 'planets': out})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/xray_preview', methods=['POST'])
def xray_preview():
    """Compute L_X (X-ray only, erg/s) vs time for the illustration panel.
    X-axis is always fixed: 1e6 – 5e9 yr."""
    data = request.get_json()
    try:
        st_mass     = float(data.get('st_mass', 1.0))
        xray_model  = data.get('xray_model', 'Jackson')
        xray_params = data.get('xray_params')

        # Fixed time axis regardless of integration window
        time_array = np.linspace(1e6, 5e9, 300)

        if xray_model == 'custom' and xray_params:
            t_Gyr   = time_array / 1e9
            L0_ergs = 10 ** xray_params['log10_L0']
            L_X_ergs = np.where(
                t_Gyr < xray_params['t_sat'],
                xray_params['k_sat']   * L0_ergs * t_Gyr ** xray_params['idx_sat'],
                xray_params['k_unsat'] * L0_ergs * t_Gyr ** xray_params['idx_unsat'],
            )
        else:
            method      = 'Selsis' if xray_model == 'Selsis' else 'Jackson'
            L_X_to_bol  = cs.calculate_L_X_to_bol(st_mass, time_array, method=method)
            L_bol       = 3.846e26 * 10 ** cs.L_bol_interpolator_B15(
                              (st_mass, np.log10(time_array)))
            L_X_ergs    = L_X_to_bol * L_bol * 1e7   # W → erg/s

        return jsonify({
            'status':    'ok',
            'time_array': time_array.tolist(),
            'L_X_ergs':   L_X_ergs.tolist(),
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '__main__':
    print("Starting Cosmic Shoreline web tool at http://localhost:5050")
    app.run(debug=True, port=5050)
