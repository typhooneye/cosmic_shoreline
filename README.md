# The Cosmic Shoreline Revisited

This repository accompanies the paper:

X. Ji et al., *The Cosmic Shoreline Revisited: A Metric for Atmospheric Retention Informed by Hydrodynamic Escape*  
DOI: `10.3847/1538-4357/adfe69`  
arXiv: <https://arxiv.org/abs/2504.19872>

## Purpose

This codebase is intended to reproduce the main calculations and figures in the paper. 

## Web Calculator

https://typhooneye.github.io/cosmic_shoreline/

The `docs/` folder contains an interactive HTML-based calculator for exploring cumulative atmospheric loss and cumulative XUV exposure.

The calculator is useful for quick comparisons: Hydrodynamic escape depends nonlinearly on XUV flux, so cumulative XUV alone is not enough to determine cumulative atmospheric loss. A planet with lower cumulative XUV does not necessarily experience less total atmospheric escape.

## Repository Contents

- `cosmic_shoreline.py`: core model implementation
- `data-interpolation/`: interpolation tables used by the model
- `data-montecarlo/`: Monte Carlo outputs and derived loss-rate products
- `exoplanets_data/`: archived exoplanet tables used in the analysis
- `docs/`: browser-based interactive calculator

