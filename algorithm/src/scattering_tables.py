"""Simplified scattering & absorption tables.
This module creates lightweight parametric tables for extinction (k) given Z,
and for brightness temperature emissivity mapping. In the full ATBD, these are
precomputed Mie/DDA-based tables indexed by mu, Nw, Dm, temperature, etc.
Here we provide simple, smooth functions and small lookup arrays for demonstration.
"""
import numpy as np
from scipy.interpolate import interp1d

def build_simple_k_table():
    # Build a simple k(Z) table: Z in mm6/m3 -> k (1/km)
    Zvals = np.logspace(-2, 6, 200)
    # empirical-like power-law coefficients that vary with Z
    a = 1e-4
    b = 0.9
    kvals = a * Zvals ** b
    return interp1d(Zvals, kvals, bounds_error=False, fill_value=(kvals[0], kvals[-1]))

def lookup_k_from_Z(Z, k_interp):
    # Z may contain zeros; avoid issues
    return k_interp(np.maximum(Z, 1e-8))

def build_emissivity_lookup():
    # returns a simple function emissivity(surface_type, freq_idx, wind, Ts)
    def emissivity(surface='ocean', freq_idx=0, wind=5.0, Ts=300.0):
        # coarse model: emissivity increases slightly with frequency and wind
        base = 0.3 + 0.05*freq_idx
        if surface == 'ocean':
            return np.clip(base + 0.01*(wind-5.0), 0.02, 0.99)
        else:
            # land more variable
            return np.clip(base + 0.02*np.sin(freq_idx), 0.05, 0.99)
    return emissivity
