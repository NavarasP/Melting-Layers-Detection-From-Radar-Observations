"""Utility functions: conversions, small helpers."""
import numpy as np

def dbz_to_linear(dbz):
    return 10.0 ** (dbz / 10.0)

def linear_to_dbz(Z):
    return 10.0 * np.log10(np.maximum(Z, 1e-12))

def running_mean(a, window=3):
    import numpy as np
    if window <= 1:
        return a
    pad = window // 2
    a_p = np.pad(a, pad, mode='edge')
    out = np.convolve(a_p, np.ones(window)/window, mode='valid')
    return out
