"""Radiance Enhancement Module (simplified)
 - builds a small linear filter using simulated DPR-resolution TBs convolved to GMI resolution
 - applies a deconvolution-style correction to observed GMI TBs to produce DPR-resolution TB estimates
This is a regression-based approach similar in spirit to ATBD's description but simpler.
"""
import numpy as np
from scipy.linalg import lstsq

def build_enhancement_filter(TB_sim_highres, TB_sim_lowres, n_pixels=9):
    # TB_sim_highres: (n_samples, nchannels) simulated at DPR res
    # TB_sim_lowres: (n_samples, nchannels) convolved to GMI res (neighborhood)
    # We solve for linear mapping: TB_high = A * TB_low + b  -> A via least squares per channel
    nch = TB_sim_highres.shape[1]
    A = np.zeros((nch, nch))
    b = np.zeros(nch)
    # solve small multivariate regression per channel independently
    for c in range(nch):
        Y = TB_sim_highres[:, c]
        X = TB_sim_lowres  # shape (n_samples, nch)
        coef, *_ = lstsq(X, Y)
        A[c, :] = coef
        # compute residual intercept
        b[c] = np.mean(Y - X.dot(coef))
    return A, b

def apply_enhancement(TB_obs_lowres, A, b):
    # TB_obs_lowres: (nch, ) observed vector at GMI resolution for neighborhood center (flattened)
    return A.dot(TB_obs_lowres) + b

if __name__ == '__main__':
    # quick demonstration
    nch = 9
    ns = 200
    TB_high = np.random.randn(ns, nch) * 3 + 250
    TB_low = TB_high.mean(axis=0) + 0.5*np.random.randn(ns, nch)
    A, b = build_enhancement_filter(TB_high, TB_low)
    tb_est = apply_enhancement(TB_low[0], A, b)
    print('tb_est shape', tb_est.shape)
