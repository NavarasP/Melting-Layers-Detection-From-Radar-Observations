"""Ku Radar Module (simplified implementation)
 - ensemble generation for qv/qcld/Nw
 - generalized Hitschfeld-Bordan attenuation correction (iterative)
 - output: ensemble of profiles (Dm, Nw, corrected Z, R)
Note: This is a simplified but runnable version suitable for testing & development.
"""
import numpy as np
from .utils import dbz_to_linear, linear_to_dbz
from .scattering_tables import build_simple_k_table, lookup_k_from_Z

ENSEMBLE_SIZE = 40
MU = 2.0
Q = 0.2 * np.log(10.0)

def generate_env_ensemble(base_qv, base_qcld, n=ENSEMBLE_SIZE):
    nb = base_qv.size
    qv_ens = np.zeros((n, nb))
    qcld_ens = np.zeros((n, nb))
    for i in range(n):
        qv_ens[i] = base_qv * np.exp(0.25 * np.random.randn(nb))
        qcld_ens[i] = np.maximum(1e-8, base_qcld * np.exp(0.6 * np.random.randn(nb)))
    return qv_ens, qcld_ens

def init_Nw_ensemble(n, nb, convective=False):
    if convective:
        mean = 4.0
    else:
        mean = 3.0
    log10Nw = np.random.randn(n, nb) * 0.4 + mean
    return 10.0 ** log10Nw

def generalized_hb(Zmeas, alt, k_interp, max_iter=60, tol=1e-3):
    """Iteratively correct attenuation using a simple HB-like recursion."""
    # convert to linear if input is dBZ
    if np.nanmax(Zmeas) < 30 and np.nanmin(Zmeas) > -30:
        # probably dBZ -> convert
        Z = dbz_to_linear(Zmeas)
    else:
        Z = Zmeas.copy()
    nb = Z.size
    # initial guess: no attenuation correction
    Zcorr = Z.copy()
    dr = np.abs(np.diff(alt, prepend=alt[0])) + 1e-6
    for it in range(max_iter):
        k_profile = lookup_k_from_Z(Zcorr, k_interp)
        tau = np.cumsum(k_profile * dr)  # one-way integrated extinction (km * 1/km -> unitless)
        Z_new = Z * np.exp(Q * tau)
        rel = np.max(np.abs(Z_new - Zcorr) / (Zcorr + 1e-9))
        Zcorr = Z_new
        if rel < tol:
            break
    # derive Dm from Zcorr using an empirical relation (placeholder)
    Dm = (Zcorr + 1e-6) ** (1.0/6.0)
    return Zcorr, Dm, k_profile

def ku_radar_module_for_footprint(ZKu_dBZ, alt, base_qv=None, base_qcld=None, convective=False):
    # prepare base env if not provided
    nb = ZKu_dBZ.size
    if base_qv is None:
        base_qv = np.ones(nb) * 0.01
    if base_qcld is None:
        base_qcld = np.ones(nb) * 1e-4
    # build k table
    k_interp = build_simple_k_table()
    # generate ensembles
    qv_ens, qcld_ens = generate_env_ensemble(base_qv, base_qcld)
    Nw_ens = init_Nw_ensemble(ENSEMBLE_SIZE, nb, convective)
    # loop ensemble members and perform HB inversion
    ensemble_results = []
    Z_lin = dbz_to_linear(ZKu_dBZ)
    for i in range(ENSEMBLE_SIZE):
        # apply slight perturbs to measured Z to simulate sampling
        Z_pert = Z_lin * np.exp(0.1 * np.random.randn(nb))
        Zcorr, Dm, kprof = generalized_hb(Z_pert, alt, k_interp)
        # compute rain rate R from Dm and Nw (empirical): R ~ a * Nw * Dm^4 (placeholder)
        R = 1e-3 * Nw_ens[i] * Dm ** 4
        ensemble_results.append({
            'Dm': Dm,
            'Nw': Nw_ens[i],
            'qv': qv_ens[i],
            'qcld': qcld_ens[i],
            'Zcorr': Zcorr,
            'kprof': kprof,
            'R': R
        })
    return ensemble_results

if __name__ == '__main__':
    # quick smoke test using synthetic profile
    nb = 176
    ZKu = np.linspace(-10, 40, nb)  # dBZ
    alt = np.arange(nb) * 0.25  # km
    ens = ku_radar_module_for_footprint(ZKu, alt)
    print('Ensemble members produced:', len(ens), 'bins:', len(ens[0]['Dm']))
