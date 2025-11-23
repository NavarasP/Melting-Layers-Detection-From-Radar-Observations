"""Forward Model Module (simplified)
 - simulate ZKa by scaling ZKu according to microphysics
 - compute PIA as integral of k
 - simulate brightness temperatures via simple two-stream approx.
This is a computationally-light placeholder for the ATBD forward model.
"""
import numpy as np
from .utils import linear_to_dbz
from .scattering_tables import build_emissivity_lookup, build_simple_k_table, lookup_k_from_Z

def simulate_Zka_from_Zku(Zku_linear, scale_factor=0.9):
    # simple mapping: Zka ~ alpha * Zku (Ka more attenuated for larger drop sizes)
    return Zku_linear * scale_factor

def compute_pia(k_profile, alt):
    # PIA (two-way) approximate: 2 * integral k ds (units consistent: k in 1/km, alt in km)
    dr = np.abs(np.diff(alt, prepend=alt[0])) + 1e-6
    tau = np.cumsum(k_profile * dr)
    pia_two_way = 2.0 * tau[-1]
    return pia_two_way

def simulate_tb_from_profiles(ensemble_member, surface_type='ocean', Ts=300.0, wind=5.0, freqs=range(9)):
    # ensemble_member: dict with keys Dm, Nw, qv, qcld, Zcorr, kprof
    # returns an array of TBs for provided freqs (length of freqs)
    emissivity_fn = build_emissivity_lookup()
    tb_list = []
    for fi in freqs:
        eps = emissivity_fn(surface_type, fi, wind, Ts)
        # upwelling TB simplified: TB = (1 - eps) * Ts * e^{-tau} + eps * Ts_surface (very rough)
        tau = np.mean(ensemble_member['kprof']) * 0.1  # placeholder for channel-dependent tau
        tb = (1 - eps) * Ts * np.exp(-tau) + eps * Ts * (1 - np.exp(-tau))
        tb_list.append(tb)
    return np.array(tb_list)

def forward_model_for_ensemble(ensemble_results, alt, surface_type='ocean', freqs=range(9)):
    outputs = []
    for member in ensemble_results:
        Zku = member['Zcorr']
        Zka = simulate_Zka_from_Zku(Zku, scale_factor=0.85)
        # compute k (use existing kprof)
        kprof = member.get('kprof')
        pia = compute_pia(kprof, alt)
        tb = simulate_tb_from_profiles(member, surface_type=surface_type, freqs=freqs)
        outputs.append({'Zka': Zka, 'PIA': pia, 'TBsim': tb})
    return outputs

if __name__ == '__main__':
    # smoke test
    from .ku_module import ku_radar_module_for_footprint
    nb = 176
    ZKu = np.linspace(-10, 40, nb)
    alt = np.arange(nb) * 0.25
    ens = ku_radar_module_for_footprint(ZKu, alt)
    fwd = forward_model_for_ensemble(ens, alt)
    print('Forward model results for ensemble size:', len(fwd))
