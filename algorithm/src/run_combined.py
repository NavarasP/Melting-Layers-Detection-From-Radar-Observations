"""Run combined pipeline on a single DPR/GMI footprint (demo)
Usage: adapt DPR/GMI filenames and indexes in the main() below.
"""
import numpy as np
from .io import read_dpr_ku, read_gmi_tb, read_env
from .ku_module import ku_radar_module_for_footprint
from .forward_model import forward_model_for_ensemble
from .radiance_enhancement import build_enhancement_filter, apply_enhancement
from .filter_module import ensemble_filter_update
from .scattering_tables import build_simple_k_table

def run_one_footprint(dpr_file, gmi_file, env_file, scan_idx=0, ray_idx=10, pixel_idx=10):
    print('Reading DPR...')
    ZKu_dBZ, alt, flagBB = read_dpr_ku(dpr_file, scan_idx, ray_idx)
    env = read_env(env_file, scan_idx, ray_idx)
    print('Running Ku Radar Module (ensemble)...')
    ensemble = ku_radar_module_for_footprint(ZKu_dBZ, alt, base_qv=env['qv'], base_qcld=np.ones_like(env['qv'])*1e-4)
    print('Forward model...')
    fwd = forward_model_for_ensemble(ensemble, alt)
    print('Read observed GMI TB (if available)')
    try:
        TB_obs = read_gmi_tb(gmi_file, scan_idx, pixel_idx)
    except Exception as e:
        print('GMI read failed:', e)
        TB_obs = np.ones(9)*250.0
    # Build a toy enhancement filter using simulated TBs: use first half ensemble as training
    TB_sim_high = np.vstack([m['TBsim'] for m in fwd[:20]])
    # fake 'lowres' as neighborhood mean
    TB_sim_low = TB_sim_high.mean(axis=1, keepdims=True) * np.ones((TB_sim_high.shape[0], TB_sim_high.shape[1]))
    A, b = build_enhancement_filter(TB_sim_high, TB_sim_low)
    TB_enh = apply_enhancement(TB_obs, A, b)
    print('Apply ensemble filter using observed TB and PIA/ZKa...')
    obs_dict = {'TB_obs': TB_enh, 'PIA_obs': np.mean([x['PIA'] for x in fwd]), 'Zka_obs': fwd[0]['Zka']}
    updated = ensemble_filter_update(ensemble, fwd, obs_dict)
    print('Pipeline finished; returning updated ensemble.')
    return updated, fwd

if __name__ == '__main__':
    # demo run with synthetic files (placeholders) - user should replace paths
    DPR = 'data/sample_dpr.h5'
    GMI = 'data/sample_gmi.h5'
    ENV = 'data/sample_env.h5'
    # create small synthetic files if they don't exist to allow the demo to run
    import h5py, numpy as np
    if not os.path.exists(DPR):
        with h5py.File(DPR, 'w') as f:
            f.create_dataset('/FS/SLV/zFactorFinal', data=np.zeros((1,176,49,1))+5.0)
            f.create_dataset('/FS/SLV/heightAboveGeoid', data=np.tile(np.arange(176).reshape(176,1,1), (1,49,1))*0.25)
            f.create_dataset('/FS/CSF/flagBB', data=np.zeros((49,1), dtype=int))
    if not os.path.exists(GMI):
        with h5py.File(GMI, 'w') as f:
            f.create_dataset('/S1/Tb', data=np.ones((1,9,221))+250.0)
    if not os.path.exists(ENV):
        with h5py.File(ENV, 'w') as f:
            f.create_dataset('/ENV/airTemperature', data=np.linspace(290,250,176).reshape(176,1,1))
            f.create_dataset('/ENV/airPressure', data=np.linspace(101300,20000,176).reshape(176,1,1))
            f.create_dataset('/ENV/waterVapor', data=np.ones((176,1,1))*0.01)
    updated, fwd = run_one_footprint(DPR, GMI, ENV)
    print('Example updated ensemble member keys:', list(updated[0].keys()))
