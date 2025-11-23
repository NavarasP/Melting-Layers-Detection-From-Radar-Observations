"""I/O utilities for GPM Combined implementation (simplified).
Functions:
 - read_dpr_ku(filename, scan_index, ray_index)
 - read_gmi_tb(filename, scan_index, pixel_index)
 - read_env(filename, scan_index, ray_index)
This module uses common paths found in GPM DPR/GMI L1/L2 HDF5 files; adapt dataset keys to your files.
"""
import h5py
import numpy as np

def safe_read(h5, path):
    if path in h5:
        return h5[path][:]
    # try to find similar keys
    for k in h5.keys():
        if path.split('/')[-1] in k:
            return h5[k][:]
    raise KeyError(f"Dataset {path} not found in file. Available top-level keys: {list(h5.keys())}")

def read_dpr_ku(filename, scan_index=0, ray_index=0):
    with h5py.File(filename, 'r') as f:
        # common DPR path used in many files; adapt if needed
        try:
            zfactor = f['/FS/SLV/zFactorFinal'][:]  # actual shape: (nscan, nray, nbin, nfreq)
            # Correct indexing: [scan_index, ray_index, :, freq_index]
            ZKu_dBZ = zfactor[scan_index, ray_index, :, 0].astype(float)
        except Exception:
            # fallback: try a different path
            if '/FS/SLV/zFactor' in f:
                ZKu_dBZ = f['/FS/SLV/zFactor'][scan_index, ray_index, :].astype(float)
            else:
                raise
        # heights/altitude
        if '/FS/SLV/heightAboveGeoid' in f:
            alt_data = f['/FS/SLV/heightAboveGeoid'][:]
            # Check shape and index correctly
            if alt_data.ndim == 3:  # (nscan, nray, nbin)
                alt = alt_data[scan_index, ray_index, :].astype(float)
            else:
                alt = np.arange(ZKu_dBZ.size) * 0.125  # 125m per bin
        else:
            # create synthetic alt based on actual number of bins
            alt = np.arange(ZKu_dBZ.size) * 0.125  # km spacing (125m per bin)
        # bright band flag
            # bright band flag - handle multiple possible shapes
        if '/FS/CSF/flagBB' in f:
            flag = f['/FS/CSF/flagBB']

            # case 1: 3D (nbin, nray, nscan)
            if flag.ndim == 3:
                flagBB = flag[:, ray_index, scan_index].astype(int)

            # case 2: 2D (nscan, nray)
            elif flag.ndim == 2:
                # your file uses this:
                # shape = (7937, 49) = (nscan, nray)
                # correct indexing:
                flagBB = flag[scan_index, ray_index] * np.ones_like(ZKu_dBZ, dtype=int)

            # case 3: 1D (nray,)
            elif flag.ndim == 1:
                flagBB = flag[ray_index] * np.ones_like(ZKu_dBZ, dtype=int)

            else:
                flagBB = np.zeros_like(ZKu_dBZ, dtype=int)

        else:
            flagBB = np.zeros_like(ZKu_dBZ, dtype=int)

    return ZKu_dBZ, alt, flagBB

def read_gmi_tb(filename, scan_index=0, pixel_index=0):
    with h5py.File(filename, 'r') as f:
        # typical GMI L1C path: '/S1/Tb' or '/S1/Tb' variants
        # we'll try a few likely names and return a vector of channels
        candidates = ['/S1/Tb', '/S1/Tbs', '/S1/brightness_temperature', '/S1/TbS1']
        for c in candidates:
            if c in f:
                tbs = f[c][scan_index, :, pixel_index]  # (channels,)
                return tbs.astype(float)
        # fallback: any dataset containing 'Tb' in name
        for k in f.keys():
            if 'Tb' in k or 'tb' in k:
                arr = f[k]
                try:
                    return arr[scan_index, :, pixel_index].astype(float)
                except Exception:
                    continue
    raise KeyError('GMI Tb dataset not found.')

def read_env(filename, scan_index=0, ray_index=0):
    # Environmental fields (temperature, pressure, vapor) ideally come from ENV product or analysis
    with h5py.File(filename, 'r') as f:
        out = {}
        out['temperature'] = f['/ENV/airTemperature'][:, ray_index, scan_index].astype(float) if '/ENV/airTemperature' in f else np.linspace(290, 250, 176)
        out['pressure'] = f['/ENV/airPressure'][:, ray_index, scan_index].astype(float) if '/ENV/airPressure' in f else np.linspace(101300, 20000, 176)
        out['qv'] = f['/ENV/waterVapor'][:, ray_index, scan_index].astype(float) if '/ENV/waterVapor' in f else np.ones(176)*0.01
    return out
