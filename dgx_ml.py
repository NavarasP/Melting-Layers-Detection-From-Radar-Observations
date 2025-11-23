import os
import glob
import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------
# File discovery
# ----------------------

def validate_hdf5_file(filepath):
    """Check if HDF5 file can be opened without errors."""
    try:
        with h5py.File(filepath, 'r') as hf:
            # Try to access a common field to verify integrity
            _ = list(hf.keys())
        return True
    except (OSError, IOError, RuntimeError) as e:
        print(f"WARNING: Skipping corrupted file {os.path.basename(filepath)}: {e}")
        return False

def list_hdf5_files(data_dir, pattern="**/*.HDF5", validate=True):
    paths = glob.glob(os.path.join(data_dir, pattern), recursive=True)
    paths = [p for p in paths if os.path.isfile(p)]
    
    if validate:
        print(f"Validating {len(paths)} HDF5 files...")
        valid_paths = []
        for p in paths:
            if validate_hdf5_file(p):
                valid_paths.append(p)
        print(f"Valid files: {len(valid_paths)}/{len(paths)}")
        return sorted(valid_paths)
    
    return sorted(paths)

# ----------------------
# Safe dataset loader
# ----------------------

def safe_load_dataset(hf, path):
    try:
        ds = hf[path]
        arr = np.array(ds)
        return arr
    except Exception:
        return None

# ----------------------
# Reflectivity readers and helpers
# ----------------------

def _apply_attrs(arr, ds):
    try:
        scale = ds.attrs.get('scale_factor', 1.0)
        offset = ds.attrs.get('add_offset', 0.0)
        fill = ds.attrs.get('_FillValue', None)
    except Exception:
        scale, offset, fill = 1.0, 0.0, None
    arr = arr.astype(np.float32) * (scale if scale is not None else 1.0) + (offset if offset is not None else 0.0)
    if fill is not None:
        arr = np.where(arr == fill, np.nan, arr)
    return arr

def read_reflectivity_band(hf, group='FS', ds_path='/SLV/zFactorFinal', fallback='/PRE/zFactorMeasured'):
    """
    Read reflectivity and return shape (nscan, nray, nbin).
    Handles:
    - Fallback between Final/Measured
    - 4D arrays (select first polarization/frequency)
    - Dimension order fixes using Latitude for orientation (nscan, nray) vs (nscan, nbin)
    """
    paths = [f"/{group}{ds_path}", f"/{group}{fallback}"]
    last_err = None
    for p in paths:
        try:
            ds = hf[p]
            arr = _apply_attrs(np.array(ds), ds)
            # Reduce frequency/polarization dim if present
            if arr.ndim == 4:
                # Expected (nscan, nray, nbin, nfreq)
                arr = arr[..., 0]
            # Align to (nscan, nray, nbin) using Latitude if available
            if arr.ndim == 3:
                try:
                    lat_path = f"/{group}/Latitude"
                    lat = safe_load_dataset(hf, lat_path)
                    if lat is not None and lat.ndim == 2 and lat.shape[0] == arr.shape[0]:
                        # If second dim doesn't match nray but third does, transpose
                        if arr.shape[1] != lat.shape[1] and arr.shape[2] == lat.shape[1]:
                            arr = np.transpose(arr, (0, 2, 1))
                except Exception:
                    pass
            if arr.ndim != 3:
                raise ValueError(f"Unexpected reflectivity shape {arr.shape} for dataset {p}")
            return arr
        except Exception as e:
            last_err = e
            continue
    raise KeyError(f"Could not find reflectivity dataset in {paths}: last error={last_err}")

# ----------------------
# Height helpers
# ----------------------

def get_height_axis(nbin, top_km=21.5, bottom_km=0.0):
    return np.linspace(bottom_km, top_km, num=nbin, dtype=np.float32)


def get_height_axis_from_hdf5(hf, scan_idx, path='/FS/PRE/height', to_agl=True):
    try:
        h = hf[path]
        # Expect shape (nscan, nbin) or (nscan, nray, nbin). We take an average over rays if needed
        arr = np.array(h)
        if arr.ndim == 3:
            arr = arr[scan_idx].mean(axis=0)
        elif arr.ndim == 2:
            arr = arr[scan_idx]
        else:
            return None
        arr = arr.astype(np.float32)
        if to_agl:
            arr = np.maximum(0.0, arr)
        return arr
    except Exception:
        return None

# ----------------------
# Mask builders
# ----------------------

def build_mask_from_bins(topBB, bottomBB, flagBB, qualityBB, nbin):
    if topBB is None or bottomBB is None or flagBB is None:
        return None
    nscan, nray = flagBB.shape
    mask = np.zeros((nscan, nray, nbin), dtype=np.uint8)
    for s in range(nscan):
        for r in range(nray):
            if flagBB[s, r] == 1:
                t = int(topBB[s, r])
                b = int(bottomBB[s, r])
                if 0 <= t < nbin and 0 <= b < nbin and b >= t:
                    mask[s, r, t:b+1] = 1
    return mask


def build_improved_mask_from_csf(hf, nbin, quality_threshold=50, use_dfr_ml=True):
    flagBB = safe_load_dataset(hf, '/FS/CSF/flagBB')
    topBB = safe_load_dataset(hf, '/FS/CSF/binBBTop')
    bottomBB = safe_load_dataset(hf, '/FS/CSF/binBBBottom')
    qBB = safe_load_dataset(hf, '/FS/CSF/qualityBB')
    if qBB is None:
        qBB = safe_load_dataset(hf, '/FS/CSF/flagMLquality')
    base = build_mask_from_bins(topBB, bottomBB, flagBB, qBB, nbin)
    if base is None:
        return None
    if qBB is not None and quality_threshold is not None:
        keep = (qBB >= quality_threshold).astype(np.uint8)
        base = base * keep[:, :, None]
    return base


def build_dataset_full_mask(hf, nbin, use_dfr_ml=True, quality_threshold=None):
    # Prefer DFR-based ML bounds when available
    top = safe_load_dataset(hf, '/FS/CSF/binDFRmMLTop')
    bot = safe_load_dataset(hf, '/FS/CSF/binDFRmMLBottom')
    if top is None or bot is None:
        return build_improved_mask_from_csf(hf, nbin, quality_threshold=quality_threshold, use_dfr_ml=use_dfr_ml)
    flag = np.ones_like(top, dtype=np.uint8)
    qual = np.ones_like(top, dtype=np.uint8) * 100
    return build_mask_from_bins(top, bot, flag, qual, nbin)

# ----------------------
# Preprocessing
# ----------------------

def normalize_and_pad_scan(z2d, nbin_target=88, min_val=-30.0, max_val=60.0):
    # z2d: (nbin, nray)
    nbin_src, nray = z2d.shape
    z = np.nan_to_num(z2d, nan=min_val)
    z = (z - min_val) / (max_val - min_val)
    z = np.clip(z, 0.0, 1.0)
    if nbin_target == nbin_src:
        return z
    if nbin_target > nbin_src:
        pad_total = nbin_target - nbin_src
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        return np.pad(z, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0.0)
    else:
        start = (nbin_src - nbin_target) // 2
        return z[start:start+nbin_target, :]


def resample_rays_2d(z2d, target_nray):
    nbin_src, nray_src = z2d.shape
    if nray_src == target_nray:
        return z2d
    x_old = np.linspace(0, 1, nray_src)
    x_new = np.linspace(0, 1, target_nray)
    out = np.empty((nbin_src, target_nray), dtype=np.float32)
    for i in range(nbin_src):
        out[i] = np.interp(x_new, x_old, z2d[i].astype(np.float32))
    return out

# ----------------------
# Dataset builder
# ----------------------

def build_arrays_for_mask_mode(file_list, mask_mode='baseline', use_z_final=True, use_dfr_ml=True,
                               include_latlon=False, nbin_target=88, min_val=-30.0, max_val=60.0,
                               max_scans=None):
    X_list, Y_list = [], []
    count_scans = 0
    skipped_files = 0
    
    for fp in file_list:
        try:
            with h5py.File(fp, 'r') as hf:
                z_ku = read_reflectivity_band(hf, group='FS', ds_path='/SLV/zFactorFinal' if use_z_final else '/PRE/zFactorMeasured')
                try:
                    z_ka = read_reflectivity_band(hf, group='HS', ds_path='/SLV/zFactorFinal' if use_z_final else '/PRE/zFactorMeasured')
                except Exception:
                    z_ka = np.full_like(z_ku, np.nan)
                nscan, nray, nbin = z_ku.shape
                if mask_mode == 'baseline':
                    flagBB = safe_load_dataset(hf, '/FS/CSF/flagBB')
                    topBB = safe_load_dataset(hf, '/FS/CSF/binBBTop')
                    bottomBB = safe_load_dataset(hf, '/FS/CSF/binBBBottom')
                    qualityBB = safe_load_dataset(hf, '/FS/CSF/qualityBB')
                    if qualityBB is None:
                        qualityBB = safe_load_dataset(hf, '/FS/CSF/flagMLquality')
                    masks = build_mask_from_bins(topBB, bottomBB, flagBB, qualityBB, nbin)
                elif mask_mode == 'conservative':
                    masks = build_improved_mask_from_csf(hf, nbin, quality_threshold=50, use_dfr_ml=use_dfr_ml)
                else:
                    masks = build_dataset_full_mask(hf, nbin, use_dfr_ml=use_dfr_ml, quality_threshold=None)

                lat = safe_load_dataset(hf, '/FS/Latitude')
                lon = safe_load_dataset(hf, '/FS/Longitude')

                for s in range(nscan):
                    ku = np.transpose(z_ku[s], (1, 0))  # (nbin, nray)
                    ka = np.transpose(z_ka[s], (1, 0))
                    # Resample Ka rays if mismatch
                    if ka.shape[1] != ku.shape[1]:
                        ka = resample_rays_2d(ka, ku.shape[1])
                    ku = np.nan_to_num(ku, nan=min_val)
                    ka = np.nan_to_num(ka, nan=min_val)
                    ku_n = normalize_and_pad_scan(ku, nbin_target=nbin_target, min_val=min_val, max_val=max_val)
                    ka_n = normalize_and_pad_scan(ka, nbin_target=nbin_target, min_val=min_val, max_val=max_val)

                    if include_latlon:
                        lat_s = lat[s] if lat is not None else np.zeros(ku.shape[1])
                        lon_s = lon[s] if lon is not None else np.zeros(ku.shape[1])
                        if lat_s.shape[0] != ku.shape[1]:
                            x_old = np.linspace(0, 1, num=lat_s.shape[0])
                            x_new = np.linspace(0, 1, num=ku.shape[1])
                            lat_s = np.interp(x_new, x_old, lat_s)
                            lon_s = np.interp(x_new, x_old, lon_s)
                        lat_img = np.tile(((lat_s + 90.0) / 180.0)[np.newaxis, :], (nbin_target, 1))
                        lon_img = np.tile(((lon_s + 180.0) / 360.0)[np.newaxis, :], (nbin_target, 1))
                        x_img = np.stack([ku_n, ka_n, lat_img, lon_img], axis=-1)
                    else:
                        x_img = np.stack([ku_n, ka_n], axis=-1)

                    m = masks[s].T  # (nbin, nray)
                    m = normalize_and_pad_scan(m, nbin_target=nbin_target, min_val=0.0, max_val=1.0)
                    X_list.append(x_img.astype(np.float32))
                    Y_list.append(m.astype(np.float32))
                    count_scans += 1
                    if max_scans is not None and count_scans >= max_scans:
                        break
                if max_scans is not None and count_scans >= max_scans:
                    break
        except (OSError, IOError, RuntimeError) as e:
            print(f"WARNING: Skipping corrupted file {os.path.basename(fp)}: {e}")
            skipped_files += 1
            continue
    
    if skipped_files > 0:
        print(f"Skipped {skipped_files} corrupted file(s)")
    
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)[..., np.newaxis]
    return X, Y

# ----------------------
# Model
# ----------------------

def build_unet_2d(input_channels=2, nbin_target=88, base_filters=16):
    inp = layers.Input(shape=(nbin_target, None, input_channels))
    # Encoder (down only in vertical axis)
    c1 = layers.Conv2D(base_filters, 3, padding='same', activation='relu')(inp)
    c1 = layers.Conv2D(base_filters, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPool2D(pool_size=(2,1))(c1)

    c2 = layers.Conv2D(base_filters*2, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(base_filters*2, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPool2D(pool_size=(2,1))(c2)

    c3 = layers.Conv2D(base_filters*4, 3, padding='same', activation='relu')(p2)
    c3 = layers.Conv2D(base_filters*4, 3, padding='same', activation='relu')(c3)
    p3 = layers.MaxPool2D(pool_size=(2,1))(c3)

    bn = layers.Conv2D(base_filters*8, 3, padding='same', activation='relu')(p3)
    bn = layers.Conv2D(base_filters*8, 3, padding='same', activation='relu')(bn)

    u3 = layers.UpSampling2D(size=(2,1))(bn)
    u3 = layers.Concatenate()([u3, c3])
    c4 = layers.Conv2D(base_filters*4, 3, padding='same', activation='relu')(u3)
    c4 = layers.Conv2D(base_filters*4, 3, padding='same', activation='relu')(c4)

    u2 = layers.UpSampling2D(size=(2,1))(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = layers.Conv2D(base_filters*2, 3, padding='same', activation='relu')(u2)
    c5 = layers.Conv2D(base_filters*2, 3, padding='same', activation='relu')(c5)

    u1 = layers.UpSampling2D(size=(2,1))(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = layers.Conv2D(base_filters, 3, padding='same', activation='relu')(u1)
    c6 = layers.Conv2D(base_filters, 3, padding='same', activation='relu')(c6)

    out = layers.Conv2D(1, 1, activation='sigmoid')(c6)
    model = keras.Model(inputs=inp, outputs=out)
    return model

# ----------------------
# Loss and metrics
# ----------------------

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        ce = - (alpha * y_true * tf.math.log(y_pred) + (1-alpha) * (1-y_true) * tf.math.log(1-y_pred))
        fl = tf.pow(1-y_pred, gamma) * ce
        return tf.reduce_mean(fl)
    return loss


def calc_iou_dice(y_true, y_prob, thresh=0.45):
    yt = (y_true >= 0.5).astype(np.uint8)
    yp = (y_prob >= thresh).astype(np.uint8)
    inter = np.logical_and(yt==1, yp==1).sum()
    union = np.logical_or(yt==1, yp==1).sum()
    iou = inter / max(1, union)
    s_true = yt.sum(); s_pred = yp.sum()
    dice = 2*inter / max(1, s_true + s_pred)
    return float(iou), float(dice)


def evaluate_model_on_files(model, files, mask_mode='baseline', thresh=0.45, **build_kwargs):
    X, Y = build_arrays_for_mask_mode(files, mask_mode=mask_mode, **build_kwargs)
    Yp = model.predict(X, verbose=0)
    if Yp.ndim == 4 and Yp.shape[-1] == 1:
        Yp = Yp[..., 0]
    if Y.ndim == 4 and Y.shape[-1] == 1:
        Yt = Y[..., 0]
    else:
        Yt = Y
    ious, dices = [], []
    for i in range(Yt.shape[0]):
        iou, dice = calc_iou_dice(Yt[i], Yp[i], thresh=thresh)
        ious.append(iou); dices.append(dice)
    return {
        'n_samples': int(Yt.shape[0]),
        'mean_iou': float(np.mean(ious)),
        'mean_dice': float(np.mean(dices)),
        'mean_pred_max': float(np.mean(np.max(Yp, axis=(1,2))))
    }
