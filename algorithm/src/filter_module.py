"""Ensemble Filter Module (simplified EnKF-style update)
 - state vector contains log10(Nw) profile + selected environment scalars + TBsim entries
 - performs ensemble covariance computations and applies update formula
 - returns updated ensemble_results list with modified profiles where applicable
"""
import numpy as np

def form_state_vector(member, nbins, nch_tb):
    # Build a compact state vector for filtering:
    # [log10(Nw  (nbins)), qv (nbins), qcld(nbins), TBsim(nch)]
    sv = np.concatenate([np.log10(np.maximum(member['Nw'], 1e-12)),
                         member['qv'],
                         member['qcld'],
                         member['TBsim']])
    return sv

def unpack_state_vector(sv, nbins, nch):
    Nw = 10.0 ** sv[:nbins]
    qv = sv[nbins:2*nbins]
    qcld = sv[2*nbins:3*nbins]
    TBsim = sv[3*nbins:3*nbins+nch]
    return {'Nw': Nw, 'qv': qv, 'qcld': qcld, 'TBsim': TBsim}

def ensemble_filter_update(ensemble_members, forward_outputs, obs_dict, R_diag=None):
    # ensemble_members: list of member dicts (from ku module)
    # forward_outputs: list of dicts {'Zka','PIA','TBsim'}
    # obs_dict: {'Zka_obs': array, 'PIA_obs': scalar, 'TB_obs': array}
    # R_diag: observation error variances (vector) matching observation vector length
    N = len(ensemble_members)
    nb = ensemble_members[0]['Nw'].size
    nch = forward_outputs[0]['TBsim'].size
    # form state matrix X (nstate x N)
    state_size = 3*nb + nch
    X = np.zeros((state_size, N))
    Ysim = np.zeros((nch + 2, N))  # we include ZKa aggregated stat? keep TB + PIA + mean Zka
    for i in range(N):
        member = ensemble_members[i].copy()
        member['TBsim'] = forward_outputs[i]['TBsim']
        # form state vector and simulated obs vector
        sv = form_state_vector(member, nb, nch)
        X[:, i] = sv
        # build simulated obs: TBsim (nch), PIA (1), mean ZKa (1)
        Ysim[:, i] = np.concatenate([forward_outputs[i]['TBsim'], [forward_outputs[i]['PIA']], [np.nanmean(forward_outputs[i]['Zka'])]])
    # observation vector Yobs (match order above)
    Yobs = np.concatenate([obs_dict['TB_obs'], [obs_dict['PIA_obs']], [np.nanmean(obs_dict['Zka_obs'])]])
    # compute ensemble means
    X_mean = np.mean(X, axis=1, keepdims=True)
    Y_mean = np.mean(Ysim, axis=1, keepdims=True)
    # perturbations
    Xp = X - X_mean
    Yp = Ysim - Y_mean
    # covariance matrices (sample covariances)
    PHT = (Xp.dot(Yp.T)) / (N - 1)  # state x obs
    HPHT = (Yp.dot(Yp.T)) / (N - 1)
    # observation error matrix R
    if R_diag is None:
        R = np.eye(HPHT.shape[0]) * 1.0  # generic
    else:
        R = np.diag(R_diag)
    # Kalman gain-like operator
    inv = np.linalg.inv(HPHT + R)
    K = PHT.dot(inv)
    # update ensemble members (apply to each member)
    updates = K.dot((Yobs.reshape(-1,1) - Ysim))
    X_updated = X + updates
    # unpack updated state into ensemble_members
    updated_members = []
    for i in range(N):
        sv_up = X_updated[:, i]
        new = unpack_state_vector(sv_up, nb, nch)
        # copy other fields from original
        orig = ensemble_members[i].copy()
        orig['Nw'] = new['Nw']
        orig['qv'] = new['qv']
        orig['qcld'] = new['qcld']
        orig['TBsim'] = new['TBsim']
        # recompute derived fields (simple)
        orig['R'] = 1e-3 * orig['Nw'] * (orig.get('Dm', 0.1) ** 4)
        updated_members.append(orig)
    return updated_members

if __name__ == '__main__':
    # smoke test with synthetic ensembles
    from .ku_module import ku_radar_module_for_footprint
    nb = 20
    ZKu = np.linspace(-5, 30, nb)
    alt = np.arange(nb) * 0.25
    ens = ku_radar_module_for_footprint(ZKu, alt)
    # create fake forward outputs and obs
    fwd = [{'Zka': e['Zcorr']*0.9, 'PIA':0.5, 'TBsim': np.ones(9)*250} for e in ens]
    obs = {'Zka_obs': fwd[0]['Zka'], 'PIA_obs':0.6, 'TB_obs': np.ones(9)*251}
    up = ensemble_filter_update(ens, fwd, obs)
    print('Updated ensemble size:', len(up))
