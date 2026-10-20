import numpy as np

HIST, HOR = 12, 12


def prepare(path='data/processed/clean.npz'):
    # 🌟 FIX: Added allow_pickle=True to handle the object array (timestamps)
    z = np.load(path, allow_pickle=True)
    speed = z['speed'].astype(np.float32)                    # [T,N], NaN = missing
    T, N = speed.shape
    n_tr, n_va = int(T * 0.7), int(T * 0.1)
    bounds = {'train': (0, n_tr), 'val': (n_tr, n_tr + n_va), 'test': (n_tr + n_va, T)}
    tr = speed[:n_tr]
    mean, std = float(np.nanmean(tr)), float(np.nanstd(tr))   # training data only: no leakage
    free_flow = np.nanpercentile(tr, 95, axis=0).astype(np.float32)
    obs = ~np.isnan(speed)
    sp = np.where(obs, (speed - mean) / std, 0.0).astype(np.float32)
    ang = 2 * np.pi * z['tod']
    tfeat = np.stack([np.sin(ang), np.cos(ang), z['dow'] / 6.0, z['hol']], axis=1).astype(np.float32)
    feats = np.concatenate([sp[:, :, None], obs[:, :, None].astype(np.float32),
                            np.repeat(tfeat[:, None, :], N, axis=1)], axis=2)    # [T,N,6]
    A = z['adj'].astype(np.float32)

    def rw(M):
        d = M.sum(1, keepdims=True)
        d[d == 0] = 1
        return M / d
    supports = np.stack([rw(A), rw(A.T)]).astype(np.float32)  # forward / backward diffusion
    t_idx = {k: np.arange(lo + HIST - 1, hi - HOR) for k, (lo, hi) in bounds.items()}
    
    # 🌟 FIX: Explicitly cast timestamps to avoid unpickling layout bugs down the line
    return dict(speed=speed, feats=feats, supports=supports, adj=A, mean=mean, std=std,
                free_flow=free_flow, bounds=bounds, t_idx=t_idx,
                bin=np.rint(z['tod'] * 288).astype(int) % 288, dow=z['dow'].astype(int),
                timestamps=z['timestamps'].astype(object))


def get_batch(P, split, ids):
    """x [B,12,N,6] model inputs, y [B,12,N] true speeds in mph (NaN = missing)"""
    t = P['t_idx'][split][ids]
    xi = t[:, None] + np.arange(-HIST + 1, 1)[None, :]
    yi = t[:, None] + np.arange(1, HOR + 1)[None, :]
    return P['feats'][xi], P['speed'][yi]


if __name__ == '__main__':
    P = prepare()
    print('feats', P['feats'].shape, 'supports', P['supports'].shape)
    print('mean/std', round(P['mean'], 2), round(P['std'], 2))
    print('free-flow speed range', P['free_flow'].min().round(1), P['free_flow'].max().round(1))
    for k, v in P['t_idx'].items():
        print(k, 'windows:', len(v))
    x, y = get_batch(P, 'train', np.arange(4))
    print('batch x', x.shape, 'y', y.shape)
