import os
import numpy as np, pandas as pd

RAW, PROC = 'data/raw', 'data/processed'

def _read_h5(path):
    try:
        return pd.read_hdf(path)
    except Exception as e:
        print('pandas could not read h5 (', type(e).__name__, '), using h5py fallback')
        import h5py
        with h5py.File(path, 'r') as f:
            g = f[list(f.keys())[0]]
            vals = g['block0_values'][:]
            cols = [str(c.decode() if isinstance(c, bytes) else c) for c in g['axis0'][:]]
            index = pd.to_datetime(g['axis1'][:])
        if vals.shape[0] == len(cols) and vals.shape[0] != len(index):
            vals = vals.T
        return pd.DataFrame(vals, index=index, columns=cols)

def load_raw():
    df = _read_h5(os.path.join(RAW, 'metr-la.h5'))
    df.columns = df.columns.astype(str)
    ids = [s.strip() for s in open(os.path.join(RAW, 'graph_sensor_ids.txt')).read().strip().split(',')]
    df = df[ids]
    d = pd.read_csv(os.path.join(RAW, 'distances_la_2012.csv'))
    idx = {s: i for i, s in enumerate(ids)}
    N = len(ids)
    dist = np.full((N, N), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0)
    for a, b, c in zip(d['from'].astype(str), d['to'].astype(str), d['cost']):
        if a in idx and b in idx:
            dist[idx[a], idx[b]] = c
    std = dist[~np.isinf(dist)].std()
    A = np.exp(-np.square(dist / std))
    A[A < 0.1] = 0                      # directed Gaussian-kernel adjacency (DCRNN recipe)
    loc = pd.read_csv(os.path.join(RAW, 'graph_sensor_locations.csv'))
    return df, A.astype(np.float32), loc

def clean_speed(speed, max_gap=12):
    """zeros = sensor dropouts -> NaN; fill gaps up to 1 hour by interpolation, keep longer gaps as NaN"""
    s = speed.astype(np.float32).copy()
    s[s == 0] = np.nan
    out = s.copy()
    for j in range(s.shape[1]):
        col = pd.Series(s[:, j])
        isn = col.isna().values
        run = pd.Series(isn).groupby((~isn).cumsum()).transform('sum').values
        filled = col.interpolate(limit_area='inside').values
        keep = isn & (run <= max_gap)
        out[keep, j] = filled[keep]
    return out

def time_features(index):
    ts = pd.DatetimeIndex(index)
    tod = ((ts.hour * 60 + ts.minute) / 1440).values.astype(np.float32)
    dow = ts.dayofweek.values.astype(np.int64)
    hol = ts.normalize().isin(pd.to_datetime(['2012-05-28'])).astype(np.float32)  # Memorial Day
    return ts, tod, dow, hol

if __name__ == '__main__':
    df, A, loc = load_raw()
    speed = df.values.astype(np.float32)
    print('speed matrix', speed.shape, 'from', df.index[0], 'to', df.index[-1])
    print('min', float(speed.min()), 'max', float(speed.max()), 'mean', float(speed.mean()))
    print('share of zeros:', round(float((speed == 0).mean()), 4))
    print('adj', A.shape, 'nonzero', int((A > 0).sum()), 'symmetric', bool(np.allclose(A, A.T)))
    print(loc.head(3)); print('locations rows', len(loc))
    clean = clean_speed(speed)
    print('missing after cleaning:', round(float(np.isnan(clean).mean()), 4))
    ts, tod, dow, hol = time_features(df.index)
    os.makedirs(PROC, exist_ok=True)
    np.savez_compressed(f'{PROC}/clean.npz', speed=clean, raw_missing=(speed == 0), tod=tod, dow=dow, hol=hol,
                        timestamps=ts.strftime('%Y-%m-%d %H:%M').values.astype('U16'), adj=A)
    loc.to_csv(f'{PROC}/sensors_raw.csv', index=False)
    print('saved', PROC)