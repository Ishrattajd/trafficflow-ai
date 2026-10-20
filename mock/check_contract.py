"""Checks a data folder against the frontend data contract.
Usage: python mock/check_contract.py [folder]
Default folder: app_data/ if it has meta.json, else app_data_mock/."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if len(sys.argv) > 1:
    D = Path(sys.argv[1])
else:
    D = ROOT / 'app_data' if (ROOT / 'app_data' / 'meta.json').exists() else ROOT / 'app_data_mock'
fails, warns = [], []

def ok(m): print('PASS', m)
def fail(m): fails.append(m); print('FAIL', m)
def warn(m): warns.append(m); print('WARN', m)

def done():
    print(f'\nSummary: {len(fails)} FAIL, {len(warns)} WARN')
    sys.exit(1 if fails else 0)

print('Checking', D)
if not (D / 'meta.json').exists():
    fail('meta.json missing'); done()
meta = json.loads((D / 'meta.json').read_text())
need = ['dataset', 'region', 'n_sensors', 'horizon_steps', 'step_minutes', 'n_timesteps',
        'models', 'main_model', 'thresholds', 'free_flow_speed']
miss = [k for k in need if k not in meta]
if miss:
    fail(f'meta.json missing keys: {miss}'); done()
ok('meta.json has all required keys')
N, H, T = meta['n_sensors'], meta['horizon_steps'], meta['n_timesteps']

if meta['main_model'] in meta['models']: ok('main_model is in models')
else: fail('main_model is not in the models list')
th = meta['thresholds']
if 'free' in th and 'slow' in th and th['free'] > th['slow'] > 0: ok('thresholds look valid')
else: fail(f'thresholds invalid: {th}')
ffs = np.array(meta['free_flow_speed'], dtype=float)
if len(ffs) == N and np.all(ffs > 0): ok(f'free_flow_speed has {N} positive values')
else: fail(f'free_flow_speed must have {N} positive values (got {len(ffs)})')

def need_csv(name, cols):
    p = D / name
    if not p.exists():
        fail(f'{name} missing'); return None
    df = pd.read_csv(p)
    if not set(cols).issubset(df.columns):
        fail(f'{name} needs columns {cols}, found {list(df.columns)}'); return None
    return df

s = need_csv('sensors.csv', ['sensor_id', 'latitude', 'longitude'])
if s is not None:
    if len(s) != N: fail(f'sensors.csv has {len(s)} rows, expected {N}')
    elif not (s['sensor_id'].values == np.arange(N)).all(): fail('sensor_id must be 0..N-1 in order')
    elif s[['latitude', 'longitude']].isna().any().any(): fail('sensors.csv has missing coordinates')
    else: ok('sensors.csv is valid')

e = need_csv('edges.csv', ['source', 'target', 'weight'])
if e is not None:
    if e[['source', 'target']].min().min() < 0 or e[['source', 'target']].max().max() >= N:
        fail('edges.csv has sensor ids outside 0..N-1')
    else: ok(f'edges.csv is valid ({len(e)} edges)')

tdf = need_csv('timestamps.csv', ['timestamp'])
if tdf is not None:
    ts = pd.to_datetime(tdf['timestamp'])
    if len(ts) != T: fail(f'timestamps.csv has {len(ts)} rows, expected {T}')
    elif not (ts.diff().dropna() == pd.Timedelta(minutes=meta['step_minutes'])).all():
        fail('timestamps are not evenly spaced by step_minutes')
    else: ok('timestamps.csv is valid')

actual = None
if (D / 'actual.npy').exists():
    actual = np.load(D / 'actual.npy')
    if actual.shape != (T, N): fail(f'actual.npy shape {actual.shape}, expected {(T, N)}'); actual = None
    else:
        if actual.dtype != np.float16: warn(f'actual.npy dtype is {actual.dtype}, contract says float16')
        ok(f'actual.npy shape OK, {np.isnan(actual).mean() * 100:.1f}% NaN')
        mx = float(np.nanmax(actual))
        if mx > 150: warn(f'actual.npy max speed {mx:.0f} looks too high for mph')
else:
    fail('actual.npy missing')

main_pred = None
for m in meta['models']:
    p = D / f'pred_{m}.npy'
    if not p.exists():
        (fail if m == meta['main_model'] else warn)(f'pred_{m}.npy missing (app will skip this model)')
        continue
    a = np.load(p, mmap_mode='r')
    if a.shape != (T, H, N): fail(f'pred_{m}.npy shape {a.shape}, expected {(T, H, N)}'); continue
    if a.dtype != np.float16: warn(f'pred_{m}.npy dtype is {a.dtype}, contract says float16')
    nan_pct = float(np.isnan(np.asarray(a)).mean() * 100)
    ok(f'pred_{m}.npy shape OK' + (f', {nan_pct:.1f}% NaN' if nan_pct else ''))
    if m == meta['main_model']: main_pred = np.asarray(a, dtype=np.float32)

if main_pred is not None and actual is not None:
    a32 = actual.astype(np.float32)
    n = T - 12
    h = 2  # 15 minutes ahead
    res = {off: float(np.nanmean(np.abs(main_pred[:n, h, :] - a32[off:off + n, :]))) for off in (h, h + 1, h + 2)}
    best = min(res, key=res.get)
    if best == h + 1:
        ok(f'alignment OK: pred[t,h] lines up with actual[t+h+1] (MAE {res[h + 1]:.2f} mph at +15 min)')
    else:
        fail(f'ALIGNMENT PROBLEM: pred[t,h] matches actual[t+{best}] best, expected t+{h + 1}. MAE by offset: {res}')
    mp = D / 'metrics.json'
    if mp.exists():
        try:
            reported = float(json.loads(mp.read_text())[meta['main_model']]['mae']['15'])
            if abs(reported - res[h + 1]) / max(reported, 1e-6) > 0.5:
                warn(f'metrics.json MAE@15 = {reported:.2f} but the files give {res[h + 1]:.2f} on this window '
                     '(expected on mock data; on real data ask the backend teammate)')
        except Exception:
            pass

mp = D / 'metrics.json'
if mp.exists():
    metrics = json.loads(mp.read_text())
    bad = []
    for m in meta['models']:
        if m not in metrics: bad.append(f'{m}: no entry'); continue
        for k in ('mae', 'rmse', 'mape', 'accuracy', 'macro_f1'):
            for hz in ('15', '30', '60'):
                try: float(metrics[m][k][hz])
                except Exception: bad.append(f'{m}.{k}.{hz}')
    if bad: warn(f'metrics.json missing or invalid values (shown as n/a): {bad[:8]}')
    else: ok('metrics.json has every model, metric and horizon')
else:
    fail('metrics.json missing')

need_csv('robustness.csv', ['model', 'missing_pct', 'mae'])
need_csv('ablation.csv', ['variant', 'mae_15', 'mae_30', 'mae_60'])

for f in sorted(D.iterdir()):
    if f.is_file() and f.stat().st_size > 25 * 1024 * 1024:
        fail(f'{f.name} is {f.stat().st_size / 1e6:.0f} MB, the limit is 25 MB')

done()
