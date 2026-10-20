import json, os
import numpy as np, pandas as pd

OUT = 'app_data_mock'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

N, T, H = 207, 864, 12
ts = pd.date_range('2012-06-04 00:00', periods=T, freq='5min')
lat = 34.05 + rng.normal(0, 0.05, N)
lon = -118.25 + rng.normal(0, 0.08, N)
pd.DataFrame({'sensor_id': range(N), 'latitude': lat, 'longitude': lon}).to_csv(f'{OUT}/sensors.csv', index=False)

order = np.argsort(lon)
rows = [(int(order[i]), int(order[i + k]), 1.0) for i in range(N - 2) for k in (1, 2)]
pd.DataFrame(rows, columns=['source', 'target', 'weight']).to_csv(f'{OUT}/edges.csv', index=False)
pd.DataFrame({'timestamp': ts}).to_csv(f'{OUT}/timestamps.csv', index=False)

hour = ts.hour.values + ts.minute.values / 60
rush = np.exp(-((hour - 8) ** 2) / 3) + np.exp(-((hour - 17) ** 2) / 3)
ff = rng.uniform(60, 70, N)
sens = rng.uniform(0.3, 0.8, N)
actual = ff[None, :] * (1 - sens[None, :] * rush[:, None]) + rng.normal(0, 2, (T, N))
actual = np.clip(actual, 5, 80)
actual[rng.random((T, N)) < 0.01] = np.nan
np.save(f'{OUT}/actual.npy', actual.astype(np.float16))

def make_pred(noise):
    pred = np.zeros((T, H, N), dtype=np.float32)
    for h in range(H):
        idx = np.minimum(np.arange(T) + h + 1, T - 1)
        pred[:, h, :] = np.nan_to_num(actual[idx], nan=50) + rng.normal(0, noise * (1 + h / 6), (T, N))
    return pred.astype(np.float16)

models = {'trafficflow_st': 1.0, 'historical_average': 3.0, 'persistence': 2.0, 'lightgbm': 1.6, 'lstm': 1.4}
metrics = {}
for m, nz in models.items():
    np.save(f'{OUT}/pred_{m}.npy', make_pred(nz))
    base = nz * 1.8
    f = lambda k: {'15': round(base * k, 2), '30': round(base * k * 1.15, 2), '60': round(base * k * 1.35, 2)}
    metrics[m] = {
        'mae': f(1), 'rmse': f(1.5), 'mape': f(2.2),
        'accuracy': {'15': round(0.95 - 0.02 * nz, 3), '30': round(0.93 - 0.02 * nz, 3), '60': round(0.9 - 0.02 * nz, 3)},
        'macro_f1': {'15': round(0.85 - 0.03 * nz, 3), '30': round(0.8 - 0.03 * nz, 3), '60': round(0.75 - 0.03 * nz, 3)},
    }
json.dump(metrics, open(f'{OUT}/metrics.json', 'w'), indent=2)

rob = [(m, p, round(models[m] * 1.8 * (1 + p / 100 * k), 2))
       for m, k in [('trafficflow_st', 0.8), ('lstm', 1.6)] for p in (0, 10, 20, 30, 50)]
pd.DataFrame(rob, columns=['model', 'missing_pct', 'mae']).to_csv(f'{OUT}/robustness.csv', index=False)

abl = [('Full model', 2.9, 3.3, 3.8), ('Without graph', 3.2, 3.7, 4.3),
       ('Without calendar features', 3.0, 3.4, 4.0), ('Without sensor masking', 2.9, 3.4, 3.9)]
pd.DataFrame(abl, columns=['variant', 'mae_15', 'mae_30', 'mae_60']).to_csv(f'{OUT}/ablation.csv', index=False)

meta = {'dataset': 'METR-LA (mock)', 'region': 'Los Angeles, CA', 'n_sensors': N, 'horizon_steps': H,
        'step_minutes': 5, 'n_timesteps': T, 'models': list(models), 'main_model': 'trafficflow_st',
        'thresholds': {'free': 0.8, 'slow': 0.5}, 'free_flow_speed': [round(float(v), 1) for v in ff]}
json.dump(meta, open(f'{OUT}/meta.json', 'w'), indent=2)
print('mock data written to', OUT)
