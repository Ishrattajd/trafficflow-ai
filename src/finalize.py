import json, os
import numpy as np, pandas as pd, torch
from prep import prepare, HOR
from model import TrafficFlowST
from train import predict

P = prepare()
N = P['speed'].shape[1]
os.makedirs('app_data', exist_ok=True)
t_test = P['t_idx']['test']
NAMES = ['trafficflow_st', 'no_graph']
BASE = ['historical_average', 'persistence']


def load(name):
    sup = np.stack([np.eye(N), np.eye(N)]).astype(np.float32) if name == 'no_graph' else P['supports']
    m = TrafficFlowST(torch.from_numpy(sup), hid=48, layers=1)
    m.load_state_dict(torch.load(f'results/{name}.pt'))
    m.eval()
    return m


models = {n: load(n) for n in NAMES if os.path.exists(f'results/{n}.pt')}
print('models found:', list(models))

# ---- demo window: Tuesday 5 June 2012, 3 days of forecast times (inside the test period)
t0 = int(np.where(P['timestamps'] == '2012-06-05 00:00')[0][0])
Td = 864
ids = np.arange(t0 - t_test[0], t0 - t_test[0] + Td)
np.save('app_data/actual.npy', P['speed'][t0:t0 + Td].astype(np.float16))
pd.DataFrame({'timestamp': P['timestamps'][t0:t0 + Td]}).to_csv('app_data/timestamps.csv', index=False)
for n, m in models.items():
    np.save(f'app_data/pred_{n}.npy', predict(m, P, 'test', ids=ids).astype(np.float16))
for b in BASE:
    np.save(f'app_data/pred_{b}.npy', np.load(f'results/pred_{b}_test.npy')[ids])

# ---- sensors and edges
loc = pd.read_csv('data/processed/sensors_raw.csv')
pd.DataFrame({'sensor_id': range(N), 'latitude': loc['latitude'], 'longitude': loc['longitude'],
              'original_id': loc['sensor_id']}).to_csv('app_data/sensors.csv', index=False)
A = P['adj']
rows = [(i, j, round(float(A[i, j]), 4)) for i in range(N) for j in range(N) if i != j and A[i, j] > 0]
pd.DataFrame(rows, columns=['source', 'target', 'weight']).to_csv('app_data/edges.csv', index=False)

# ---- metrics (full test set, 6,832 windows)
metrics = json.load(open('results/metrics_baselines.json'))
for n in NAMES:
    if os.path.exists(f'results/metrics_{n}.json'):
        metrics.update(json.load(open(f'results/metrics_{n}.json')))
json.dump(metrics, open('app_data/metrics.json', 'w'), indent=2)

# ---- robustness: hide a share of sensors at test time (every 4th test window to save time)
sub = np.arange(0, len(t_test), 4)
ysub = P['speed'][t_test[sub][:, None] + np.arange(1, HOR + 1)[None, :]]
rob = []
rob_models = dict(models)
if os.path.exists('results/no_mask.pt'):
    rob_models['no_mask'] = load('no_mask')
for n, m in rob_models.items():
    for pct in (0, 10, 20, 30, 50):
        pr = predict(m, P, 'test', drop_frac=pct / 100, ids=sub)
        mae = float(np.nanmean(np.abs(pr - ysub)))
        rob.append((n, pct, round(mae, 3)))
        print(f'robustness {n:15s} missing {pct:2d}% -> MAE {mae:.3f}', flush=True)
pd.DataFrame(rob, columns=['model', 'missing_pct', 'mae']).to_csv('app_data/robustness.csv', index=False)

# ---- ablation: one row per trained variant
abl = []
for label, key in [('Full model (with road graph)', 'trafficflow_st'), ('Without road graph', 'no_graph'),
                   ('Without sensor masking', 'no_mask'), ('Without calendar features', 'no_cal')]:
    f = f'results/metrics_{key}.json'
    if os.path.exists(f):
        r = json.load(open(f))[key]['mae']
        abl.append((label, r['15'], r['30'], r['60']))
pd.DataFrame(abl, columns=['variant', 'mae_15', 'mae_30', 'mae_60']).to_csv('app_data/ablation.csv', index=False)

# ---- meta
meta = {'dataset': 'METR-LA', 'region': 'Los Angeles, CA', 'n_sensors': N, 'horizon_steps': HOR,
        'step_minutes': 5, 'n_timesteps': Td, 'models': list(models) + BASE, 'main_model': 'trafficflow_st',
        'thresholds': {'free': 0.8, 'slow': 0.5},
        'free_flow_speed': [round(float(v), 1) for v in P['free_flow']]}
json.dump(meta, open('app_data/meta.json', 'w'), indent=2)
print('app_data written:', sorted(os.listdir('app_data')))