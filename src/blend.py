import json
import numpy as np, torch
from prep import prepare, HOR
from model import TrafficFlowST
from metrics import evaluate, show
from train import predict

P = prepare()
speed, ff = P['speed'], P['free_flow']
lo, hi = P['bounds']['train']
sm = np.nanmean(speed[lo:hi], axis=0)
key_all = P['dow'] * 288 + P['bin']
tr, ktr = speed[lo:hi], key_all[lo:hi]
ha = np.stack([np.nanmean(tr[ktr == k], axis=0) if (ktr == k).any() else sm for k in range(7 * 288)])
ha = np.where(np.isnan(ha), sm, ha).astype(np.float32)


def windows(split):
    t = P['t_idx'][split]
    yi = t[:, None] + np.arange(1, HOR + 1)[None, :]
    return ha[key_all[yi]], speed[yi]


model = TrafficFlowST(torch.from_numpy(P['supports']), hid=48, layers=1)
model.load_state_dict(torch.load('results/trafficflow_st.pt'))
model.eval()

# weights chosen on the VALIDATION set only (no test leakage)
hv, yv = windows('val')
mv = predict(model, P, 'val')
grid = np.linspace(0, 1, 21)
w = np.array([grid[np.argmin([np.nanmean(np.abs(g * mv[:, h] + (1 - g) * hv[:, h] - yv[:, h])) for g in grid])]
              for h in range(HOR)], dtype=np.float32)
print('weight on neural model, +5 .. +60 min:', w.round(2))

ht, yt = windows('test')
mt = predict(model, P, 'test')
blend = w[None, :, None] * mt + (1 - w)[None, :, None] * ht
NAME = 'trafficflow_blend'
res = evaluate(blend, yt, ff)
json.dump({NAME: res}, open(f'results/metrics_{NAME}.json', 'w'), indent=2)

# demo window, same as finalize.py
t_test = P['t_idx']['test']
t0 = int(np.where(P['timestamps'] == '2012-06-05 00:00')[0][0])
ids = np.arange(t0 - t_test[0], t0 - t_test[0] + 864)
np.save(f'app_data/pred_{NAME}.npy', blend[ids].astype(np.float16))

metrics = json.load(open('app_data/metrics.json'))
metrics[NAME] = res
json.dump(metrics, open('app_data/metrics.json', 'w'), indent=2)
meta = json.load(open('app_data/meta.json'))
if NAME not in meta['models']:
    meta['models'].insert(0, NAME)
score = {m: np.mean([r['mae'][k] for k in ('15', '30', '60')]) for m, r in metrics.items()}
meta['main_model'] = min(score, key=score.get)
json.dump(meta, open('app_data/meta.json', 'w'), indent=2)
print('main model for the demo:', meta['main_model'])
show(metrics)