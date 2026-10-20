import json, os, warnings
import numpy as np, pandas as pd
from prep import prepare, HOR
from metrics import evaluate, show

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)
P = prepare()
speed, ff = P['speed'], P['free_flow']
lo, hi = P['bounds']['train']
t = P['t_idx']['test']
yi = t[:, None] + np.arange(1, HOR + 1)[None, :]
y = speed[yi]                                           # [S,12,N] true future speeds
sm = np.nanmean(speed[lo:hi], axis=0)                   # per-sensor training mean

# 1) Historical Average: mean speed per sensor for each (weekday, 5-minute slot) in the training period
key_all = P['dow'] * 288 + P['bin']
tr, ktr = speed[lo:hi], key_all[lo:hi]
ha = np.stack([np.nanmean(tr[ktr == k], axis=0) if (ktr == k).any() else sm for k in range(7 * 288)])
ha = np.where(np.isnan(ha), sm, ha).astype(np.float32)
pred_ha = ha[key_all[yi]]

# 2) Persistence: repeat the last observed speed
filled = pd.DataFrame(speed).ffill().values
filled = np.where(np.isnan(filled), sm, filled).astype(np.float32)
pred_pe = np.repeat(filled[t][:, None, :], HOR, axis=1)

results = {}
for name, pred in [('historical_average', pred_ha), ('persistence', pred_pe)]:
    results[name] = evaluate(pred, y, ff)
    np.save(f'results/pred_{name}_test.npy', pred.astype(np.float16))
json.dump(results, open('results/metrics_baselines.json', 'w'), indent=2)
print('test windows:', len(t))
show(results)