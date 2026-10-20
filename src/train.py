import argparse, json, os, time
import numpy as np, torch
from prep import prepare, get_batch, HOR
from model import TrafficFlowST
from metrics import evaluate, show


def apply_drop(x, drop):
    """hide sensors (speed + observed flag) where drop [B,1,N] is True"""
    x = x.clone()
    x[..., 0:2] = x[..., 0:2] * (~drop).float().unsqueeze(-1)
    return x


def predict(model, P, split, drop_frac=0.0, ids=None, bs=256, cal=True):
    model.eval()
    ids = np.arange(len(P['t_idx'][split])) if ids is None else ids
    rng = np.random.default_rng(0)
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), bs):
            x = torch.from_numpy(get_batch(P, split, ids[i:i + bs])[0])
            if not cal:
                x[..., 2:] = 0
            if drop_frac > 0:
                x = apply_drop(x, torch.from_numpy(rng.random((x.shape[0], 1, x.shape[2])) < drop_frac))
            out.append(model(x).numpy())
    return np.concatenate(out) * P['std'] + P['mean']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='trafficflow_st')
    ap.add_argument('--minutes', type=float, default=20)
    ap.add_argument('--spe', type=int, default=150, help='batches per epoch')
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--hid', type=int, default=48)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--no_graph', action='store_true')
    ap.add_argument('--no_mask', action='store_true')
    ap.add_argument('--no_cal', action='store_true')
    a = ap.parse_args()

    os.makedirs('results', exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)
    torch.set_num_threads(os.cpu_count() or 4)
    P = prepare()
    N = P['speed'].shape[1]
    sup = np.stack([np.eye(N), np.eye(N)]).astype(np.float32) if a.no_graph else P['supports']
    model = TrafficFlowST(torch.from_numpy(sup), hid=a.hid, layers=a.layers)
    print('run:', a.name, '| parameters:', sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    n_tr = len(P['t_idx']['train'])
    val_ids = np.arange(0, len(P['t_idx']['val']), 6)
    yv = get_batch(P, 'val', val_ids)[1]

    t0, best, ep = time.time(), 1e9, 0
    while time.time() - t0 < a.minutes * 60:
        ep += 1
        model.train()
        losses = []
        for _ in range(a.spe):
            x, y = get_batch(P, 'train', np.random.randint(0, n_tr, a.bs))
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            if a.no_cal:
                x[..., 2:] = 0
            if not a.no_mask:                                   # robustness: hide up to 20% of sensors
                x = apply_drop(x, torch.rand(x.shape[0], 1, x.shape[2]) < np.random.uniform(0, 0.2))
            m = ~torch.isnan(y)
            yn = torch.where(m, (y - P['mean']) / P['std'], torch.zeros_like(y))
            loss = (torch.abs(model(x) - yn) * m).sum() / m.sum()   # masked MAE
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        v = float(np.nanmean(np.abs(predict(model, P, 'val', ids=val_ids, cal=not a.no_cal) - yv)))
        sched.step(v)
        flag = ''
        if v < best:
            best = v
            torch.save(model.state_dict(), f'results/{a.name}.pt')
            flag = '  <- saved'
        print(f'epoch {ep:2d} | train {np.mean(losses) * P["std"]:.3f} mph | val MAE {v:.3f} | {int(time.time() - t0)}s{flag}', flush=True)

    model.load_state_dict(torch.load(f'results/{a.name}.pt'))
    t = P['t_idx']['test']
    y = P['speed'][t[:, None] + np.arange(1, HOR + 1)[None, :]]
    pred = predict(model, P, 'test', cal=not a.no_cal)
    res = {a.name: evaluate(pred, y, P['free_flow'])}
    np.save(f'results/pred_{a.name}_test.npy', pred.astype(np.float16))
    json.dump(res, open(f'results/metrics_{a.name}.json', 'w'), indent=2)
    allr = {}
    for f in os.listdir('results'):
        if f.startswith('metrics_') and f.endswith('.json'):
            allr.update(json.load(open(f'results/{f}')))
    show(allr)


if __name__ == '__main__':
    main()