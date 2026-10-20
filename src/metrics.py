import numpy as np

HZ = {'15': 2, '30': 5, '60': 11}      # step index for +15, +30, +60 minutes


def level(speed, ff, th=(0.8, 0.5)):
    r = speed / ff
    out = np.where(r >= th[0], 0, np.where(r >= th[1], 1, 2))
    return np.where(np.isnan(speed), -1, out)


def reg(pred, y):
    m = ~np.isnan(y)
    e = (pred - y)[m]
    return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean())), float((np.abs(e) / y[m]).mean() * 100)


def clf(pred, y, ff):
    m = ~np.isnan(y)
    lp, ly = level(pred, ff)[m], level(y, ff)[m]
    f1 = []
    for c in (0, 1, 2):
        tp = ((lp == c) & (ly == c)).sum()
        fp = ((lp == c) & (ly != c)).sum()
        fn = ((lp != c) & (ly == c)).sum()
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float((lp == ly).mean()), float(np.mean(f1))


def evaluate(pred, y, ff):
    """pred, y: [S,12,N] in mph; ff: [N] free-flow speeds. Returns the metrics.json format."""
    out = {k: {} for k in ('mae', 'rmse', 'mape', 'accuracy', 'macro_f1')}
    for name, h in HZ.items():
        mae, rmse, mape = reg(pred[:, h], y[:, h])
        acc, f1 = clf(pred[:, h], y[:, h], ff)
        out['mae'][name], out['rmse'][name], out['mape'][name] = round(mae, 3), round(rmse, 3), round(mape, 2)
        out['accuracy'][name], out['macro_f1'][name] = round(acc, 3), round(f1, 3)
    return out


def show(results):
    print(f"{'model':22s}{'MAE15':>7s}{'MAE30':>7s}{'MAE60':>7s}{'RMSE60':>8s}{'Acc30':>7s}{'F1_30':>7s}")
    for m, r in results.items():
        print(f"{m:22s}{r['mae']['15']:7.2f}{r['mae']['30']:7.2f}{r['mae']['60']:7.2f}"
              f"{r['rmse']['60']:8.2f}{r['accuracy']['30']:7.3f}{r['macro_f1']['30']:7.3f}")