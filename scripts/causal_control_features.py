"""Causal summaries of exactly the history available to the sequence encoder."""
import numpy as np

WINDOWS = (6, 12, 24, 72)
STATS = ('last', 'mean', 'std', 'min', 'max', 'slope', 'change', 'count', 'age', 'seen')


def prefix_features(e, median, std):
    x = np.clip((np.asarray(e['x'], np.float32) - median) / std, -10, 10)
    t = np.asarray(e['t'], np.float64)
    if len(t) > 1 and not np.all(np.diff(t) == 1):
        raise ValueError('These summaries require the audited unit-hour target grid')
    T, F = x.shape
    obs = np.isfinite(x)
    z = np.where(obs, x, 0).astype(np.float64)
    index = np.arange(T)[:, None]
    last = np.maximum.accumulate(np.where(obs, index, -1), axis=0)
    first = np.minimum.accumulate(np.where(obs, index, T)[::-1], axis=0)[::-1]

    def cumulative(a):
        return np.vstack([np.zeros((1, F)), np.cumsum(a, axis=0)])

    cum = [cumulative(a) for a in (obs, z, z*z, obs*t[:, None],
                                  obs*t[:, None]**2, z*t[:, None])]
    blocks = []
    for w in WINDOWS:
        begin = np.maximum(0, np.arange(T)-w+1)
        n, total, square, st, st2, stx = [a[1:]-a[begin] for a in cum]
        seen = n > 0
        mean = total / np.maximum(n, 1)
        variance = np.maximum(0, square/np.maximum(n, 1)-mean*mean)
        den = n*st2-st*st
        slope = np.divide(n*stx-st*total, den, out=np.zeros_like(total), where=den > 0)
        latest = z[np.maximum(last, 0), np.arange(F)]
        earliest = z[np.minimum(first[begin], T-1), np.arange(F)]
        age = np.where(seen, t[:, None]-t[np.maximum(last, 0)], w)
        padded = np.pad(x, ((w-1, 0), (0, 0)), constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(padded, w, axis=0)
        low = np.min(np.where(np.isfinite(windows), windows, np.inf), axis=-1)
        high = np.max(np.where(np.isfinite(windows), windows, -np.inf), axis=-1)
        values = [latest, mean, np.sqrt(variance), low, high, slope,
                  latest-earliest, n, age, seen]
        # Missing values map to normalized median/zero, accompanied by count and seen.
        values = [np.where(seen, v, 0) if k < 8 else v for k, v in enumerate(values)]
        blocks.append(np.stack(values, axis=-1).reshape(T, -1))
    static = np.broadcast_to(e['static'], (T, 4))
    result = np.concatenate([*blocks, static, t[:, None]], axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite causal feature')
    return result


def columns(nfeatures, total_features=34):
    block = total_features * len(STATS)
    return np.r_[np.concatenate([np.arange(w*block, w*block+nfeatures*len(STATS))
                                 for w in range(len(WINDOWS))]),
                 np.arange(len(WINDOWS)*block, len(WINDOWS)*block+5)]
