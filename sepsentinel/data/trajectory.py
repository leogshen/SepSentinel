# Causal trajectory feature computation for time-series clinical data.
#
# Per raw feature, computes three trajectory descriptors:
#   diff_1h:  x[t] - x[t-1]               (immediate rate of change)
#   mean_6h:  causal rolling mean           (recent patient baseline)
#   dev_6h:   x[t] - mean_6h[t]            (deviation from baseline)
#
# All computations are strictly causal (no future information).
# Applied AFTER imputation/clipping, BEFORE z-score normalization.

import numpy as np

TRAJ_SUFFIXES = ["diff_1h", "mean_6h", "dev_6h"]
N_TRAJ_PER_FEATURE = len(TRAJ_SUFFIXES)


def compute_trajectory(values, window=6):
    """Compute causal trajectory features for all features.

    Args:
        values: (n_steps, n_features) — imputed, clipped values (pre-normalization).
        window: Rolling mean window size in timesteps (hours for hourly data).

    Returns:
        (n_steps, n_features * 3) — interleaved [diff, mean, dev] per feature.
    """
    n_steps, n_feat = values.shape

    # Precompute window parameters (vectorized across timesteps)
    indices = np.arange(n_steps)
    starts = np.maximum(0, indices - window + 1)
    window_sizes = (indices - starts + 1).astype(np.float32)

    # diff_1h: immediate change, 0 at t=0
    diff = np.zeros_like(values)
    diff[1:] = values[1:] - values[:-1]

    # rolling_mean_6h: causal mean over last `window` steps (vectorized)
    cs = np.vstack([np.zeros(n_feat, dtype=np.float64), np.cumsum(values, axis=0)])
    rmean = ((cs[indices + 1] - cs[starts]) / window_sizes[:, None]).astype(np.float32)

    # dev_from_rolling: deviation from recent baseline
    dev = values - rmean

    # Interleave: [diff_f0, mean_f0, dev_f0, diff_f1, mean_f1, dev_f1, ...]
    out = np.zeros((n_steps, n_feat * N_TRAJ_PER_FEATURE), dtype=np.float32)
    out[:, 0::3] = diff
    out[:, 1::3] = rmean
    out[:, 2::3] = dev

    return out
