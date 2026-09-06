# Event-stream -> hourly-grid episode construction for EHR datasets
# (MIMIC-IV, SICdb). Produces the same episode schema as
# physionet.load_physionet(), plus multi-stay fields.
#
# Grid semantics (DATA_ACCESS_SPEC.md section 5):
#   - Hour bins [t, t+1) from ICU admission (intime).
#   - Vitals: median of in-bin measurements (robust to charting bursts).
#   - Labs: last in-bin value (by event time).
#   - Observation mask for Strategy B comes from NaN pattern of the grid,
#     exactly as with PhysioNet (a bin with no measurement stays NaN).
#   - Labels generated from unshifted t_sepsis_hour at load time:
#         labels[t] = 1  if  t >= t_sepsis_hour - label_shift_hours
#     (label_shift_hours=6 reproduces PhysioNet 2019 semantics).

import numpy as np

MAX_HOURS_DEFAULT = 336          # 14 days; PhysioNet max, keeps T^2 attention sane
POST_ONSET_TRUNCATE_H = 24       # keep at most this many hours after t_sepsis


def grid_stay(events, n_hours, features, vitals, sums=()):
    """Grid one stay's events into an hourly (n_hours, n_features) array.

    Args:
        events: iterable of (feature_name, hours_from_intime, value) tuples,
            any order. Events outside [0, n_hours) are dropped.
        n_hours: Grid length.
        features: Ordered feature list (vitals first, then labs).
        vitals: Set/list of feature names aggregated by median;
            all others use last-in-bin.
        sums: Feature names aggregated by SUM within the bin. Volumes such as
            urine output are accumulated, not sampled -- taking the median or
            the last of three charted volumes in one hour would understate the
            hour's output.

    Returns:
        (n_hours, len(features)) float32 array, NaN where unobserved.
    """
    vitals = set(vitals)
    sums = set(sums)
    feat_idx = {f: j for j, f in enumerate(features)}

    # Collect per-cell values; labs keep (time, value) to resolve "last".
    cells = {}
    for feat, hours, value in events:
        j = feat_idx.get(feat)
        if j is None or value is None:
            continue
        if not np.isfinite(value):
            continue
        t = int(hours)
        if hours < 0 or t >= n_hours:
            continue
        cells.setdefault((t, j), []).append((hours, float(value)))

    grid = np.full((n_hours, len(features)), np.nan, dtype=np.float32)
    for (t, j), vals in cells.items():
        if features[j] in sums:
            grid[t, j] = float(np.sum([v for _, v in vals]))
        elif features[j] in vitals:
            grid[t, j] = float(np.median([v for _, v in vals]))
        else:
            grid[t, j] = max(vals)[1]  # latest event time wins
    return grid


def make_labels(n_hours, t_sepsis_hour, label_shift_hours=6,
                prodrome_window_h=None):
    """Per-hour labels from unshifted clinical onset.

    Default ("standard"): 1 from t_sepsis - label_shift_hours onward, which
    reproduces PhysioNet 2019 semantics.

    With `prodrome_window_h` set, positives are confined to the PRE-ONSET
    window [t_sepsis - prodrome_window_h, t_sepsis) and every hour from
    t_sepsis onward is masked out (NaN) so it can be dropped from training and
    evaluation. Rationale: under the standard target the easiest positives are
    the post-onset hours -- established sepsis, trivially recognisable and
    worth nothing as early warning -- so a model maximising per-hour
    discrimination spends its capacity there and alarms late. Confining
    positives to the prodrome makes early detection the only way to score.

    Returns a float32 array of {0, 1, NaN}; NaN means "exclude this hour".
    """
    labels = np.zeros(n_hours, dtype=np.float32)
    if t_sepsis_hour is None:
        return labels
    if prodrome_window_h is None:
        start = max(0, int(np.ceil(t_sepsis_hour - label_shift_hours)))
        if start < n_hours:
            labels[start:] = 1.0
        return labels

    onset = int(np.ceil(t_sepsis_hour))
    start = max(0, int(np.ceil(t_sepsis_hour - prodrome_window_h)))
    if start < n_hours:
        labels[start:min(onset, n_hours)] = 1.0
    if onset < n_hours:
        labels[onset:] = np.nan          # established sepsis: not a target
    return labels


def build_episode(stay_id, subject_id, events, los_hours, features, vitals,
                  sums=(), t_sepsis_hour=None, dataset="mimic4",
                  label_shift_hours=6, prodrome_window_h=None,
                  max_hours=MAX_HOURS_DEFAULT,
                  post_onset_truncate_h=POST_ONSET_TRUNCATE_H,
                  min_length=6):
    """Build one episode dict in the physionet.py schema (+ multi-stay fields).

    Returns None if the stay is shorter than min_length hours after
    truncation, or has no observations at all.
    """
    n_hours = int(min(np.floor(los_hours), max_hours))

    # Post-onset truncation: hours beyond t_sepsis + K add label-1 steps but
    # no early-warning signal (spec section 2).
    if t_sepsis_hour is not None and post_onset_truncate_h is not None:
        n_hours = min(n_hours, int(np.ceil(t_sepsis_hour + post_onset_truncate_h)))

    if n_hours < min_length:
        return None

    signals = grid_stay(events, n_hours, features, vitals, sums=sums)
    if np.all(np.isnan(signals)):
        return None

    labels = make_labels(n_hours, t_sepsis_hour, label_shift_hours,
                         prodrome_window_h)
    if np.nanmax(labels) > 0:
        label = 1
        onset_step = int(np.argmax(labels > 0))
    else:
        label = 0
        onset_step = None

    return {
        "patient_id": str(stay_id),
        "subject_id": str(subject_id),
        "stay_id": str(stay_id),
        "time": np.arange(n_hours, dtype=np.float32),
        "signals": signals,
        "labels": labels,
        "label": label,
        "onset_step": onset_step,
        "t_sepsis_hour": float(t_sepsis_hour) if t_sepsis_hour is not None else None,
        "dataset": dataset,
        "features": list(features),
        "vitals": list(vitals),
    }
