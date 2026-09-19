# Alarm episodes with a refractory period

Episodes: `results/mimic31_full_ext_prodrome.pkl`. Flat XGBoost, grouped subject split (seed 42), threshold pinned on VALIDATION under the same episode policy that defines the budget (<=1.0 per nonseptic patient-day), applied unchanged to test.
Built in 8.2 min.

`R=0` is per-hour counting, i.e. every number the project has reported so far. Merging hours into episodes lowers burden at a fixed threshold, so the same budget buys a lower threshold, an earlier first crossing, and therefore different recall, lead and capture. That is why capture and lead are recomputed here rather than carried over.

Timestep precision/recall/F1 stay hour-based: they describe how hours are labelled and are not affected by how alarms are merged for the human reader.

| | R=0 h | R=2 h | R=6 h | R=12 h | R=2 h - R=0 h (95% CI) | R=6 h - R=0 h (95% CI) | R=12 h - R=0 h (95% CI) |
|---|---|---|---|---|---|---|---|
| **Patient recall** | 0.636 | 0.760 | 0.910 | 0.979 | +0.124 [+0.105, +0.145] ** | +0.274 [+0.248, +0.301] ** | +0.343 [+0.314, +0.372] ** |
| Patient precision | 0.225 | 0.194 | 0.151 | 0.126 | -0.031 [-0.038, -0.025] ** | -0.075 [-0.085, -0.066] ** | -0.099 [-0.110, -0.089] ** |
| Timestep precision | 0.097 | 0.084 | 0.066 | 0.052 | -0.013 [-0.017, -0.009] ** | -0.031 [-0.038, -0.024] ** | -0.045 [-0.053, -0.037] ** |
| Timestep recall | 0.195 | 0.293 | 0.486 | 0.665 | +0.098 [+0.090, +0.107] ** | +0.292 [+0.274, +0.311] ** | +0.471 [+0.449, +0.492] ** |
| Timestep F1 | 0.130 | 0.131 | 0.116 | 0.097 | +0.002 [-0.003, +0.006] | -0.013 [-0.022, -0.005] ** | -0.033 [-0.042, -0.023] ** |
| Median lead (h) | 20.58 | 21.00 | 21.73 | 23.00 | +0.296 [-1.000, +1.559] | +1.214 [-0.241, +2.916] | +2.547 [+1.000, +4.092] ** |
| Capture >=3h | 0.590 | 0.719 | 0.892 | 0.972 | +0.130 [+0.110, +0.150] ** | +0.302 [+0.276, +0.331] ** | +0.383 [+0.355, +0.412] ** |
| Capture >=6h | 0.533 | 0.654 | 0.828 | 0.927 | +0.121 [+0.101, +0.141] ** | +0.295 [+0.268, +0.323] ** | +0.394 [+0.364, +0.424] ** |
| Capture >=12h | 0.422 | 0.520 | 0.662 | 0.746 | +0.098 [+0.082, +0.117] ** | +0.240 [+0.215, +0.266] ** | +0.324 [+0.297, +0.353] ** |
| Realised burden (episodes/pt-day) | 0.93 | 1.01 | 1.00 | 1.00 | +0.082 [+0.057, +0.106] ** | +0.071 [+0.022, +0.116] ** | +0.066 [+0.009, +0.121] ** |
| **Nonseptic patients ever alarmed** | 0.280 | 0.404 | 0.658 | 0.867 | +0.124 [+0.117, +0.131] ** | +0.378 [+0.368, +0.389] ** | +0.587 [+0.576, +0.598] ** |

| Refractory | Threshold (from val) |
|---|---|
| R=0 h | 0.73 |
| R=2 h | 0.68 |
| R=6 h | 0.58 |
| R=12 h | 0.47 |

`**` marks a paired 95% interval excluding zero, from 2000 patient-level resamples.
