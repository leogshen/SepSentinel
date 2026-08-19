# Migration Guide: Moving SepSentinel to the GPU Machine

Everything needed to continue work on another machine (Windows + NVIDIA GPU).

## 1. Code

All code travels via git. After cloning:

```
git clone https://github.com/leogshen/SepSentinel
cd SepSentinel
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then replace CPU torch with the CUDA build (pick the CUDA version matching
the installed driver; cu124 shown):

```
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Dev machine used Python 3.14; anything 3.10+ works.

## 2. Data

### PhysioNet CinC 2019 (primary training data, 52 MB)
Not in git. Re-download on the new machine (public Kaggle dataset, kagglehub
handles caching):

```python
import kagglehub
path = kagglehub.dataset_download("tea340yashjoshi/sepsis-prediction-dataset")
print(path)
```

The loader (`sepsentinel/data/physionet.py`) looks in
`~/.cache/kagglehub/datasets/tea340yashjoshi/sepsis-prediction-dataset/versions/1/Dataset.csv`
by default, which is exactly where kagglehub puts it.

### Cytokine reference datasets (not needed for training)
Gitignored (`*.sav`). On the old machine they live in `C:\Users\openq\Downloads`:
- `journal.pone.0260623.s001.sav` (Ozger et al., PLOS ONE) — re-download from
  https://doi.org/10.1371/journal.pone.0260623 supplement S1
- `doi_10_7272_Q6MS3R18__v20240509.zip` (Dryad) — re-download from
  https://datadryad.org/dataset/doi:10.7272/Q6MS3R18
Findings from both are already summarized in DATASETS.md; the raw files are
only needed to redo that analysis.

## 3. Trained checkpoints

`*.pt` files are gitignored. JSON results, logs, and plots ARE committed under
`results/`. If you want the actual model weights on the new machine, zip and
copy these from the old machine (or simply retrain — each run is minutes on GPU):

- `results/experiment6_mae/20260817_143526/mae_pretrain/mae_best.pt` (MAE encoder)
- `results/experiment6_mae/20260817_143526/finetune/*/checkpoints/best_model.pt` (9 models)
- `results/feature_ablation/20260808_135533/experiments/I_all_minus_creatinine/checkpoints/best_model.pt`
  (production Config I model, used by experiment 5 phase 0)

## 4. Reproducing / continuing experiments

All experiments are CLI scripts run from the repo root. On the GPU machine add
`--device cuda`:

```
python experiment2_imputation.py                     # imputation strategy ablation
python experiment3_feature_ablation.py               # feature ablation (Config I here)
python experiment5_recall_study.py --device cuda     # threshold sweep + loss study
python experiment6_mae_pretraining.py --device cuda  # MAE pretraining + fine-tune
python reeval_patient_metrics.py                     # corrected patient-level metrics
python diagnose_pairing_bug.py                       # pairing-bug verification
```

Frozen hyperparameters (do not change without a new experiment): split seed 42,
70/15/15 patient-level stratified split, seeds {42,123,456}, epochs 50, batch 32,
lr 1e-3, patience 7 — all defined in `experiment2_imputation.py` and imported
everywhere else.

## 5. State of play (2026-08-19)

- **Production model**: Config I (9 features, no creatinine), Strategy B
  preprocessing (causal ffill + mask + delta, 19 channels), causal Transformer
  (d_model=64, 2 layers). Test AUROC 0.814.
- **CRITICAL bug fixed 2026-08-19**: patient-metric pairing in
  `collect_patient_predictions` (collate sorts batches by length; metadata was
  attached by dataset order). All patient-level metrics reported before this
  date were wrong and PESSIMISTIC. Corrected numbers:
  `results/experiment6_mae/20260817_143526/summary/corrected_patient_metrics.json`.
  Corrected headline: at 0.70 patient recall -> precision 9.3%, 1.7 false
  alerts/patient-day, median lead 23.5h.
- **Experiment 6 verdict**: MAE pretraining does not improve discrimination
  (data ceiling, not representation ceiling); it does cut seed variance 3.6x.
- **Waiting on**: PhysioNet credentialing (MIMIC-IV etc.) — 5 weeks pending.
- **Queued ideas**: Experiment 7 = TabPFN baseline (installed, v8.2.0, unused);
  Chronos/MOMENT frozen-embedding probe on dense vitals; unified MAE across
  availability classes once MIMIC-IV lands.
- **Experiment 4** (trajectory gating) exists but has never been run.

## 6. Remote access option

If instead of migrating you want the current session to drive the GPU box
remotely: install OpenSSH Server (Windows optional feature) + Tailscale on the
GPU machine, then commands can be run from the dev machine via
`ssh user@gpu-box "cd SepSentinel && python experiment6_mae_pretraining.py --device cuda"`.
