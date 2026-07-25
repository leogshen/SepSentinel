# Stage 1 Model B Results

Per-timestep sepsis prediction using only physiological signals.

## Dataset

**PhysioNet/CinC 2019 Sepsis Challenge** (Reyna et al.)
- 14,057 patients (12,818 healthy, 1,239 septic; 8.8% sepsis prevalence)
- Hourly ICU time-series, variable length (8-336 hours, mean 39)
- Downloaded via kagglehub

### Selected Features (Stage 1)

| Feature | PhysioNet Column | NaN Density |
|---------|-----------------|-------------|
| Heart Rate | HR | 7.7% |
| SpO2 | O2Sat | 12.0% |
| Temperature | Temp | 66.2% |
| Respiratory Rate | Resp | 9.8% |

Stage 1 uses only physiological signals available across all patients.
Biomarkers (Lactate, pH, IL-6) are reserved for Stages 2-3.

## Data Split

Patient-level stratified split (70/15/15), ensuring all timesteps from one
patient stay in exactly one partition.

| Split | Patients | Healthy | Septic | Sepsis % | Timesteps |
|-------|----------|---------|--------|----------|-----------|
| Train | 9,839 | 8,972 | 867 | 8.8% | 382,661 |
| Val | 2,109 | 1,923 | 186 | 8.8% | 82,478 |
| Test | 2,109 | 1,923 | 186 | 8.8% | 80,983 |

Random seed: 42. Stratification on patient-level sepsis label.

## Preprocessing

Applied in order, fitted on training data only:

1. **Outlier clipping** to physiological ranges (e.g. HR 20-250, SpO2 50-100)
2. **Forward-fill** within each patient episode
3. **Mean imputation** for remaining NaNs (using training-set per-feature means)
4. **Z-score normalization** (training-set mean and std)

Temperature has 66% missing values. Forward-fill handles temporal gaps within
a stay; mean imputation covers patients with no recorded temperature at all.

## Models

Five architectures compared, spanning flat (no temporal context) and sequential
(per-timestep prediction with temporal dependencies):

### Flat Models

| Model | Description |
|-------|-------------|
| **Random Forest** | 100-tree ensemble on per-timestep feature vectors (no temporal context) |
| **XGBoost** | 200-tree gradient boosting with `scale_pos_weight` for class imbalance |

Flat models treat each hourly observation as an independent sample.

### Sequential Models

All sequential models produce per-timestep logits, trained with masked
`BCEWithLogitsLoss` (padding ignored), class-weighted with pos_weight=45.1.
Early stopping on validation AUROC (patience=7). Adam optimizer with
ReduceLROnPlateau scheduling.

| Model | Architecture | Parameters |
|-------|-------------|------------|
| **GRU** | 2-layer GRU (hidden=64) + linear head, pack_padded_sequence | 38,465 |
| **TCN** | 4-layer causal dilated convolutions (hidden=64, kernel=3) + residual blocks | 88,705 |
| **Transformer** | Linear projection + sinusoidal PE + 2-layer causal self-attention (d=64, 4 heads) | 67,329 |

## Results

### Held-Out Test Set Metrics

| Model | AUROC | AUPRC | F1 | Recall | Precision | Specificity | Time |
|-------|-------|-------|-----|--------|-----------|-------------|------|
| **Transformer** | **0.7926** | **0.1180** | **0.1295** | 0.6212 | 0.0723 | 0.8215 | 10.8m |
| TCN | 0.7867 | 0.1092 | 0.1093 | 0.6770 | 0.0594 | 0.7601 | 9.1m |
| GRU | 0.7815 | 0.1087 | 0.1259 | 0.6026 | 0.0703 | 0.8215 | 14.3m |
| XGBoost | 0.6380 | 0.0384 | 0.0706 | 0.4605 | 0.0382 | 0.7448 | 1s |
| Random Forest | 0.5762 | 0.0272 | 0.0038 | 0.0023 | 0.0118 | 0.9998 | 2.2m |

All sequential models trained on CPU. Training times would be substantially
lower on GPU.

### Best Model: Transformer

- **AUROC 0.793**: Good discrimination between septic and non-septic timesteps
- **AUPRC 0.118**: Low in absolute terms, but reflects the extreme class
  imbalance (2.2% positive rate); 5.4x better than the 0.022 random baseline
- **Recall 0.621 / Specificity 0.822** at default threshold (0.5): catches 62%
  of sepsis timesteps while correctly classifying 82% of healthy timesteps

### Interpretation

1. **Sequential models vastly outperform flat models.** The GRU, TCN, and
   Transformer all achieve AUROC ~0.78-0.79, while XGBoost and Random Forest
   score 0.58-0.64. Temporal context (trends in vitals over hours) is critical
   for early sepsis detection.

2. **The three sequential architectures perform similarly.** Transformer has a
   slight edge in AUROC and AUPRC. TCN has the highest recall (0.677),
   catching the most sepsis events at the cost of more false positives.

3. **Low absolute AUPRC is expected.** With only 2.2% positive rate, even a
   well-calibrated model will have low precision at moderate recall thresholds.
   AUPRC of 0.118 is 5.4x the random baseline (0.022), indicating the models
   are learning meaningful signal.

4. **Random Forest achieves 99.98% specificity but 0.2% recall** — it simply
   predicts "healthy" for almost every timestep. This confirms that accuracy
   alone is misleading for imbalanced clinical datasets.

## Current Limitations

- **Stage 1 features only.** Biomarkers (Lactate, pH) are unavailable in most
  PhysioNet records (~89-97% NaN). Adding them requires MIMIC-IV or targeted
  cohort selection.
- **Default threshold (0.5).** No threshold optimization has been performed.
  Clinical deployment would require tuning the operating point for the desired
  sensitivity/specificity tradeoff.
- **No feature engineering.** Models receive raw (normalized) values only.
  Derived features (delta HR, trend slopes, rolling statistics) could improve
  discriminative power.
- **CPU-only training.** Training times are practical but would benefit from
  GPU acceleration for hyperparameter sweeps.
- **Single random seed.** Results may vary with different splits; cross-validation
  or repeated trials would provide confidence intervals.

## Recommended Next Step

**Threshold optimization and clinical operating-point analysis.** The default
0.5 threshold is arbitrary. Sweeping thresholds on the validation set to find
the optimal sensitivity/specificity tradeoff — guided by clinical requirements
(e.g., target >=90% sensitivity) — would make these results actionable. This
should precede adding Stage 2 biomarker features.

## Reproducibility

```bash
pip install -r requirements.txt
python train_stage1.py                     # all 5 models, ~40 min on CPU
python train_stage1.py --models gru tcn    # specific models only
python train_stage1.py --epochs 30         # override max epochs
```

Requires the PhysioNet dataset at `~/.cache/kagglehub/datasets/tea340yashjoshi/sepsis-prediction-dataset/versions/1/Dataset.csv`.
To download: `python -c "import kagglehub; kagglehub.dataset_download('tea340yashjoshi/sepsis-prediction-dataset')"`.
