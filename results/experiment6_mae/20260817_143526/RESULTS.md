# Experiment 6: MAE Self-Supervised Pretraining

> **CORRECTION (2026-08-19)**: A pairing bug in `collect_patient_predictions`
> (batch length-sorting vs dataset-order metadata) scrambled patient attribution
> for 97% of test episodes. **All patient-level numbers in this file (patient
> recall, lead times, alerts/day, capture rates) are invalid and pessimistic.**
> Timestep metrics, AUROC, and AUPRC are unaffected. Corrected patient-level
> metrics: `summary/corrected_patient_metrics.json`. Corrected 3-seed summary
> at 0.70 patient recall:
>
> | Config | Threshold | Precision | Alerts/Day | Median Lead | Cap 6h |
> |--------|-----------|-----------|------------|-------------|--------|
> | A_scratch | 0.622 | 9.31% | 1.68 | 23.5h | 56.1% |
> | B_mae_finetune | 0.608 | 9.71% | 1.54 | 23.2h | 56.8% |
> | C_mae_frozen | 0.532 | 8.81% | 1.81 | 24.7h | 58.6% |
>
> At best-F1 thresholds the corrected patient recall is 30-45% (not 4-8%) with
> *positive* median leads of 22-29h (not negative) and 0.02-0.32 alerts/day.
> The MAE-vs-scratch conclusion (no discrimination gain, stability win) stands.

## Question

Does Masked Autoencoder (MAE) pretraining on unlabeled clinical time series
improve downstream sepsis prediction -- specifically, can it push past the
AUROC ~0.81 ceiling and improve the recall-precision tradeoff?

## Setup

### Data
- PhysioNet/CinC 2019: 14,057 episodes (1,239 septic, 8.8%)
- Split: 9,839 train / 2,109 val / 2,109 test (patient-level, stratified)
- Config I features (9): HR, SpO2, Resp, Temp, Lactate, pH, WBC, Platelets, Bilirubin
- Strategy B preprocessing: 19 input channels (4 vital values + 5 lab values + 5 masks + 5 deltas)

### Phase 0: MAE Pretraining (self-supervised)
- **Task**: Randomly mask 40% of available feature values per timestep, reconstruct them
- **Encoder**: Same TransformerEncoder as production (d_model=64, 2 layers, 4 heads) -- 68K params
- **Decoder**: Lightweight 1-layer transformer (d_model=32) -- 11K params, discarded after pretraining
- **Masking**: When a value is masked, its corresponding observation-mask channel is also zeroed
- **Loss**: MSE on masked positions only (9 value channels, not mask/delta)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01), cosine annealing
- **Convergence**: 70 epochs (early stop at patience=10), best val MSE at epoch 60

### Phase 1: Fine-tuning Comparison (3 seeds x 3 configs = 9 runs)
- **A_scratch**: From-scratch SepsisTransformer (baseline, no pretraining)
- **B_mae_finetune**: MAE-pretrained encoder weights loaded, full model fine-tuned
- **C_mae_frozen**: MAE-pretrained encoder weights loaded and frozen, only classification head trained
- Training: BCE + pos_weight=45.1, Adam lr=1e-3, patience=7, up to 50 epochs

## Results

### MAE Pretraining Curve
```
Epoch  1: val_mse = 0.738
Epoch 10: val_mse = 0.348
Epoch 30: val_mse = 0.291
Epoch 60: val_mse = 0.269  <-- best
Epoch 70: val_mse = 0.270  (early stop)

Reconstruction MSE reduced 72% (0.977 -> 0.269)
Total pretraining time: 70 min (CPU)
```

### Discrimination Metrics (3-seed mean +/- std)

| Config | AUROC | AUPRC | Best-F1 |
|--------|-------|-------|---------|
| A_scratch | **0.814 +/- 0.004** | **0.144 +/- 0.004** | 0.229 |
| B_mae_finetune | 0.816 +/- 0.001 | 0.137 +/- 0.009 | 0.225 |
| C_mae_frozen | 0.776 +/- 0.001 | 0.118 +/- 0.001 | 0.196 |

### At Best-F1 Threshold (3-seed mean)

| Config | Threshold | TS Recall | Precision | Patient Recall |
|--------|-----------|-----------|-----------|----------------|
| A_scratch | 0.878 | 0.255 | 0.209 | **4.8%** |
| B_mae_finetune | 0.865 | 0.267 | 0.194 | 4.6% |
| C_mae_frozen | 0.717 | 0.295 | 0.148 | 8.8% |

### At 0.70 Patient Recall Target (primary evaluation)

| Config | Threshold | Patient Recall | Precision | Alerts/Day | Median Lead (h) | Cap 6h |
|--------|-----------|---------------|-----------|------------|-----------------|--------|
| A_scratch | 0.382 | **72.0%** | **5.42%** | **7.5** | **26.8** | **59.0%** |
| B_mae_finetune | 0.328 | 71.0% | 5.15% | 8.0 | 26.5 | 58.2% |
| C_mae_frozen | 0.372 | 70.4% | 5.02% | 7.4 | 25.3 | 56.1% |

### Per-Seed Detail at 0.70 Patient Recall

**A_scratch (baseline)**
| Seed | AUROC | AUPRC | Threshold | Precision | Alerts/Day | Median Lead |
|------|-------|-------|-----------|-----------|------------|-------------|
| 42 | 0.816 | 0.149 | 0.480 | 5.46% | 7.5 | 27.5h |
| 123 | 0.809 | 0.141 | 0.345 | 5.42% | 7.6 | 26.0h |
| 456 | 0.817 | 0.141 | 0.320 | 5.40% | 7.5 | 27.0h |

**B_mae_finetune**
| Seed | AUROC | AUPRC | Threshold | Precision | Alerts/Day | Median Lead |
|------|-------|-------|-----------|-----------|------------|-------------|
| 42 | 0.815 | 0.150 | 0.330 | 5.07% | 8.3 | 26.5h |
| 123 | 0.816 | 0.128 | 0.435 | 5.32% | 7.7 | 25.5h |
| 456 | 0.816 | 0.134 | 0.220 | 5.05% | 8.2 | 27.5h |

**C_mae_frozen**
| Seed | AUROC | AUPRC | Threshold | Precision | Alerts/Day | Median Lead |
|------|-------|-------|-----------|-----------|------------|-------------|
| 42 | 0.777 | 0.117 | 0.360 | 4.96% | 7.7 | 25.0h |
| 123 | 0.775 | 0.116 | 0.375 | 5.11% | 7.1 | 25.0h |
| 456 | 0.776 | 0.118 | 0.380 | 4.98% | 7.6 | 26.0h |

## MAE vs Baseline Delta

| Metric | A (baseline) | B (MAE) | Delta |
|--------|-------------|---------|-------|
| AUROC | 0.814 | 0.816 | +0.002 |
| AUPRC | 0.144 | 0.137 | -0.007 |
| Precision @ 0.70 recall | 5.42% | 5.15% | -0.27% |
| Alerts/day @ 0.70 recall | 7.5 | 8.0 | +0.5 |
| Median lead @ 0.70 recall | 26.8h | 26.5h | -0.3h |

## Biggest Win

**MAE pretraining achieved the lowest AUROC variance across seeds** (0.001 vs 0.004
for scratch). This means the pretrained encoder provides a more stable initialization
-- training converges to a tighter range of solutions. In a clinical deployment where
you train once and ship, lower variance means more predictable performance.

However, this stability did not translate into better discrimination or a better
recall-precision tradeoff at the 0.70 patient recall target.

## Interpretation

1. **The AUROC ~0.81 ceiling is a data limitation, not a representation limitation.**
   MAE pretraining successfully learned temporal dynamics (val MSE dropped 72%), but
   the downstream classifier gained nothing from it. The encoder already had enough
   capacity (68K params) to learn these patterns during supervised training alone.

2. **Frozen encoder clearly fails** (AUROC 0.776). The MAE-learned representations
   are not directly useful for classification without fine-tuning -- the reconstruction
   objective learns different features than what the classification task needs.

3. **Self-supervised pretraining helps most with large models on limited labeled data.**
   Our setup (68K-param encoder, 9,839 labeled training episodes) is too small and
   too data-rich for pretraining to matter. MAE shines when you have millions of
   parameters and limited supervision.

4. **The recall-precision tradeoff is fundamentally constrained by:**
   - 2.2% sepsis prevalence at timestep level (base rate)
   - Hourly resolution (limits how early you can detect)
   - 9 features with 89-97% lab missingness
   - No features that capture sub-hour dynamics

## What This Rules Out

- Self-supervised representation learning on this dataset/architecture scale
- The hypothesis that better temporal representations = better classification
  (they're different objectives with different optima)

## What Remains to Try

- **More data**: MIMIC-IV (richer features, larger cohort, flexible cohort definitions)
- **Better features**: IL-6 (pending sensor), more granular vitals, clinical notes
- **Architecture changes**: Multi-scale attention, patient-level aggregation, longer context
- **Different self-supervised objectives**: Contrastive learning (SimCLR/TS2Vec) may align
  better with classification than reconstruction

## Files

- `mae_pretrain/mae_best.pt` -- Pretrained MAE weights (encoder + decoder)
- `mae_pretrain/mae_history.json` -- Per-epoch pretraining loss
- `mae_pretrain/mae_loss.png` -- Pretraining loss curve
- `finetune/{config}_seed{N}/` -- Per-run checkpoints and results
- `summary/all_results.json` -- Complete results for all 9 runs
- `summary/summary.json` -- Aggregated comparison
- `summary/comparison.png` -- Bar chart comparison at 0.70 recall
