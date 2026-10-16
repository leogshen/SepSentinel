# Longer target MAE pilot

Two sequential seed-42 runs coexist with the supervised training queue on one GPU:
target-only MAE and source-initialized target MAE. This extends **CinC target
pretraining**, not the original MIMIC/SIC source pretraining. All 34 measurements
and the existing 72-hour causal masking/reconstruction pipeline are used.

Maximum 20,000 optimizer updates, minimum 5,000, batch 32, AdamW learning rate
0.0001 (same as the earlier 1,000-update MAE). Decoder, sampling, and masking seeds
match across arms. Validation every 500 updates uses the same 512 sampled tuning
prefixes and masks each time; stop after eight checks without improvement.
Select the lowest validation patient/channel-balanced Huber loss, including step
zero. Save recovery state every 500 steps, including optimizer and RNG states.
No calibration or test data are loaded. All available fit-prefix times are used,
as in the earlier target MAE. This is unsupervised representation learning and
does not establish downstream sepsis improvement without matched fine-tuning.

The process uses two CPU threads and caps PyTorch GPU allocation at 30% of VRAM.
This does not cap CUDA context overhead or guarantee a throughput improvement.
Model checkpoints are local and do not replace existing experimental outputs.

Run from the repository root:
`..\.conda\python.exe -u scripts/run_long_target_mae.py`

This is a one-seed convergence pilot; multiple seeds and matched downstream
fine-tuning remain necessary before claiming a benefit from longer MAE.
