# SepSentinel

A wearable multimodal platform for early sepsis detection using continuous biosensor data and machine learning.

## What is SepSentinel?

SepSentinel is a wearable patch system that monitors physiological signals and measures interstitial fluid biomarkers in situ via microneedle-integrated electrochemical sensors. No fluid is extracted or transported — the sensors contact ISF directly within the skin. This repository contains the software prototype: data pipeline, ML models, and monitoring dashboard.

### Hardware Concept

```
Microneedle Patch (inserted in skin)        Physiological Sensors
         |                                         |
Electrochemical sensors measure               HR, RR, Temp, SpO2
analytes in ISF (no fluid extraction)              |
         |                                         |
  pH     (potentiometric — TBD)                    |
  Lactate (amperometric — TBD)                     |
  IL-6   (E-AB, three-electrode, SWV)              |
         |                                         |
         +---------- Potentiostat -----------------+
                         |
                  Analog Front End / ADC
                         |
                   Microcontroller
                (signal processing + calibration)
                         |
                      Bluetooth
                         |
                       Phone
                         |
                   ML Inference
                         |
                 Sepsis Risk Score
```

The IL-6 sensor uses electrochemical aptamer-based (E-AB) sensing with a three-electrode system (WE/RE/CE) in direct ISF contact. Lactate and pH sensor hardware remains modular; likely electrochemical but final implementations may differ. The software pipeline receives processed biomarker values regardless of sensing modality.

## Input Signals

7 features measured approximately every 5 minutes:

| Signal | Unit | Normal Range | Source |
|--------|------|-------------|--------|
| Heart Rate | bpm | 60-100 | PPG sensor |
| Respiratory Rate | breaths/min | 12-20 | Impedance/accelerometer |
| Temperature | C | 36.1-37.2 | Thermistor |
| SpO2 | % | 95-100 | Pulse oximetry |
| pH | pH units | 7.35-7.45 | ISF, potentiometric (TBD) |
| Lactate | mmol/L | 0.5-2.0 | ISF, amperometric (TBD) |
| IL-6 | pg/mL | 0-7 | ISF, E-AB sensor (SWV) |

## Model Architecture

The system supports multiple interchangeable model architectures through a shared `SepsisModel` interface:

| Model | Type | Input | Status |
|-------|------|-------|--------|
| Random Forest | Flat baseline | Flattened feature vector | Implemented |
| XGBoost | Flat baseline | Flattened feature vector | Module 7 |
| TCN | Sequential | (batch, timesteps, 7) | Module 7 |
| Transformer | Sequential | (batch, timesteps, 7) | Module 7 |

Sequential models use a rolling history window (configurable: 30min to 4hrs at 5-min intervals) to capture temporal patterns in the signals.

The encoder is separated from the prediction head, enabling future dual-branch architectures:
- Physiological branch (HR, RR, Temp, SpO2) with its own encoder
- Biomarker branch (pH, Lactate, IL-6) with its own encoder
- Feature fusion before the final prediction

### Output

- Probability of sepsis: 0.0 to 1.0
- Risk score: 0-100%
- Risk category: Low / Medium / High

## Project Structure

```
sepsentinel/
    config/
        signals.py            # 7 signal definitions (physiological + electrochemical)
        thresholds.py         # Alert thresholds (WARNING / CRITICAL)
    data/
        synthetic.py          # Synthetic data generator
        sequences.py          # Sliding window / tensor construction (Module 6)
        preprocessing.py      # Normalization, imputation (Module 6)
        mimic.py              # MIMIC-IV loader (Module 9)
    models/
        base.py               # SepsisModel ABC + SequenceEncoder ABC
        registry.py           # Model factory
        random_forest.py      # RF baseline
        xgboost_model.py      # XGBoost baseline (Module 7)
        tcn.py                # Temporal Convolutional Network (Module 7)
        transformer.py        # Transformer encoder (Module 7)
        evaluation.py         # Metrics and comparison (Module 7)
    dashboard/
        app.py                # Streamlit web dashboard
        components.py         # Reusable UI components
    hardware/
        bluetooth.py          # BLE data reception (Module 10)
    alerts.py                 # Alert checking logic
    visualization.py          # Matplotlib plots
    simulation.py             # 7-signal patient simulation
main.py                       # CLI entry point
app.py                        # Streamlit Cloud entry point
data/                         # Datasets
models/                       # Saved model artifacts
results/                      # Evaluation plots
```

## How to Run

### Requirements

- Python 3.10+
- `pip install -r requirements.txt`

### Quick Start

```bash
pip install -r requirements.txt
python main.py
```

### Menu Options

1. **Simulate a worsening patient** - 7-signal simulation with plots
2. **Enter signal values manually** - type values, get risk score
3. **Train Random Forest** - train on synthetic 7-feature data
4. **Launch Dashboard** - opens Streamlit dashboard
5. **Exit**

### Running the Dashboard Directly

```bash
streamlit run sepsentinel/dashboard/app.py
```

## Alert Thresholds

| Signal | Warning | Critical |
|--------|---------|----------|
| Heart Rate | >100 or <50 bpm | >120 or <40 bpm |
| Respiratory Rate | >22 or <10 br/min | >30 or <8 br/min |
| Temperature | >38.0 or <35.5 C | >39.0 or <35.0 C |
| SpO2 | <94% | <90% |
| pH | <=7.35 | <=7.25 |
| Lactate | >=2.0 mmol/L | >=4.0 mmol/L |
| IL-6 | >=7 pg/mL | >=50 pg/mL |
| Risk Score | >=30% | >=60% |

## Data Strategy

**Current**: Synthetic data for pipeline development and architecture testing. Not for clinical validation.

**Target**: MIMIC-IV Clinical Database and MIMIC-IV Waveform Database (pending PhysioNet credentialed access). The pipeline is designed so synthetic data can be swapped for MIMIC data with minimal code changes.

See [DATASETS.md](DATASETS.md) for details.

## Roadmap

### Completed
- **v1.0-v1.1** - Proof of concept (3 biomarkers, RF, Streamlit dashboard)
- **Module 5** - Architecture refactor (7 signals, model interfaces, new package structure)

### Planned
- **Module 6** - Time-series data pipeline (episode generator, sliding windows, preprocessing)
- **Module 7** - Model implementation (XGBoost, TCN, Transformer) and comparison
- **Module 8** - Dashboard v2 (live temporal plots, model selector, alert history)
- **Module 9** - MIMIC-IV integration
- **Module 10** - Hardware integration (Bluetooth, real-time inference, multi-patient)

## References

- SepAI: "SepAl: Sepsis Alerts on Low Power Wearables With Digital Biomarkers and On-Device Tiny Machine Learning" - temporal learning and feature fusion architecture
- MIMIC-IV: Johnson et al., PhysioNet
