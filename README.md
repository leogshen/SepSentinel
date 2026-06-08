# SepSentinel

A prototype for non-invasive early sepsis detection using simulated wearable biosensor data.

## What is SepSentinel?

SepSentinel is a concept for a wearable patch that monitors biomarkers in real time to detect sepsis early. This repository contains the software prototype that demonstrates how biomarker data flows from sensors to risk scoring.

## Current Status: Module 3

### Module 1 - Foundation
- Biomarker definitions for Lactate, IL-6, and pH
- Simulated sensor data showing a worsening patient over 60 minutes
- Visualization of biomarker trends using matplotlib
- Rule-based risk scoring function (0-100%)

### Module 2 - Machine Learning
- Synthetic dataset generator (500 labeled patient records)
- Logistic regression ML model (99% accuracy on synthetic data)
- Model persistence (save/load trained models)
- Terminal menu with manual biomarker input
- Risk score gauge visualization

### Module 3 - Dashboard and Alerts
- Streamlit web dashboard with two modes:
  - **Manual Input**: sliders for Lactate, IL-6, pH with live risk scoring
  - **Patient Simulation**: auto-generated worsening patient with trend charts
- Alert system with WARNING and CRITICAL levels per biomarker
- Color-coded risk gauge and status indicators
- Raw data table viewer in simulation mode

## Project Structure

```
sepsentinel/
    __init__.py            # Makes this a Python package
    biomarkers.py          # Biomarker definitions and metadata
    sensor_simulation.py   # Simulated patient data generation
    risk_model.py          # ML-based sepsis risk score (with rule-based fallback)
    visualization.py       # Matplotlib plotting functions and risk gauge
    alerts.py              # Alert system with WARNING/CRITICAL thresholds
    dashboard.py           # Streamlit web dashboard
    data_generator.py      # Synthetic training data generator
main.py                    # Entry point - run this file
data/                      # Generated training datasets
models/                    # Trained ML model files
README.md                  # This file
```

## How to Run

### Requirements

- Python 3.10+
- matplotlib (`pip install matplotlib`)
- scikit-learn (`pip install scikit-learn`)
- pandas (`pip install pandas`)
- streamlit (`pip install streamlit`)

### Quick Start

```bash
pip install matplotlib scikit-learn pandas streamlit
python main.py
```

### Menu Options

1. **Simulate a worsening patient** - auto-generated data with plots
2. **Enter biomarker values manually** - type in values, get risk score
3. **Train the ML model** - generate synthetic data and train
4. **Launch Dashboard** - opens Streamlit dashboard in your browser
5. **Exit**

### Running the Dashboard Directly

```bash
streamlit run sepsentinel/dashboard.py
```

## Biomarkers

| Biomarker | Unit    | Normal Range  | Role in Sepsis                  |
|-----------|---------|---------------|---------------------------------|
| Lactate   | mmol/L  | 0.5 - 2.0    | Tissue oxygen / perfusion       |
| IL-6      | pg/mL   | 0 - 7         | Immune response / inflammation  |
| pH        | pH units| 7.35 - 7.45   | Acid-base balance               |

## Alert Thresholds

| Biomarker | Warning       | Critical      |
|-----------|---------------|---------------|
| Lactate   | >= 2.0 mmol/L | >= 4.0 mmol/L |
| IL-6      | >= 7 pg/mL    | >= 50 pg/mL   |
| pH        | <= 7.35       | <= 7.25       |
| Risk Score| >= 30%        | >= 60%        |

## Roadmap

- **Module 1** - Foundation (biomarkers, simulation, visualization, dummy scoring)
- **Module 2** - Machine learning risk model trained on synthetic data
- **Module 3** - Real-time dashboard with alerts (current)
- **Module 4** - Multi-patient tracking and database integration
