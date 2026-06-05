# SepSentinel

A prototype for non-invasive early sepsis detection using simulated wearable biosensor data.

## What is SepSentinel?

SepSentinel is a concept for a wearable patch that monitors biomarkers in real time to detect sepsis early. This repository contains the software prototype that demonstrates how biomarker data flows from sensors to risk scoring.

## Prototype 2 - Module 1 (Current)

Module 1 builds the foundation:

- **Biomarker definitions** for Lactate, IL-6, and pH
- **Simulated sensor data** showing a worsening patient over 60 minutes
- **Visualization** of biomarker trends using matplotlib
- **Dummy risk scoring** function (0-100%) as a placeholder for future ML

## Project Structure

```
sepsentinel/
    __init__.py            # Makes this a Python package
    biomarkers.py          # Biomarker definitions and metadata
    sensor_simulation.py   # Simulated patient data generation
    risk_model.py          # Sepsis risk score calculation
    visualization.py       # Matplotlib plotting functions
    dashboard.py           # Placeholder for future dashboard
main.py                    # Entry point - run this file
README.md                  # This file
```

## How to Run

### Requirements

- Python 3.10+
- matplotlib (`pip install matplotlib`)

### Running in PyCharm

1. Open this project in PyCharm
2. Make sure matplotlib is installed in your virtual environment
3. Right-click `main.py` and select **Run 'main'**

### Running from Terminal

```bash
pip install matplotlib
python main.py
```

## Biomarkers

| Biomarker | Unit    | Normal Range  | Role in Sepsis                  |
|-----------|---------|---------------|---------------------------------|
| Lactate   | mmol/L  | 0.5 - 2.0    | Tissue oxygen / perfusion       |
| IL-6      | pg/mL   | 0 - 7         | Immune response / inflammation  |
| pH        | pH units| 7.35 - 7.45   | Acid-base balance               |

## Roadmap

- **Module 1** - Foundation (biomarkers, simulation, visualization, dummy scoring)
- **Module 2** - Machine learning risk model trained on synthetic data
- **Module 3** - Real-time dashboard with alerts
- **Module 4** - Multi-patient tracking and database integration
