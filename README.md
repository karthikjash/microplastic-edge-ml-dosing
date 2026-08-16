# Intelligent Microplastic Detection and Magnetic Separation System

Fluorescence sensing + edge ML microplastic detection and dosing-calibrated
separation system, targeting the **FRDM-MCXN236** (dual Cortex-M33, onboard
eIQ Neutron NPU).

## Project structure

microplastic_project/
├── data/                      # datasets (synthetic for now, real later)
│   └── synthetic_fluorescence_dataset.csv
├── src/                       # all source code
│   ├── generate_dataset.py    # synthetic fluorescence pulse simulator + feature extraction
│   └── train_baseline.py      # baseline model training (Linear, RF, small NN)
├── models/                    # trained model artifacts (generated)
├── results/                   # metrics, plots (generated)
├── notebooks/                 # exploratory notebooks (optional)
├── requirements.txt
└── README.md

## Setup

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Usage

# 1. Generate the synthetic dataset (5 features -> concentration_mgl)
python3 src/generate_dataset.py

# 2. Train baseline models (Linear Regression, Random Forest, small Keras NN)
python3 src/train_baseline.py

Outputs land in results/ (metrics.json, prediction plots, training curves)
and models/ (saved model files).