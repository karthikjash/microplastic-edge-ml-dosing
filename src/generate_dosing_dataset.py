"""
Synthetic Dosing Dataset Generator (Model B)
================================================
Simulates the dose-response relationship for Substance X: given a
contamination level (mg/L, the output of Model A), what is the
minimum effective dose (mg/L of Substance X) needed to treat the water?

Physical model
--------------
Coagulant/separation dose-response typically follows a saturating
relationship too - you need roughly proportionally more substance for
more contamination, but with diminishing returns at high doses (excess
substance stops helping much beyond a point), plus a small fixed
"minimum activation" dose even at very low contamination (the substance
needs some minimum presence to nucleate binding at all).

We model minimum effective dose as:

    dose = dose_min + k * (1 - exp(-alpha * contamination))

Where:
    dose_min -> minimum baseline dose regardless of contamination (mg/L)
    k        -> maximum additional dose scaling (mg/L)
    alpha    -> how quickly dose requirement saturates with contamination

Plus:
    - Measurement/process noise (real dosing isn't perfectly deterministic)
    - Small random variation in k/alpha per "batch" to mimic real-world
      variability in water chemistry, temperature, pH, etc.

Label
-----
`dose_mgl` (continuous) is the regression target - what Model B predicts.
"""

import numpy as np
import pandas as pd

RNG_SEED = 7
N_SAMPLES = 1500
CONTAMINATION_RANGE_MGL = (0.0, 100.0)  # matches Model A's output range

# Nominal dose-response parameters (single substance, "Substance X")
DOSE_MIN_NOMINAL = 2.0      # mg/L, minimum baseline dose
K_NOMINAL = 18.0            # mg/L, max additional dose scaling
ALPHA_NOMINAL = 0.035       # saturation rate

rng = np.random.default_rng(RNG_SEED)


def simulate_dose(contamination_mgl, rng=rng):
    """Simulate the minimum effective dose for a given contamination level."""
    # batch-to-batch variability in the dose-response parameters
    dose_min = max(rng.normal(DOSE_MIN_NOMINAL, 0.2), 0)
    k = max(rng.normal(K_NOMINAL, 1.5), 1.0)
    alpha = max(rng.normal(ALPHA_NOMINAL, 0.004), 0.01)

    dose = dose_min + k * (1 - np.exp(-alpha * contamination_mgl))

    # measurement/process noise
    dose += rng.normal(0, 0.4)
    dose = max(dose, 0)
    return dose


def generate_dosing_dataset(n_samples=N_SAMPLES, conc_range=CONTAMINATION_RANGE_MGL, rng=rng):
    rows = []
    for _ in range(n_samples):
        contamination = rng.uniform(*conc_range)
        dose = simulate_dose(contamination, rng=rng)
        rows.append({"contamination_mgl": contamination, "dose_mgl": dose})
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    import os

    df = generate_dosing_dataset()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_dosing_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} samples -> {out_path}")
    print(df.describe())