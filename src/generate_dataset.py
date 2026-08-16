"""
Synthetic Fluorescence Dataset Generator
==========================================
Simulates Nile-Red fluorescence photodiode response curves for
microplastic concentration sensing, and extracts the 5 features
used by the downstream ML model:

    - peak_intensity
    - mean_intensity
    - rise_time
    - decay_time
    - energy (area under curve)

Physical model
--------------
When the blue LED excites the Nile-Red-stained sample, the photodiode
signal roughly follows a fast-rise / exponential-decay pulse shape,
similar to a fluorescence lifetime response convolved with the
LED's turn-on ramp:

    I(t) = A * (1 - exp(-t / tau_rise)) * exp(-t / tau_decay)

Where:
    A          -> scales with microplastic concentration (+ noise)
    tau_rise   -> rise time constant (mostly sensor/optical, small variance)
    tau_decay  -> decay time constant (varies with polymer type / dye binding)

We add:
    - Gaussian sensor/electronic noise
    - Baseline offset drift (ambient light / dark current)
    - Small random jitter in tau_rise / tau_decay to mimic sample-to-sample
      variability and different polymer types

Label
-----
`concentration` (mg/L, continuous) is the regression target.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_SAMPLES = 2000          # total synthetic samples
FS_HZ = 5000              # sampling rate of the "photodiode" (Hz)
WINDOW_MS = 50            # total capture window per pulse (ms)
CONC_RANGE_MGL = (0.0, 100.0)  # microplastic concentration range to simulate

rng = np.random.default_rng(RNG_SEED)


def simulate_pulse(concentration_mgl, fs_hz=FS_HZ, window_ms=WINDOW_MS, rng=rng):
    """Simulate one fluorescence pulse for a given true concentration."""
    n_points = int(fs_hz * window_ms / 1000)
    t = np.linspace(0, window_ms / 1000, n_points)  # seconds

    # --- concentration -> peak amplitude (roughly linear + saturation) ---
    # Real fluorescence often saturates at high concentration (self-quenching),
    # so use a soft-saturation (Michaelis-Menten-like) curve instead of pure linear.
    k_sat = 60.0  # half-saturation constant (mg/L)
    A_max = 3.3   # max amplitude (V), matches a 3.3V ADC reference
    A = A_max * (concentration_mgl / (concentration_mgl + k_sat))
    A += rng.normal(0, 0.01)  # small amplitude jitter (LED drive/optical variance)
    A = max(A, 0)

    # --- time constants (sample-to-sample / polymer-type jitter) ---
    tau_rise = max(rng.normal(0.0015, 0.0002), 0.0003)   # ~1.5 ms nominal
    tau_decay = max(rng.normal(0.008, 0.0015), 0.002)    # ~8 ms nominal

    signal = A * (1 - np.exp(-t / tau_rise)) * np.exp(-t / tau_decay)

    # --- baseline offset (ambient light / dark current) ---
    baseline = rng.normal(0.05, 0.005)
    signal = signal + baseline

    # --- electronic/sensor noise ---
    noise_std = 0.01 + 0.002 * rng.random()  # slight variance in noise floor
    signal = signal + rng.normal(0, noise_std, size=signal.shape)

    signal = np.clip(signal, 0, 3.3)  # ADC rail clipping
    return t, signal, baseline


def extract_features(t, signal, baseline):
    """Extract the 5 features from a raw pulse waveform."""
    corrected = signal - baseline

    peak_intensity = float(np.max(corrected))
    mean_intensity = float(np.mean(corrected))
    peak_idx = int(np.argmax(corrected))

    # Rise time: 10%-90% of peak, before the peak index
    peak_val = corrected[peak_idx]
    rising = corrected[: peak_idx + 1]
    if peak_val > 0 and len(rising) > 1:
        t10 = np.searchsorted(rising, 0.1 * peak_val)
        t90 = np.searchsorted(rising, 0.9 * peak_val)
        rise_time = float(t[min(t90, len(t) - 1)] - t[min(t10, len(t) - 1)])
    else:
        rise_time = 0.0
    rise_time = max(rise_time, 0.0)

    # Decay time: time from peak to 1/e of peak value (after peak index)
    falling = corrected[peak_idx:]
    if peak_val > 0 and len(falling) > 1:
        target = peak_val / np.e
        below = np.where(falling <= target)[0]
        if len(below) > 0:
            decay_idx = peak_idx + below[0]
            decay_time = float(t[decay_idx] - t[peak_idx])
        else:
            decay_time = float(t[-1] - t[peak_idx])
    else:
        decay_time = 0.0
    decay_time = max(decay_time, 0.0)

    # Energy: area under curve (trapezoidal integration)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    energy = float(trapz_fn(np.clip(corrected, 0, None), t))

    return {
        "peak_intensity": peak_intensity,
        "mean_intensity": mean_intensity,
        "rise_time": rise_time,
        "decay_time": decay_time,
        "energy": energy,
    }


def generate_dataset(n_samples=N_SAMPLES, conc_range=CONC_RANGE_MGL, rng=rng):
    rows = []
    for _ in range(n_samples):
        concentration = rng.uniform(*conc_range)
        t, signal, baseline = simulate_pulse(concentration, rng=rng)
        feats = extract_features(t, signal, baseline)
        feats["concentration_mgl"] = concentration
        rows.append(feats)
    df = pd.DataFrame(rows)
    # reorder columns: features first, label last
    cols = ["peak_intensity", "mean_intensity", "rise_time", "decay_time", "energy", "concentration_mgl"]
    return df[cols]


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "synthetic_fluorescence_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} samples -> {out_path}")
    print(df.describe())