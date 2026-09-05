/*
 * Feature Scaler Constants - Model A (contamination detection)
 * =================================================================
 * Exported from models/feature_scaler.joblib (sklearn StandardScaler).
 * Apply to raw feature values BEFORE feeding into the TFLite int8
 * model: scaled_value = (raw_value - mean) / scale
 *
 * Feature order (must match training):
 *   0: peak_intensity
 *   1: mean_intensity
 *   2: rise_time
 *   3: decay_time
 *   4: energy
 */

#ifndef SCALER_CONSTANTS_H
#define SCALER_CONSTANTS_H

#define NUM_FEATURES 5

static const float FEATURE_MEAN[NUM_FEATURES] = {
    0.8124982314867875f,     // peak_intensity
    0.17934137706532383f,    // mean_intensity
    0.0016283885542168154f,  // rise_time
    0.008751506024096321f,   // decay_time
    0.009043756453990708f    // energy
};

static const float FEATURE_SCALE[NUM_FEATURES] = {
    0.3381621355655501f,     // peak_intensity
    0.08569155021944348f,    // mean_intensity
    0.0007415435121622629f,  // rise_time
    0.002045712666398357f,   // decay_time
    0.0042734623118885215f   // energy
};

static inline void scale_features(const float raw_features[NUM_FEATURES],
                                   float scaled_features[NUM_FEATURES]) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaled_features[i] = (raw_features[i] - FEATURE_MEAN[i]) / FEATURE_SCALE[i];
    }
}

#endif  // SCALER_CONSTANTS_H