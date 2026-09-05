#ifndef SENSOR_ACQUISITION_H_
#define SENSOR_ACQUISITION_H_

#include "feature_extraction.h"

// The LED GPIO is intentionally not guessed. Set this abstraction to the
// project-specific GPIO after the optical assembly pin is assigned.
constexpr bool kLedControlAvailable = false;

void SensorAcquisition_Init();
bool CaptureBackgroundAndSignal(float corrected_waveform[kCaptureSamples]);
void GenerateSyntheticWaveform(float waveform[kCaptureSamples]);

#endif
