#ifndef SENSOR_ACQUISITION_H_
#define SENSOR_ACQUISITION_H_

#include "feature_extraction.h"

#include <cstdint>

struct RawAdcSample {
    uint32_t adc_counts;
    float voltage;
};

void SensorAcquisition_Init();
bool CaptureBackgroundAndSignal(float corrected_waveform[kCaptureSamples]);
void GenerateSyntheticWaveform(float waveform[kCaptureSamples]);

#endif
