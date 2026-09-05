#ifndef FEATURE_EXTRACTION_H_
#define FEATURE_EXTRACTION_H_

#include <cstddef>

constexpr std::size_t kCaptureSamples = 250U;
constexpr float kSampleRateHz = 5000.0f;
constexpr float kSamplePeriodSeconds = 1.0f / kSampleRateHz;

struct FluorescenceFeatures {
    float peak_intensity;
    float mean_intensity;
    float rise_time;
    float decay_time;
    float energy;
};

FluorescenceFeatures ExtractFluorescenceFeatures(
    const float corrected_waveform[kCaptureSamples]);

#endif
