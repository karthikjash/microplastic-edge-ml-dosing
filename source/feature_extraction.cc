#include "feature_extraction.h"

#include <cmath>

namespace {
constexpr float kMinimumPeak = 1.0e-6f;
constexpr float kE = 2.718281828459045f;
}

FluorescenceFeatures ExtractFluorescenceFeatures(
    const float corrected_waveform[kCaptureSamples])
{
    FluorescenceFeatures features = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::size_t peak_index = 0U;

    for (std::size_t i = 0U; i < kCaptureSamples; ++i)
    {
        const float sample = corrected_waveform[i] > 0.0f ? corrected_waveform[i] : 0.0f;
        features.mean_intensity += sample;
        if (sample > features.peak_intensity)
        {
            features.peak_intensity = sample;
            peak_index = i;
        }
    }
    features.mean_intensity /= static_cast<float>(kCaptureSamples);

    for (std::size_t i = 0U; i + 1U < kCaptureSamples; ++i)
    {
        const float left = corrected_waveform[i] > 0.0f ? corrected_waveform[i] : 0.0f;
        const float right = corrected_waveform[i + 1U] > 0.0f ? corrected_waveform[i + 1U] : 0.0f;
        features.energy += 0.5f * (left + right) * kSamplePeriodSeconds;
    }

    if (features.peak_intensity <= kMinimumPeak)
    {
        return features;
    }

    const float ten_percent = features.peak_intensity * 0.10f;
    const float ninety_percent = features.peak_intensity * 0.90f;
    std::size_t ten_index = peak_index;
    std::size_t ninety_index = peak_index;
    bool found_ten = false;
    bool found_ninety = false;

    for (std::size_t i = 0U; i <= peak_index; ++i)
    {
        const float sample = corrected_waveform[i] > 0.0f ? corrected_waveform[i] : 0.0f;
        if (!found_ten && sample >= ten_percent)
        {
            ten_index = i;
            found_ten = true;
        }
        if (!found_ninety && sample >= ninety_percent)
        {
            ninety_index = i;
            found_ninety = true;
        }
    }
    if (found_ten && found_ninety && ninety_index >= ten_index)
    {
        features.rise_time = static_cast<float>(ninety_index - ten_index) * kSamplePeriodSeconds;
    }

    const float decay_level = features.peak_intensity / kE;
    for (std::size_t i = peak_index + 1U; i < kCaptureSamples; ++i)
    {
        const float sample = corrected_waveform[i] > 0.0f ? corrected_waveform[i] : 0.0f;
        if (sample <= decay_level)
        {
            features.decay_time =
                static_cast<float>(i - peak_index) * kSamplePeriodSeconds;
            break;
        }
    }
    return features;
}
