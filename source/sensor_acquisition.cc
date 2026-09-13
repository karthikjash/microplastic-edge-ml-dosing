#include "sensor_acquisition.h"

#include "board.h"
#include "hardware_config.h"
#include "fsl_gpio.h"
#include "fsl_debug_console.h"
#include "fsl_lpadc.h"
#include "fsl_clock.h"
#include "fsl_common.h"

#include <cmath>

namespace {
ADC_Type *const kAdcBase = ADC0;
constexpr uint32_t kAdcChannel = MICROPLASTIC_ADC_CHANNEL;
constexpr uint32_t kAdcCommandId = 1U;
constexpr uint32_t kAdcTriggerMask = 1U;
constexpr lpadc_reference_voltage_source_t kAdcReferenceSource =
    kLPADC_ReferenceVoltageAlt3;
constexpr float kAdcReferenceVolts = 3.3f;
constexpr uint32_t kAdcFullScale = 4095U;
constexpr uint32_t kSamplePeriodUs = 200U;
constexpr uint32_t kLedStabilizationUs = 1500U;

lpadc_conv_result_t g_result;
}

void SensorAcquisition_Init()
{
#if MICROPLASTIC_LED_CONTROL_ENABLED
    gpio_pin_config_t led_config = {kGPIO_DigitalOutput, 0U};
    GPIO_PinInit(MICROPLASTIC_LED_GPIO_BASE, MICROPLASTIC_LED_GPIO_PIN, &led_config);
#endif
    lpadc_config_t config;
    lpadc_conv_command_config_t command;
    lpadc_conv_trigger_config_t trigger;

    LPADC_GetDefaultConfig(&config);
    config.referenceVoltageSource = kAdcReferenceSource;
    config.enableAnalogPreliminary = true;
    LPADC_Init(kAdcBase, &config);
    LPADC_DoAutoCalibration(kAdcBase);

    LPADC_GetDefaultConvCommandConfig(&command);
    command.channelNumber = kAdcChannel;
    LPADC_SetConvCommandConfig(kAdcBase, kAdcCommandId, &command);

    LPADC_GetDefaultConvTriggerConfig(&trigger);
    trigger.targetCommandId = kAdcCommandId;
    trigger.enableHardwareTrigger = false;
    LPADC_SetConvTriggerConfig(kAdcBase, 0U, &trigger);
}

namespace {
bool ReadAdcSample(RawAdcSample *sample)
{
    LPADC_DoSoftwareTrigger(kAdcBase, kAdcTriggerMask);
#if (defined(FSL_FEATURE_LPADC_FIFO_COUNT) && (FSL_FEATURE_LPADC_FIFO_COUNT == 2U))
    while (!LPADC_GetConvResult(kAdcBase, &g_result, 0U))
#else
    while (!LPADC_GetConvResult(kAdcBase, &g_result))
#endif
    {
    }
    const uint32_t counts = (g_result.convValue >> 3U) & kAdcFullScale;
    sample->adc_counts = counts;
    sample->voltage = static_cast<float>(counts) * kAdcReferenceVolts /
                      static_cast<float>(kAdcFullScale);
    SDK_DelayAtLeastUs(kSamplePeriodUs, SystemCoreClock);
    return true;
}
}

bool CaptureBackgroundAndSignal(float corrected_waveform[kCaptureSamples])
{
    RawAdcSample background[kCaptureSamples];
    RawAdcSample signal[kCaptureSamples];
#if MICROPLASTIC_LED_CONTROL_ENABLED
    GPIO_PinWrite(MICROPLASTIC_LED_GPIO_BASE, MICROPLASTIC_LED_GPIO_PIN, 0U);
#endif
    for (std::size_t i = 0U; i < kCaptureSamples; ++i)
    {
        if (!ReadAdcSample(&background[i])) return false;
    }
#if MICROPLASTIC_LED_CONTROL_ENABLED
    GPIO_PinWrite(MICROPLASTIC_LED_GPIO_BASE, MICROPLASTIC_LED_GPIO_PIN, 1U);
    SDK_DelayAtLeastUs(kLedStabilizationUs, SystemCoreClock);
#endif
    for (std::size_t i = 0U; i < kCaptureSamples; ++i)
    {
        if (!ReadAdcSample(&signal[i])) return false;
        const float corrected = signal[i].voltage - background[i].voltage;
        corrected_waveform[i] = corrected > 0.0f ? corrected : 0.0f;
#if MICROPLASTIC_RAW_STREAM
        PRINTF("RAW,%u,%u,%u,", static_cast<unsigned int>(i),
               static_cast<unsigned int>(i * kSamplePeriodUs / 1000U),
               static_cast<unsigned int>(signal[i].adc_counts));
        PRINTF("%u.%06u,signal\r\n",
               static_cast<unsigned int>(signal[i].voltage),
               static_cast<unsigned int>(signal[i].voltage * 1000000.0f) % 1000000U);
#endif
    }
#if MICROPLASTIC_LED_CONTROL_ENABLED
    GPIO_PinWrite(MICROPLASTIC_LED_GPIO_BASE, MICROPLASTIC_LED_GPIO_PIN, 0U);
#endif
    return true;
}

void GenerateSyntheticWaveform(float waveform[kCaptureSamples])
{
    for (std::size_t i = 0U; i < kCaptureSamples; ++i)
    {
        const float t = static_cast<float>(i) / kSampleRateHz;
        const float rise = 1.0f - std::exp(-t / 0.0012f);
        const float decay = std::exp(-t / 0.010f);
        waveform[i] = 1.1f * rise * decay;
    }
}
