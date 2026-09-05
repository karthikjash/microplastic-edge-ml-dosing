#include "board.h"
#include "fsl_debug_console.h"

#include "feature_extraction.h"
#include "model_inference.h"
#include "sensor_acquisition.h"

#include <cstdio>
#include <cstdint>
#include <cmath>

extern unsigned int models_nn_baseline_int8_tflite_len;
extern "C" void BOARD_InitHardware(void);

namespace {
void PrintFixed(const char *label, float value)
{
    int32_t scaled = static_cast<int32_t>(value * 10000.0f);
    const bool negative = scaled < 0;
    if (negative)
    {
        scaled = -scaled;
    }
    const int32_t integer = scaled / 10000;
    const int32_t fraction = scaled % 10000;
    PRINTF("%s%s%d.", label, negative ? "-" : "", static_cast<int>(integer));
    if (fraction < 1000)
    {
        PRINTF("0");
    }
    if (fraction < 100)
    {
        PRINTF("0");
    }
    if (fraction < 10)
    {
        PRINTF("0");
    }
    PRINTF("%d\r\n", static_cast<int>(fraction));
}

void ReportFault(const char *name)
{
    PRINTF("\r\nFAULT: %s\r\n", name);
    PRINTF("FAULT: CFSR=0x%x HFSR=0x%x DFSR=0x%x\r\n",
           static_cast<unsigned int>(SCB->CFSR),
           static_cast<unsigned int>(SCB->HFSR),
           static_cast<unsigned int>(SCB->DFSR));
    PRINTF("FAULT: BFAR=0x%x MMFAR=0x%x\r\n",
           static_cast<unsigned int>(SCB->BFAR),
           static_cast<unsigned int>(SCB->MMFAR));
    while (true)
    {
    }
}
}

extern "C" void HardFault_Handler(void)
{
    ReportFault("HardFault");
}

extern "C" void MemManage_Handler(void)
{
    ReportFault("MemManage");
}

extern "C" void BusFault_Handler(void)
{
    ReportFault("BusFault");
}

extern "C" void UsageFault_Handler(void)
{
    ReportFault("UsageFault");
}

#ifndef MICROPLASTIC_TEST_MODE
#define MICROPLASTIC_TEST_MODE 1
#endif

#ifndef MICROPLASTIC_INTERACTIVE_TEST
#define MICROPLASTIC_INTERACTIVE_TEST 1
#endif

static bool ParseFeature(const char **cursor, float *value)
{
    const char *p = *cursor;
    while (*p == ' ' || *p == '\t')
    {
        ++p;
    }
    bool negative = false;
    if (*p == '+' || *p == '-')
    {
        negative = *p == '-';
        ++p;
    }
    bool has_digit = false;
    float result = 0.0f;
    while (*p >= '0' && *p <= '9')
    {
        has_digit = true;
        result = result * 10.0f + static_cast<float>(*p - '0');
        ++p;
    }
    if (*p == '.')
    {
        ++p;
        float place = 0.1f;
        while (*p >= '0' && *p <= '9')
        {
            has_digit = true;
            result += static_cast<float>(*p - '0') * place;
            place *= 0.1f;
            ++p;
        }
    }
    if (!has_digit || !std::isfinite(result))
    {
        return false;
    }
    *value = negative ? -result : result;
    *cursor = p;
    return true;
}

static bool ReadInteractiveFeatures(FluorescenceFeatures *features)
{
    char line[128];
    std::size_t length = 0U;
    while (true)
    {
        const int ch = GETCHAR();
        if (ch == '\n' && length == 0U)
        {
            continue;
        }
        if (ch == '\r' || ch == '\n')
        {
            if (ch == '\r')
            {
                PUTCHAR('\r');
                PUTCHAR('\n');
            }
            break;
        }
        if (length + 1U < sizeof(line))
        {
            line[length++] = static_cast<char>(ch);
            PUTCHAR(ch);
        }
    }
    line[length] = '\0';
    const char *cursor = line;
    float values[5];
    for (int i = 0; i < 5; ++i)
    {
        if (!ParseFeature(&cursor, &values[i]))
        {
            return false;
        }
    }
    while (*cursor == ' ' || *cursor == '\t')
    {
        ++cursor;
    }
    if (*cursor != '\0')
    {
        return false;
    }
    features->peak_intensity = values[0];
    features->mean_intensity = values[1];
    features->rise_time = values[2];
    features->decay_time = values[3];
    features->energy = values[4];
    return true;
}

#if !MICROPLASTIC_INTERACTIVE_TEST
static void PrintResult(const FluorescenceFeatures &features,
                        const InferenceResult &result)
{
    PRINTF("\r\n========================================\r\n");
    PRINTF("MICROPLASTIC ML DETECTION\r\n");
    PRINTF("========================================\r\n");
    PrintFixed("Peak       : ", features.peak_intensity);
    PrintFixed("Mean       : ", features.mean_intensity);
    PrintFixed("Rise time  : ", features.rise_time);
    PrintFixed("Decay time : ", features.decay_time);
    PrintFixed("Energy     : ", features.energy);
    PRINTF("\r\nScaled features:\r\n");
    PrintFixed("[0] ", result.scaled_features[0]);
    PrintFixed("[1] ", result.scaled_features[1]);
    PrintFixed("[2] ", result.scaled_features[2]);
    PrintFixed("[3] ", result.scaled_features[3]);
    PrintFixed("[4] ", result.scaled_features[4]);
    PRINTF("\r\nRaw ML prediction:\r\n");
    PrintFixed("", result.raw_prediction_mg_l);

    PRINTF("\r\nContamination:\r\n");
    PrintFixed("", result.clipped_prediction_mg_l);
    PRINTF("Range status:\r\n%s\r\n",
           result.out_of_calibrated_range ? "OUT_OF_CALIBRATED_RANGE" : "IN_RANGE");
    PRINTF("========================================\r\n");
}
#endif

int main(void)
{
    BOARD_InitHardware();
    PRINTF("\r\nFRDM-MCXN236 Microplastic ML\r\n");
    PRINTF("Single Cortex-M33 CPU inference; no NPU is used.\r\n");
    PRINTF("Model A: nn_baseline_int8.tflite (%u bytes)\r\n",
           models_nn_baseline_int8_tflite_len);

    SensorAcquisition_Init();
#if MICROPLASTIC_INTERACTIVE_TEST
    PRINTF("\r\n========================================\r\n");
    PRINTF("MODEL A INTERACTIVE TEST\r\n");
    PRINTF("========================================\r\n");
    PRINTF("Enter 5 features:\r\n");
    PRINTF("peak mean rise_time decay_time energy\r\n");
    while (true)
    {
        PRINTF("> ");
        FluorescenceFeatures features = {};
        if (!ReadInteractiveFeatures(&features))
        {
            PRINTF("\r\nERROR: enter exactly five numeric features\r\n");
            continue;
        }
        InferenceResult result = {};
        if (!RunModelA(features, &result))
        {
            PRINTF("ERROR: Model A initialization/inference failed\r\n");
            continue;
        }
        PRINTF("\r\nFeatures:\r\n");
        PrintFixed("Peak       : ", features.peak_intensity);
        PrintFixed("Mean       : ", features.mean_intensity);
        PrintFixed("Rise time  : ", features.rise_time);
        PrintFixed("Decay time : ", features.decay_time);
        PrintFixed("Energy     : ", features.energy);
        PRINTF("\r\nScaled:\r\n");
        PrintFixed("[0] ", result.scaled_features[0]);
        PrintFixed("[1] ", result.scaled_features[1]);
        PrintFixed("[2] ", result.scaled_features[2]);
        PrintFixed("[3] ", result.scaled_features[3]);
        PrintFixed("[4] ", result.scaled_features[4]);
        PRINTF("\r\nPrediction : ");
        PrintFixed("", result.clipped_prediction_mg_l);
        PRINTF("Status     : %s\r\n\r\n", result.out_of_calibrated_range
                                             ? "OUT_OF_RANGE"
                                             : "IN_RANGE");
        PRINTF("Enter 5 features:\r\n");
    }
#else
    static float waveform[kCaptureSamples];
#if MICROPLASTIC_TEST_MODE
    PRINTF("TEST MODE: synthetic fluorescence waveform\r\n");
    PRINTF("TEST: waveform generation started\r\n");
    GenerateSyntheticWaveform(waveform);
    PRINTF("TEST: waveform generated\r\n");
#else
    PRINTF("REAL SENSOR MODE: ADC0_A2, J8 pin 12\r\n");
    if (!CaptureBackgroundAndSignal(waveform))
    {
        PRINTF("ERROR: ADC capture failed\r\n");
        return 1;
    }
#endif

    PRINTF("TEST: feature extraction started\r\n");
    const FluorescenceFeatures features = ExtractFluorescenceFeatures(waveform);
    PRINTF("TEST: feature extraction completed\r\n");
    PRINTF("TEST: features\r\n");
    PrintFixed("TEST: peak=", features.peak_intensity);
    PrintFixed("TEST: mean=", features.mean_intensity);
    PrintFixed("TEST: rise=", features.rise_time);
    PrintFixed("TEST: decay=", features.decay_time);
    PrintFixed("TEST: energy=", features.energy);
    InferenceResult result = {};
    PRINTF("TEST: scaler started\r\n");
    PRINTF("TEST: model initialization started\r\n");
    if (!RunModelA(features, &result))
    {
        PRINTF("ERROR: Model A initialization/inference failed\r\n");
        return 1;
    }
    PrintResult(features, result);
    while (true)
    {
    }
#endif
}
