#include "model_inference.h"

#include "scaler_constants.h"
#include "fsl_debug_console.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <cmath>
#include <cstdint>

extern unsigned char models_nn_baseline_int8_tflite[];
extern unsigned int models_nn_baseline_int8_tflite_len;

namespace {
constexpr int kTensorArenaSize = 8 * 1024;
alignas(16) uint8_t g_tensor_arena[kTensorArenaSize];
bool g_initialized = false;
tflite::MicroInterpreter *g_interpreter = nullptr;
TfLiteTensor *g_input = nullptr;
TfLiteTensor *g_output = nullptr;

int8_t Quantize(float value, float scale, int zero_point)
{
    if (scale <= 0.0f || !std::isfinite(value))
    {
        return 0;
    }
    const float unbounded = std::round(value / scale) + static_cast<float>(zero_point);
    if (unbounded < -128.0f)
    {
        return -128;
    }
    if (unbounded > 127.0f)
    {
        return 127;
    }
    return static_cast<int8_t>(unbounded);
}
}

bool RunModelA(const FluorescenceFeatures &features, InferenceResult *result)
{
    const float raw[NUM_FEATURES] = {
        features.peak_intensity, features.mean_intensity, features.rise_time,
        features.decay_time, features.energy};
    float scaled[NUM_FEATURES];
    scale_features(raw, scaled);
    PRINTF("TEST: scaler completed\r\n");
    for (int i = 0; i < NUM_FEATURES; ++i)
    {
        result->scaled_features[i] = scaled[i];
    }

    if (!g_initialized)
    {
        PRINTF("TEST: model data lookup started\r\n");
        const tflite::Model *model = tflite::GetModel(models_nn_baseline_int8_tflite);
        PRINTF("TEST: model data lookup completed\r\n");
        if (model == nullptr || model->version() != TFLITE_SCHEMA_VERSION)
        {
            PRINTF("TEST: model schema validation failed\r\n");
            return false;
        }
        PRINTF("TEST: model schema validated\r\n");
        static tflite::MicroMutableOpResolver<3> resolver;
        PRINTF("TEST: operator registration started\r\n");
        resolver.AddFullyConnected();
        resolver.AddRelu();
        resolver.AddDequantize();
        PRINTF("TEST: operator registration completed\r\n");
        static tflite::MicroInterpreter interpreter(
            model, resolver, g_tensor_arena, kTensorArenaSize);
        PRINTF("TEST: tensor allocation started (arena=%d)\r\n", kTensorArenaSize);
        if (interpreter.AllocateTensors() != kTfLiteOk)
        {
            PRINTF("TEST: tensor allocation failed\r\n");
            return false;
        }
        PRINTF("TEST: tensor allocation completed\r\n");
        g_interpreter = &interpreter;
        g_input = interpreter.input(0);
        g_output = interpreter.output(0);
        g_initialized = true;
        PRINTF("TEST: model initialization completed\r\n");
    }

    if (g_input->type != kTfLiteInt8 || g_output->type != kTfLiteInt8 ||
        g_input->dims->size != 2 || g_input->dims->data[0] != 1 ||
        g_input->dims->data[1] != NUM_FEATURES)
    {
        PRINTF("TEST: model tensor validation failed\r\n");
        return false;
    }
    for (int i = 0; i < NUM_FEATURES; ++i)
    {
        g_input->data.int8[i] =
            Quantize(scaled[i], g_input->params.scale, g_input->params.zero_point);
    }
    PRINTF("TEST: inference started\r\n");
    if (g_interpreter->Invoke() != kTfLiteOk)
    {
        PRINTF("TEST: inference failed\r\n");
        return false;
    }
    PRINTF("TEST: inference completed\r\n");

    result->raw_prediction_mg_l =
        (static_cast<float>(g_output->data.int8[0]) -
         static_cast<float>(g_output->params.zero_point)) *
        g_output->params.scale;
    result->out_of_calibrated_range =
        result->raw_prediction_mg_l < 0.0f || result->raw_prediction_mg_l > 100.0f;
    result->clipped_prediction_mg_l =
        result->raw_prediction_mg_l < 0.0f
            ? 0.0f
            : (result->raw_prediction_mg_l > 100.0f ? 100.0f
                                                     : result->raw_prediction_mg_l);
    result->valid = std::isfinite(result->raw_prediction_mg_l);
    return result->valid;
}
