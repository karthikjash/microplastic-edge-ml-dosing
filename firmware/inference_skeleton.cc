/*
 * TFLM Inference Skeleton - FRDM-MCXN236
 * ==========================================
 * Loads Model A (int8, contamination detection) and runs inference on
 * a 5-feature input. STARTING POINT for the person with board access -
 * adjust include paths once integrated into the MCUXpresso SDK project.
 */

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "scaler_constants.h"

extern const unsigned char models_nn_baseline_int8_tflite[];
extern const unsigned int models_nn_baseline_int8_tflite_len;

constexpr int kTensorArenaSize = 8 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

void run_model_a_inference(float raw_features[NUM_FEATURES], float *predicted_contamination) {
    const tflite::Model* model = tflite::GetModel(models_nn_baseline_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema version mismatch!");
        return;
    }

    static tflite::MicroMutableOpResolver<3> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddDequantize();

    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed!");
        return;
    }

    TfLiteTensor* input = interpreter.input(0);
    float in_scale = input->params.scale;
    int in_zero_point = input->params.zero_point;

    float scaled_features[NUM_FEATURES];
    scale_features(raw_features, scaled_features);

    for (int i = 0; i < NUM_FEATURES; i++) {
        int8_t quantized = (int8_t)(scaled_features[i] / in_scale + in_zero_point);
        input->data.int8[i] = quantized;
    }

    if (interpreter.Invoke() != kTfLiteOk) {
        MicroPrintf("Invoke() failed!");
        return;
    }

    TfLiteTensor* output = interpreter.output(0);
    float out_scale = output->params.scale;
    int out_zero_point = output->params.zero_point;
    int8_t quantized_output = output->data.int8[0];

    *predicted_contamination = (quantized_output - out_zero_point) * out_scale;
}

/*
 * TODO for integration (person with board access):
 * 1. Wire up real ADC reads -> feature extraction (peak/mean/rise/decay/energy)
 *    matching generate_dataset.py's extract_features() logic, in C.
 * 2. Repeat this pattern for Model B (dosing_model_float32.tflite) - its
 *    input/output tensors are float32, so no manual quantization needed.
 * 3. Print result over UART/USB serial (Phase 1), swap to WiFi coprocessor
 *    communication in Phase 2.
 */