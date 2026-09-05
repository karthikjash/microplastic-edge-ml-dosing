#ifndef MODEL_INFERENCE_H_
#define MODEL_INFERENCE_H_

#include "feature_extraction.h"

struct InferenceResult {
    float scaled_features[5];
    float raw_prediction_mg_l;
    float clipped_prediction_mg_l;
    bool out_of_calibrated_range;
    bool valid;
};

bool RunModelA(const FluorescenceFeatures &features, InferenceResult *result);

#endif
