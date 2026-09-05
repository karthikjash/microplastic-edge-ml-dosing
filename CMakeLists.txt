cmake_minimum_required(VERSION 3.22.0)

include(${SdkRootDirPath}/cmake/extension/mcux.cmake)

project(microplastic_ml LANGUAGES C CXX ASM
    PROJECT_BOARD_PORT_PATH ${board_root}/${board}/driver_examples/lpadc/polling)

option(MICROPLASTIC_TEST_MODE "Run the synthetic waveform/model test instead of ADC capture" ON)
option(MICROPLASTIC_INTERACTIVE_TEST "Read five Model A features from the serial console" ON)
option(MICROPLASTIC_DEBUG_ADC "Print ADC diagnostics" OFF)

mcux_add_include(
    INCLUDES
        source
        firmware
)

mcux_add_source(
    SOURCES
        source/main.cc
        source/feature_extraction.cc
        source/feature_extraction.h
        source/sensor_acquisition.cc
        source/sensor_acquisition.h
        source/model_inference.cc
        source/model_inference.h
        firmware/scaler_constants.h
        firmware/models/model_a_int8_data.cc
)

mcux_add_cmakelists(${SdkRootDirPath}/middleware/eiq/tensorflow-lite OPTIONAL)
mcux_add_source(
    BASE_PATH ${SdkRootDirPath}/middleware/eiq/tensorflow-lite
    SOURCES
        tensorflow/lite/micro/kernels/fully_connected.cpp
        tensorflow/lite/micro/debug_log.cpp
)

if(MICROPLASTIC_TEST_MODE)
    mcux_add_macro(CX "-DMICROPLASTIC_TEST_MODE=1")
else()
    mcux_add_macro(CX "-DMICROPLASTIC_TEST_MODE=0")
endif()

if(MICROPLASTIC_INTERACTIVE_TEST)
    mcux_add_macro(CX "-DMICROPLASTIC_INTERACTIVE_TEST=1")
else()
    mcux_add_macro(CX "-DMICROPLASTIC_INTERACTIVE_TEST=0")
endif()

if(MICROPLASTIC_DEBUG_ADC)
    mcux_add_macro(CX "-DMICROPLASTIC_DEBUG_ADC=1")
endif()

include(${SdkRootDirPath}/CMakeLists.txt)
include(${SdkRootDirPath}/${board_root}/${board}/driver_examples/lpadc/polling/reconfig.cmake OPTIONAL)

mcux_convert_binary(BINARY ${APPLICATION_BINARY_DIR}/${MCUX_SDK_PROJECT_NAME}.bin)
