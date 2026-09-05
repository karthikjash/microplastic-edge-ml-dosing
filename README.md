# FRDM-MCXN236 microplastic fluorescence ML firmware

This firmware implements the reduced scope: a 525 nm excitation experiment,
LED-off/LED-on background subtraction, five fluorescence features, StandardScaler
preprocessing, and Model A (`models/nn_baseline_int8.tflite`) using TensorFlow
Lite Micro on the single Cortex-M33 CPU. It does not implement separation,
dosing, networking, display, or any actuator.

## Hardware

* Target: FRDM-MCXN236 / MCXN236.
* ADC: `ADC0_A2`, board J8 pin 12, configured by the SDK LPADC polling example.
* ADC conversion: the SDK's configured LPADC reference is used; the current
  application converts the 12-bit result to volts using 3.3 V as the board
  reference assumption. Confirm the fitted board reference before quantitative
  calibration.
* LED: approximately 525 nm green LED with a project-specific GPIO still
  pending. The firmware keeps LED control behind an abstraction and does not
  guess a GPIO.
* Sensor: BPW34 and MCP6002 output must stay within the ADC input range.
* Serial: onboard MCU-Link VCOM, 115200 8-N-1, no flow control.

The ML data is synthetic and is not a Rhodamine-B or real-sensor calibration.

## Python ML pipeline

Create a virtual environment and install the training dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The `src/` directory contains dataset generation, baseline training, real-data
fine-tuning, TFLite conversion, dosing, and validation scripts. Generated
models are written to `models/`, and metrics and reports are written to
`results/`.

## Build

From the repository root, using the installed MCUXpresso SDK:

```powershell
cmake -S . -B build -G Ninja `
  -DSdkRootDirPath=C:\NXP\mcuxsdk\mcuxsdk `
  -Dboard=frdmmcxn236 `
  -DCMAKE_BUILD_TYPE=debug
cmake --build build
```

`MICROPLASTIC_TEST_MODE` defaults to `ON`. Set
`-DMICROPLASTIC_TEST_MODE=OFF` for ADC capture. The current 5 kHz pacing is
software-paced polling (250 samples, 200 us target interval); a timer-triggered
LPADC configuration remains pending if cycle-level timing is required.

## Test mode and output

Test mode generates a fluorescence-like waveform and runs the complete
waveform -> features -> scaler -> int8 Model A path. Output is labelled
`TEST MODE`; it is not real sensor validation. Real mode performs 250 LED-off
samples followed by 250 LED-on samples and clips negative subtraction to zero.

The selected model's input/output types and quantization parameters are read
from its tensors at runtime. The tensor arena is currently 8 KiB and the build
must validate whether that is sufficient.

## Flash

Use the MCUXpresso for VS Code LinkServer runner for `frdmmcxn236` (onboard
MCU-Link). A board/LinkServer connection is not available in this workspace, so
flash and terminal output remain pending.

## Measurements pending

Flash/RAM usage, actual tensor arena high-water mark, inference latency, ADC
reference confirmation, LED GPIO assignment, and real BPW34/Rhodamine-B
validation require the physical board and circuit.
