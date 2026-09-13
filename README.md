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
### TIA output-range and feedback design

The analog front end must be designed for the ADC, rather than treating the ADC range as the TIA specification. For the present 3.3 V single-ended ADC, use this initial design envelope:

* dark/offset output: **0.10 V nominal** (allow 0.05-0.20 V after trimming);
* maximum fluorescence output: **2.90 V**;
* absolute output limit: **0.0-3.3 V** (never rely on the op-amp reaching either rail).

This gives approximately 2.8 V of usable signal swing and about 0.806 mV per 12-bit ADC count at a 3.3 V reference. The envelope is provisional until the maximum BPW34 photocurrent is measured with the intended LED, optics, and sample. The firmware currently assumes a 3.3 V conversion reference; confirm the actual LPADC reference before final calibration.

For a transimpedance stage, select the feedback resistor from the measured or worst-case peak photocurrent:

```text
Rf <= (Vout_max - Vdark) / Iphoto_peak
Vout = Vdark + Iphoto * Rf
```

With the design envelope above, the available signal swing is 2.8 V. The 10 kOhm feedback resistor shown in the hardware integration document therefore supports up to approximately 280 uA peak current, but may be too insensitive if the actual signal is only a few uA. For example, if the measured peak is 100 uA, a suitable first-pass value is **27 kOhm**:

```text
0.10 V + (100 uA * 27 kOhm) = 2.80 V
```

The feedback capacitor, Cf, is for TIA stability and noise control; it is not chosen from the ADC resolution. First estimate the total inverting-node capacitance, `Cin = BPW34 capacitance + op-amp input capacitance + PCB parasitics`, and use the op-amp data-sheet stability equation:

```text
Cf >= sqrt(Cin / (2*pi*Rf*GBW))
```

For a BPW34/TLC271 implementation with approximately 70-100 pF total input capacitance, **4.7 pF** is a reasonable starting value with `Rf = 27 kOhm`; populate a footprint that also permits 10 pF and 22 pF, then verify the step response and noise on the assembled board. The final Cf must be checked with the actual photodiode bias, layout, op-amp supply, and required measurement bandwidth. Do not treat 10 kOhm/4.7 pF or 27 kOhm/4.7 pF as validated values until the peak current and oscilloscope waveform have been measured.

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

## Hardware raw-data collection

The firmware telemetry build emits one signal sample per line using:

```text
RAW,<sample_index>,<timestamp_ms>,<adc_counts>,<voltage_v>,signal
DATA,<peak>,<mean>,<rise_time_ms>,<decay_time_ms>,<pulse_energy>
```

The existing verified ADC mapping remains ADC0 channel 2 (`ADC0_A2`, J8 pin 12). LED control is deliberately disabled until the optical assembly GPIO is confirmed in `source/hardware_config.h`. After installing `requirements.txt`, build with `MICROPLASTIC_TEST_MODE=OFF`, `MICROPLASTIC_INTERACTIVE_TEST=OFF`, and `MICROPLASTIC_RAW_STREAM=ON`, then collect 250 samples with:

```bash
python3 scripts/raw_data_collector.py --port /dev/ttyACM0 --samples 250
```

The collector writes `data/raw_fluorescence_data.csv` with columns `sample_id`, `timestamp_ms`, `raw_adc`, `voltage_v`, and `pulse_stage`.

## Flash

Use the MCUXpresso for VS Code LinkServer runner for `frdmmcxn236` (onboard
MCU-Link). A board/LinkServer connection is not available in this workspace, so
flash and terminal output remain pending.

## Measurements pending

Flash/RAM usage, actual tensor arena high-water mark, inference latency, ADC
reference confirmation, LED GPIO assignment, and real BPW34/Rhodamine-B
validation require the physical board and circuit.
