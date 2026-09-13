#!/usr/bin/env python3
"""Collect RAW telemetry from the FRDM-MCXN236 into a CSV file."""
import argparse
import csv
import sys
import time
from pathlib import Path

import serial
from serial.tools import list_ports

FIELDS = ["sample_id", "timestamp_ms", "raw_adc", "voltage_v", "pulse_stage"]


def detect_port():
    ports = [p.device for p in list_ports.comports() if "ACM" in p.device or "COM" in p.device]
    if len(ports) == 1:
        return ports[0]
    if not ports:
        raise RuntimeError("No serial port found; specify --port explicitly")
    raise RuntimeError("Multiple serial ports found: " + ", ".join(ports))


def parse_raw(parts):
    if len(parts) == 5:
        _, sample_id, timestamp_ms, raw_adc, voltage_v = parts
        pulse_stage = "signal"
    elif len(parts) == 6:
        _, sample_id, timestamp_ms, raw_adc, voltage_v, pulse_stage = parts
    else:
        return None
    try:
        return {"sample_id": int(sample_id), "timestamp_ms": int(timestamp_ms),
                "raw_adc": int(raw_adc), "voltage_v": float(voltage_v),
                "pulse_stage": pulse_stage or "unknown"}
    except ValueError:
        return None


def collect(port, baud, output, samples, timeout):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    count = 0
    with serial.Serial(port, baudrate=baud, timeout=1) as connection, output.open("w", newline="", encoding="utf-8") as handle:
        connection.reset_input_buffer()
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        while count < samples:
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"Timed out after {count}/{samples} RAW samples")
            line = connection.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if parts[0] == "DATA":
                print("DATA," + ",".join(parts[1:]))
                continue
            if parts[0] != "RAW":
                continue
            row = parse_raw(parts)
            if row is None:
                print(f"Skipping malformed RAW line: {line!r}", file=sys.stderr)
                continue
            writer.writerow(row)
            handle.flush()
            count += 1
            if count % 25 == 0 or count == samples:
                print(f"Collected {count}/{samples} RAW samples")
    print(f"Saved {count} samples to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--output", default="data/raw_fluorescence_data.csv")
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    try:
        collect(args.port or detect_port(), args.baud, args.output, args.samples, args.timeout)
    except (OSError, RuntimeError, TimeoutError) as exc:
        parser.exit(1, f"raw_data_collector: {exc}\n")


if __name__ == "__main__":
    main()
