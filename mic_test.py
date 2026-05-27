#!/usr/bin/env python3
"""
Microphone Input Test
─────────────────────
Streams mic input and shows a live volume meter in the terminal.
Also runs a Vosk speech recognition test so you can confirm
words are being transcribed correctly before running the main script.

Usage:
    python3 mic_test.py            # volume meter only
    python3 mic_test.py --vosk     # volume meter + live speech transcription
"""

import sys
import math
import queue
import argparse
import numpy as np
import sounddevice as sd

# ── Config ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
BLOCK_SIZE  = 4000   # smaller = more responsive meter
DEVICE      = 2      # USB audio adapter input
METER_WIDTH = 40     # characters wide for the volume bar
# ──────────────────────────────────────────────────────────────────────────────


def rms(data: np.ndarray) -> float:
    """Root-mean-square volume level, normalised 0–1."""
    return math.sqrt(np.mean(data.astype(np.float32) ** 2)) / 32768


def draw_meter(level: float, width: int = METER_WIDTH) -> str:
    filled = int(level * width * 6)   # scale up so normal speech hits ~50%
    filled = min(filled, width)
    bar    = "█" * filled + "░" * (width - filled)

    if level < 0.05:
        label = "quiet "
    elif level < 0.2:
        label = "good  "
    else:
        label = "loud! "

    return f"\r  [{bar}] {label} (level={level:.3f})  "


def list_devices() -> None:
    print("\nAvailable audio devices:")
    print("─" * 50)
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        tag = " ◄ default" if i == sd.default.device[0] else ""
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']}  (inputs: {d['max_input_channels']}){tag}")
    print()


def run_meter_only() -> None:
    """Just show the volume meter — no speech recognition."""
    audio_q: queue.Queue = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        audio_q.put(indata.copy())

    print("\n🎤  Microphone volume meter — speak into your mic!\n")
    print("    Ctrl+C to quit.\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                        device=DEVICE, channels=1, dtype="int16",
                        callback=callback):
        try:
            while True:
                data  = audio_q.get()
                level = rms(data)
                print(draw_meter(level), end="", flush=True)
        except KeyboardInterrupt:
            print("\n\n[done] Meter stopped.")


def run_vosk_test(model_path: str) -> None:
    """Volume meter + live Vosk transcription side by side."""
    import json
    from vosk import Model, KaldiRecognizer

    print(f"\n[setup] Loading Vosk model from '{model_path}'...")
    model = Model(model_path)
    rec   = KaldiRecognizer(model, SAMPLE_RATE)

    audio_q: queue.Queue = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        audio_q.put(bytes(indata))

    print("\n🎤  Microphone test with speech recognition — speak into your mic!")
    print("    Partial results show in yellow, final results in green.")
    print("    Ctrl+C to quit.\n")

    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    RESET  = "\033[0m"

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                           device=DEVICE, dtype="int16", channels=1,
                           callback=callback):
        try:
            while True:
                data  = audio_q.get()

                # Volume meter
                arr   = np.frombuffer(data, dtype=np.int16)
                level = rms(arr)
                print(draw_meter(level), end="", flush=True)

                # Vosk recognition
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text   = result.get("text", "").strip()
                    if text:
                        print(f"\n  {GREEN}✔ Final:   \"{text}\"{RESET}")
                else:
                    partial = json.loads(rec.PartialResult())
                    text    = partial.get("partial", "").strip()
                    if text:
                        print(f"\n  {YELLOW}… Partial: \"{text}\"{RESET}")

        except KeyboardInterrupt:
            print("\n\n[done] Test stopped.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Microphone input test")
    parser.add_argument("--vosk",  action="store_true",
                        help="Also run Vosk speech recognition")
    parser.add_argument("--model", default="./vosk-model-small-en-us",
                        help="Path to Vosk model folder (used with --vosk)")
    parser.add_argument("--devices", action="store_true",
                        help="List available audio input devices and exit")
    parser.add_argument("--device", type=int, default=None,
                        help="Audio device index to use (from --devices list)")
    args = parser.parse_args()

    if args.devices:
        list_devices()
        return

    global DEVICE
    if args.device is not None:
        DEVICE = args.device
        print(f"[setup] Using device index {DEVICE}")

    list_devices()

    if args.vosk:
        run_vosk_test(args.model)
    else:
        run_meter_only()


if __name__ == "__main__":
    main()
