#!/usr/bin/env python3
"""
Keyword Detection Alarm System — Raspberry Pi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Listens via microphone for a keyword using Vosk (offline),
then plays an alarm through the speaker.

Setup: see README_keyword_alarm.md
"""

import json
import queue
import threading
import time
import sys
import os
import wave
import struct
import math

import sounddevice as sd
from vosk import Model, KaldiRecognizer
import pygame

# ── Configuration — edit these ─────────────────────────────────────────────────

KEYWORDS       = ["alarm", "help", "emergency"]  # Words that trigger the alarm
                                                   # (all lowercase, Vosk returns lowercase)

MODEL_PATH     = "./vosk-model-small-en-us"       # Path to your downloaded Vosk model folder
ALARM_FILE     = "./alarm.wav"                    # Path to alarm sound (WAV recommended)
                                                   # Leave as "" to use a generated beep instead

SAMPLE_RATE    = 16000   # Hz — Vosk small models work best at 16 kHz
BLOCK_SIZE     = 8000    # Samples per audio block (~0.5 s at 16 kHz)
MIC_DEVICE     = 2       # USB audio adapter input
                          # Run  python3 -c "import sounddevice as sd; print(sd.query_devices())"
                          # to list available devices

ALARM_DURATION = 10      # Seconds to play the alarm before auto-stopping
COOLDOWN_SEC   = 5       # Seconds to wait before listening again after an alarm

# ── Audio generation helpers ────────────────────────────────────────────────────

def generate_beep_wav(path: str, freq: int = 880, duration: float = 1.0,
                      sample_rate: int = 44100, repeats: int = 5) -> None:
    """Generate a simple repeating beep WAV so there's always a fallback alarm."""
    n_samples = int(sample_rate * duration)
    silence    = int(sample_rate * 0.1)   # 100 ms gap between beeps

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)    # 16-bit
        wf.setframerate(sample_rate)

        for _ in range(repeats):
            # Beep
            for i in range(n_samples):
                value = int(32767 * 0.8 * math.sin(2 * math.pi * freq * i / sample_rate))
                wf.writeframes(struct.pack("<h", value))
            # Silence
            wf.writeframes(b"\x00\x00" * silence)


def ensure_alarm_sound(path: str) -> str:
    """Return path to alarm sound, generating a beep WAV if none exists."""
    if path and os.path.isfile(path):
        return path

    fallback = "./beep_alarm.wav"
    if not os.path.isfile(fallback):
        print("[setup] No alarm file found — generating a fallback beep...")
        generate_beep_wav(fallback)
    return fallback


# ── Alarm player ────────────────────────────────────────────────────────────────

class AlarmPlayer:
    def __init__(self, sound_path: str, duration: int):
        pygame.mixer.init()
        self._sound   = pygame.mixer.Sound(sound_path)
        self._duration = duration
        self._active   = False
        self._lock     = threading.Lock()

    def trigger(self) -> None:
        """Start the alarm in a background thread (non-blocking)."""
        with self._lock:
            if self._active:
                return   # Already ringing
            self._active = True

        def _play():
            print("\n🚨  ALARM TRIGGERED! Press Ctrl+C to stop early.\n")
            self._sound.play(loops=-1)   # Loop indefinitely
            time.sleep(self._duration)
            self._sound.stop()
            with self._lock:
                self._active = False
            print("[alarm] Alarm stopped. Resuming listening...\n")

        threading.Thread(target=_play, daemon=True).start()

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def stop(self) -> None:
        self._sound.stop()
        with self._lock:
            self._active = False


# ── Keyword listener ────────────────────────────────────────────────────────────

class KeywordListener:
    def __init__(self, model_path: str, keywords: list[str],
                 alarm: AlarmPlayer, sample_rate: int,
                 block_size: int, device):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Vosk model not found at '{model_path}'.\n"
                f"Download one from https://alphacephei.com/vosk/models\n"
                f"and extract it so that path exists."
            )

        print(f"[setup] Loading Vosk model from '{model_path}'...")
        model          = Model(model_path)
        self._rec      = KaldiRecognizer(model, sample_rate)
        self._rec.SetWords(True)

        self._keywords   = [kw.lower() for kw in keywords]
        self._alarm      = alarm
        self._block_size = block_size
        self._device     = device
        self._sample_rate = sample_rate
        self._audio_q    = queue.Queue()
        self._running    = False

        print(f"[setup] Listening for keywords: {self._keywords}")

    # sounddevice callback — runs on audio thread
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        self._audio_q.put(bytes(indata))

    def _check_for_keywords(self, text: str) -> None:
        text_lower = text.lower()
        for kw in self._keywords:
            if kw in text_lower:
                print(f"[detect] Keyword '{kw}' detected in: \"{text}\"")
                self._alarm.trigger()
                return

    def run(self) -> None:
        """Block and process audio until KeyboardInterrupt."""
        self._running = True
        print("\n✅  Listening... (say one of your keywords to trigger the alarm)")
        print("    Press Ctrl+C to quit.\n")

        with sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=self._block_size,
            device=self._device,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        ):
            while self._running:
                # Skip processing while alarm is playing
                if self._alarm.is_active:
                    self._audio_q.queue.clear()
                    time.sleep(0.2)
                    continue

                data = self._audio_q.get()

                if self._rec.AcceptWaveform(data):
                    result = json.loads(self._rec.Result())
                    text   = result.get("text", "").strip()
                    if text:
                        print(f"[heard] {text}")
                        self._check_for_keywords(text)
                else:
                    partial = json.loads(self._rec.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    # Check partials too for faster response
                    if partial_text:
                        self._check_for_keywords(partial_text)

    def stop(self) -> None:
        self._running = False


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    alarm_path = ensure_alarm_sound(ALARM_FILE)
    alarm      = AlarmPlayer(alarm_path, ALARM_DURATION)
    listener   = KeywordListener(
        model_path  = MODEL_PATH,
        keywords    = KEYWORDS,
        alarm       = alarm,
        sample_rate = SAMPLE_RATE,
        block_size  = BLOCK_SIZE,
        device      = MIC_DEVICE,
    )

    try:
        listener.run()
    except KeyboardInterrupt:
        print("\n[exit] Stopping...")
        alarm.stop()
        listener.stop()
        pygame.mixer.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
