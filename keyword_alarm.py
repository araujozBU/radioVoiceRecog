#!/usr/bin/env python3

import json
import queue
import threading
import time
import sys
import os
import wave
import struct
import math

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from vosk import Model, KaldiRecognizer
import pygame


# ── Callsign Configuration ────────────────────────────────────────────────────

TRIGGER_PHRASES = [
    # Tango harbor 1
    "tango harbor one",
    "tango harbour one",
    "tango harbor 1",
    "tango harbour 1",
    "tango harbor",
    "tango harbour",

    # Tango harbor 2
    "tango harbor two",
    "tango harbour two",
    "tango harbor 2",
    "tango harbour 2",

    # Harbor unit
    "harbor unit",
    "harbour unit",

    # Common radio mishearings
    "tango arbor one",
    "tango arbor two",
    "tango arbor",
    "arbor unit",
]

# Safer fallback keywords
FALLBACK_KEYWORDS = [
    "harbor",
    "harbour",
]

GRAMMAR_WORDS = [
    "tango",
    "harbor",
    "harbour",
    "arbor",
    "unit",
    "one",
    "two",
    "1",
    "2",
    "[unk]",
]

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH     = "./vosk-model-small-en-us-0.15"
ALARM_FILE     = "./alarm.wav"

MIC_SAMPLE_RATE = 44100
VOSK_SAMPLE_RATE = 16000

BLOCK_SIZE     = 4000
MIC_DEVICE     = 2

ALARM_DURATION = 10
COOLDOWN_SEC   = 3


# ── Audio Helpers ─────────────────────────────────────────────────────────────

def generate_beep_wav(path: str,
                      freq: int = 880,
                      duration: float = 1.0,
                      sample_rate: int = 44100,
                      repeats: int = 5):

    n_samples = int(sample_rate * duration)
    silence = int(sample_rate * 0.1)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        for _ in range(repeats):

            for i in range(n_samples):
                value = int(
                    32767 * 0.8 *
                    math.sin(2 * math.pi * freq * i / sample_rate)
                )

                wf.writeframes(struct.pack("<h", value))

            wf.writeframes(b"\x00\x00" * silence)


def ensure_alarm_sound(path: str) -> str:

    if path and os.path.isfile(path):
        return path

    fallback = "./beep_alarm.wav"

    if not os.path.isfile(fallback):
        print("[setup] Generating fallback alarm tone...")
        generate_beep_wav(fallback)

    return fallback


def play_startup_tone():

    sample_rate = 44100

    tones = [
        (1200, 0.12),
        (1700, 0.12),
    ]

    for freq, duration in tones:

        arr = np.arange(int(sample_rate * duration))

        wave_data = (
            0.4 * np.sin(
                2 * np.pi * freq * arr / sample_rate
            )
        ).astype(np.float32)

        sd.play(wave_data, sample_rate)
        sd.wait()

        time.sleep(0.05)


# ── Alarm Player ──────────────────────────────────────────────────────────────

class AlarmPlayer:

    def __init__(self, sound_path: str, duration: int):

        pygame.mixer.init()

        self._sound = pygame.mixer.Sound(sound_path)
        self._duration = duration

        self._active = False
        self._lock = threading.Lock()

    def trigger(self, matched: str = ""):

        with self._lock:

            if self._active:
                return

            self._active = True

        def _play():

            print(
                f'\n🚨 ALARM TRIGGERED — matched "{matched}"\n'
            )

            self._sound.play(loops=-1)

            time.sleep(self._duration)

            self._sound.stop()

            with self._lock:
                self._active = False

            print("[alarm] Alarm stopped\n")

        threading.Thread(
            target=_play,
            daemon=True
        ).start()

    @property
    def is_active(self):

        with self._lock:
            return self._active

    def stop(self):

        self._sound.stop()

        with self._lock:
            self._active = False


# ── Keyword Listener ──────────────────────────────────────────────────────────

class KeywordListener:

    def __init__(
        self,
        model_path,
        trigger_phrases,
        fallback_keywords,
        grammar_words,
        alarm,
        block_size,
        device
    ):

        if not os.path.isdir(model_path):

            raise FileNotFoundError(
                f"Vosk model not found: {model_path}"
            )

        print(f"[setup] Loading Vosk model: {model_path}")

        model = Model(model_path)

        grammar_json = json.dumps(grammar_words)

        self._rec = KaldiRecognizer(
            model,
            VOSK_SAMPLE_RATE,
            grammar_json
        )

        self._triggers = [
            p.lower() for p in trigger_phrases
        ]

        self._fallbacks = [
            k.lower() for k in fallback_keywords
        ]

        self._alarm = alarm
        self._block_size = block_size
        self._device = device

        self._audio_q = queue.Queue()

        self._running = False
        self._last_trigger = 0.0

        print(f"[setup] Device: {device}")
        print(f"[setup] Mic sample rate: {MIC_SAMPLE_RATE}")
        print(f"[setup] Vosk sample rate: {VOSK_SAMPLE_RATE}")

    def _audio_callback(
        self,
        indata,
        frames,
        time_info,
        status
    ):

        if status:
            print(f"[audio] {status}", file=sys.stderr)

        audio = np.frombuffer(
            indata,
            dtype=np.int16
        )

        # 44100 → 16000 resample
        audio_16k = resample_poly(
            audio,
            VOSK_SAMPLE_RATE,
            MIC_SAMPLE_RATE
        )

        self._audio_q.put(
            audio_16k.astype(np.int16).tobytes()
        )

    def _check_text(self, text: str, source: str):

        if not text:
            return False

        t = text.lower().strip()

        print(f"[{source}] {t}")

        for phrase in self._triggers:

            if phrase in t:

                self._fire(phrase, t)
                return True

        for kw in self._fallbacks:

            if kw in t:

                self._fire(kw, t)
                return True

        return False

    def _fire(self, matched, full_text):

        now = time.time()

        if now - self._last_trigger < COOLDOWN_SEC:
            return

        self._last_trigger = now

        print(
            f'[detect] "{matched}" detected in "{full_text}"'
        )

        self._alarm.trigger(matched)

    def run(self):

        self._running = True

        with sd.RawInputStream(
            samplerate=MIC_SAMPLE_RATE,
            blocksize=self._block_size,
            device=self._device,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        ):

            print("\n✅ Listening for callsign...\n")

            play_startup_tone()

            while self._running:

                if self._alarm.is_active:

                    self._audio_q.queue.clear()

                    time.sleep(0.1)
                    continue

                data = self._audio_q.get()

                if self._rec.AcceptWaveform(data):

                    result = json.loads(
                        self._rec.Result()
                    )

                    self._check_text(
                        result.get("text", ""),
                        "final"
                    )

                else:

                    partial = json.loads(
                        self._rec.PartialResult()
                    )

                    self._check_text(
                        partial.get("partial", ""),
                        "partial"
                    )

    def stop(self):
        self._running = False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    alarm_path = ensure_alarm_sound(ALARM_FILE)

    alarm = AlarmPlayer(
        alarm_path,
        ALARM_DURATION
    )

    listener = KeywordListener(
        model_path=MODEL_PATH,
        trigger_phrases=TRIGGER_PHRASES,
        fallback_keywords=FALLBACK_KEYWORDS,
        grammar_words=GRAMMAR_WORDS,
        alarm=alarm,
        block_size=BLOCK_SIZE,
        device=MIC_DEVICE,
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