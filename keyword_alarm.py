#!/usr/bin/env python3
"""
Keyword Detection Alarm System — Raspberry Pi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Listens via microphone for a keyword/callsign using Vosk (offline),
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

# ── Callsign configuration ─────────────────────────────────────────────────────
#
# Unit callsigns to detect:
#   "tango harbor 1"   — primary unit callsign
#   "tango harbor 2"   — secondary unit callsign
#   "harbor unit"      — alternate callsign format
#
# Strategy:
#   - "harbor" is the unique identifier for this unit — other tango units
#     won't say it, so it is safe to use as a standalone fallback trigger.
#   - Full phrases checked first (most specific, lowest false-positive risk).
#   - "harbor" / "harbour" alone used as fallback — catches partial phrases
#     and noisy transmissions without risking false triggers from other
#     tango units that never say "harbor".
#   - We do NOT use "tango" alone as a trigger since other units share it.
# ──────────────────────────────────────────────────────────────────────────────

TRIGGER_PHRASES = [
    # ── Tango harbor 1 ──
    "tango harbor one",
    "tango harbour one",
    "tango harbor 1",
    "tango harbour 1",
    # Vosk sometimes drops the number entirely on noisy audio
    "tango harbor",
    "tango harbour",

    # ── Tango harbor 2 ──
    "tango harbor two",
    "tango harbour two",
    "tango harbor 2",
    "tango harbour 2",

    # ── Harbor unit ──
    "harbor unit",
    "harbour unit",

    # ── Common mishearings / phonetic drift ──
    # "harbor" can sound like "arbor" or "arber" on radio
    "tango arbor one",
    "tango arbor two",
    "tango arbor",
    "arbor unit",
]

# Fallback: fire if ANY of these words appear in the transcription.
# "harbor"/"harbour" is unique enough to this unit that it's safe alone.
FALLBACK_KEYWORDS = [
    "harbor",
    "harbour",
    "arbor",    # phonetic mishearing of harbor
]

# Vosk grammar: the ONLY words Vosk is allowed to recognise.
# Keeping this tight = faster processing + higher accuracy.
# Must include every word from TRIGGER_PHRASES and FALLBACK_KEYWORDS.
GRAMMAR_WORDS = [
    "tango",
    "harbor", "harbour",
    "arbor",              # phonetic variant
    "unit",
    "one", "two",
    "1", "2",
    "[unk]",              # always include — handles silence/noise gracefully
]

MODEL_PATH     = "./vosk-model-small-en-us"  # Path to Vosk model folder
ALARM_FILE     = "./alarm.wav"               # WAV alarm sound; "" = auto-generate beep

SAMPLE_RATE    = 44100   # Hz — do not change, Vosk small models require 16 kHz
BLOCK_SIZE     = 4000    # Reduced from 8000 → faster response (~0.25 s blocks)
MIC_DEVICE     = 2       # USB audio adapter input

ALARM_DURATION = 10      # Seconds the alarm plays before auto-stopping
COOLDOWN_SEC   = 3       # Seconds to ignore input after alarm triggers

# ── Audio generation helpers ────────────────────────────────────────────────────

def generate_beep_wav(path: str, freq: int = 880, duration: float = 1.0,
                      sample_rate: int = 44100, repeats: int = 5) -> None:
    """Generate a simple repeating beep WAV so there's always a fallback alarm."""
    n_samples = int(sample_rate * duration)
    silence    = int(sample_rate * 0.1)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        for _ in range(repeats):
            for i in range(n_samples):
                value = int(32767 * 0.8 * math.sin(2 * math.pi * freq * i / sample_rate))
                wf.writeframes(struct.pack("<h", value))
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
        self._sound    = pygame.mixer.Sound(sound_path)
        self._duration = duration
        self._active   = False
        self._lock     = threading.Lock()

    def trigger(self, matched: str = "") -> None:
        """Start the alarm in a background thread (non-blocking)."""
        with self._lock:
            if self._active:
                return
            self._active = True

        def _play():
            print(f"\n🚨  ALARM TRIGGERED!  matched: \"{matched}\"\n"
                  f"    Press Ctrl+C to stop early.\n")
            self._sound.play(loops=-1)
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
    def __init__(self, model_path: str, trigger_phrases: list[str],
                 fallback_keywords: list[str], grammar_words: list[str],
                 alarm: AlarmPlayer, sample_rate: int,
                 block_size: int, device):

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Vosk model not found at '{model_path}'.\n"
                f"Download one from https://alphacephei.com/vosk/models\n"
                f"and extract it so that path exists."
            )

        print(f"[setup] Loading Vosk model from '{model_path}'...")
        model = Model(model_path)

        # ── Constrained grammar mode ──────────────────────────────────────────
        # Pass a JSON array of allowed words to KaldiRecognizer.
        # This focuses Vosk on ONLY these words — much faster + more accurate
        # than open transcription for fixed-phrase detection.
        grammar_json = json.dumps(grammar_words)
        self._rec = KaldiRecognizer(model, sample_rate, grammar_json)

        self._triggers  = [p.lower() for p in trigger_phrases]
        self._fallbacks = [k.lower() for k in fallback_keywords]
        self._alarm     = alarm
        self._block_size  = block_size
        self._device      = device
        self._sample_rate = sample_rate
        self._audio_q     = queue.Queue()
        self._running     = False
        self._last_trigger = 0.0   # timestamp of last alarm trigger

        print(f"[setup] Trigger phrases : {self._triggers}")
        print(f"[setup] Fallback keywords: {self._fallbacks}")
        print(f"[setup] Grammar vocab   : {grammar_words}")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        self._audio_q.put(bytes(indata))

    def _check_text(self, text: str, source: str) -> bool:
        """
        Returns True and fires alarm if text matches a trigger phrase or fallback.
        Checks full phrases first (more specific), then individual keywords.
        """
        if not text:
            return False

        t = text.lower().strip()
        print(f"[{source}] {t}")

        # Check full trigger phrases first
        for phrase in self._triggers:
            if phrase in t:
                self._fire(phrase, t)
                return True

        # Fallback: individual keywords
        for kw in self._fallbacks:
            if kw in t:
                self._fire(kw, t)
                return True

        return False

    def _fire(self, matched: str, full_text: str) -> None:
        now = time.time()
        if now - self._last_trigger < COOLDOWN_SEC:
            print(f"[cooldown] Ignoring match '{matched}' — too soon after last alarm")
            return
        self._last_trigger = now
        print(f"[detect] Matched '{matched}' in: \"{full_text}\"")
        self._alarm.trigger(matched)

    def run(self) -> None:
        self._running = True
        print("\n✅  Listening... (say your callsign to trigger the alarm)")
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
                if self._alarm.is_active:
                    self._audio_q.queue.clear()
                    time.sleep(0.1)
                    continue

                data = self._audio_q.get()

                if self._rec.AcceptWaveform(data):
                    result = json.loads(self._rec.Result())
                    self._check_text(result.get("text", ""), "final  ")
                else:
                    partial = json.loads(self._rec.PartialResult())
                    self._check_text(partial.get("partial", ""), "partial")

    def stop(self) -> None:
        self._running = False


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    alarm_path = ensure_alarm_sound(ALARM_FILE)
    alarm      = AlarmPlayer(alarm_path, ALARM_DURATION)
    listener   = KeywordListener(
        model_path       = MODEL_PATH,
        trigger_phrases  = TRIGGER_PHRASES,
        fallback_keywords= FALLBACK_KEYWORDS,
        grammar_words    = GRAMMAR_WORDS,
        alarm            = alarm,
        sample_rate      = SAMPLE_RATE,
        block_size       = BLOCK_SIZE,
        device           = MIC_DEVICE,
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
