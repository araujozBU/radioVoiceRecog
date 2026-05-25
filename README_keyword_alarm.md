# Keyword Alarm System — Raspberry Pi Setup Guide

## How it works

```
Microphone → sounddevice → Vosk (offline speech recognition) → keyword match? → pygame alarm
```

- **Vosk** runs entirely offline — no internet, no API key needed.
- **sounddevice** captures raw audio from the mic.
- **pygame** plays your alarm WAV through the speaker.
- If no alarm WAV is provided, a beep tone is auto-generated.

---

## 1. System dependencies

```bash
sudo apt update
sudo apt install -y python3-pip portaudio19-dev libsdl2-mixer-2.0-0
```

## 2. Python dependencies

```bash
pip3 install vosk sounddevice pygame
```

> **Note:** On some Pi setups you may need `pip3 install --break-system-packages vosk sounddevice pygame`
> if you're on Raspberry Pi OS Bookworm (Debian 12).

---

## 3. Download a Vosk model

Vosk models live at: https://alphacephei.com/vosk/models

For a Raspberry Pi, use the **small** English model to keep it fast:

```bash
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
mv vosk-model-small-en-us-0.15 vosk-model-small-en-us
```

Make sure `MODEL_PATH` in `keyword_alarm.py` matches the folder name:
```python
MODEL_PATH = "./vosk-model-small-en-us"
```

---

## 4. (Optional) Add an alarm sound

Place any `.wav` file in the same directory and point to it:
```python
ALARM_FILE = "./alarm.wav"
```
If you leave `ALARM_FILE = ""` or the file doesn't exist, the script **auto-generates
a repeating beep** so it always works out of the box.

Free alarm sounds: https://freesound.org (search "alarm")

---

## 5. Check your audio devices

```bash
# List microphones
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Quick mic test (records 3 s and plays back)
arecord -d 3 test.wav && aplay test.wav
```

Set `MIC_DEVICE = <index>` in the script if the default isn't right.

---

## 6. Run it

```bash
python3 keyword_alarm.py
```

You should see:
```
[setup] Loading Vosk model from './vosk-model-small-en-us'...
[setup] Listening for keywords: ['alarm', 'help', 'emergency']

✅  Listening... (say one of your keywords to trigger the alarm)
    Press Ctrl+C to quit.
```

Say **"alarm"** (or one of your configured keywords) and the alarm fires.

---

## 7. Customise keywords

In `keyword_alarm.py`, change:
```python
KEYWORDS = ["alarm", "help", "emergency"]
```
to any words you want (lowercase, one per entry).

---

## 8. Run on boot (optional)

```bash
# Create a systemd service
sudo nano /etc/systemd/system/keyword-alarm.service
```

Paste:
```ini
[Unit]
Description=Keyword Alarm System
After=sound.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/keyword_alarm.py
WorkingDirectory=/home/pi
Restart=always
User=pi
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Then enable it:
```bash
sudo systemctl daemon-reload
sudo systemctl enable keyword-alarm
sudo systemctl start keyword-alarm
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'vosk'` | Run `pip3 install vosk` |
| `PortAudioError` | Run `sudo apt install portaudio19-dev` |
| No audio output | Check `aplay -l` and set correct ALSA device; try `sudo raspi-config → System Audio` |
| Model not found | Ensure the folder path in `MODEL_PATH` exactly matches the extracted folder name |
| Mic not detected | Run `arecord -l` to list capture devices; set `MIC_DEVICE` index in script |
| Slow / laggy recognition | Use the `vosk-model-small-en-us` model, not the large one |

---

## Hardware tested on

- Raspberry Pi 3B+ / 4 / Zero 2W
- USB microphone (recommended — 3.5mm mics need a USB sound card dongle)
- 3.5mm speaker or USB speaker via the Pi's audio jack
