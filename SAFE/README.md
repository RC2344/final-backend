# Gemini Voice Assistant (Python)

A simple, local voice assistant that:
- records your mic
- sends the audio to **Google Gemini** for transcription + response
- speaks the reply back using **offline TTS (pyttsx3)**

Works on macOS, Linux, and Windows.

## 1) Setup

### Python
- Python 3.9+ recommended

### Create & activate venv (recommended)
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows (PowerShell)
```

### Install deps
```bash
pip install -r requirements.txt
```

### Configure API key
Copy `.env.example` to `.env` and paste your Gemini API key.
```bash
cp .env.example .env   # macOS/Linux
# copy .env.example .env   # Windows
```

Edit `.env`:
```ini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash-001
TTS_VOICE=default
TTS_RATE=180
```

> The app uses `pyttsx3` for offline TTS. On macOS it uses the built-in system voices.

## 2) Run

```bash
python main.py
```

Controls:
- Press **ENTER** to start/stop recording your turn.
- Type `q` + ENTER to quit.

## 3) Troubleshooting

- **Microphone permission** (macOS): System Settings → Privacy & Security → Microphone → allow Terminal/your IDE.
- If audio output is silent, try another voice: set `TTS_VOICE` in `.env` (run `python list_voices.py` to see options).
- If `sounddevice` fails on Windows, install your audio drivers and try:
  ```bash
  pip install pipwin
  pipwin install pyaudio
  ```
- Network / 401 errors: make sure `GEMINI_API_KEY` is correct.

## 4) Notes

- This sample sends the **raw WAV** you record to Gemini as audio input and asks it to **reply in the same language** it hears.
- Responses are spoken using offline TTS to minimize dependencies.
- Conversation context is kept in a single chat session.
