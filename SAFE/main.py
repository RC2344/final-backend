#!/usr/bin/env python3
"""
🎙️ Gemini Voice Assistant (with CSV Q&A + Options Selection)
Mic → STT (Gemini) → Answer (CSV match or Gemini) → TTS
- Uses CSV Q&A as knowledge base
- Gives options if multiple matches, waits for user choice
- Falls back to Gemini if no match
"""

import os
import sys
import queue
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import soundfile as sf
import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import fuzz
import pyttsx3
import re

from google import genai
from google.genai import types


# ---------------- ENV + API ----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ Missing GEMINI_API_KEY in .env")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)


# ---------------- Settings ----------------
SYS_PROMPT_QA = (
    "You are a clear, friendly voice assistant. "
    "Always explain things in short, simple sentences. "
    "Make it sound natural, as if spoken to a beginner. "
    "Do not just read. Rephrase to sound human."
)

CSV_FILE = "qa.csv"
SAMPLE_RATE = 16000
CHANNELS = 1


# ---------------- Recorder ----------------
@dataclass
class Recorder:
    q: queue.Queue = queue.Queue()
    frames: list = None
    recording: bool = False

    def __post_init__(self):
        self.frames = []

    def callback(self, indata, frames, time_, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def record(self):
        self.frames.clear()
        self.recording = True
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self.callback):
            while self.recording:
                try:
                    self.frames.append(self.q.get(timeout=1))
                except queue.Empty:
                    continue

    def stop(self):
        self.recording = False

    def save_wav(self, filename="recorded.wav"):
        audio = np.concatenate(self.frames, axis=0)
        sf.write(filename, audio, SAMPLE_RATE)
        return filename


# ---------------- CSV Loader ----------------
def load_csv(csv_file):
    if not os.path.exists(csv_file):
        print(f"⚠️ CSV file {csv_file} not found. Skipping Q&A.")
        return []
    df = pd.read_csv(csv_file)
    pairs = [(str(q).strip(), str(a).strip()) for q, a in zip(df.iloc[:, 0], df.iloc[:, 1])]
    print(f"📚 Loaded {len(pairs)} Q&A entries from {csv_file}")
    return pairs


# ---------------- Advanced Fuzzy Match ----------------
def score_match(user_text: str, candidate: str) -> float:
    """Composite similarity score between user query and candidate."""

    tsr = fuzz.token_sort_ratio(user_text, candidate)
    pr = fuzz.partial_ratio(user_text, candidate)
    wr = fuzz.WRatio(user_text, candidate)

    len_ratio = min(len(user_text), len(candidate)) / max(len(user_text), len(candidate))
    len_score = 100 * len_ratio

    score = (0.4 * tsr + 0.3 * pr + 0.2 * wr + 0.1 * len_score)

    return score


def find_answer_local(user_text: str, qa_pairs: list, top_k: int = 5):
    """Find closest answers from CSV using composite scoring."""
    user_text = user_text.strip()
    if not user_text:
        return None, []

    candidates = []
    for idx, (q, _) in enumerate(qa_pairs):
        s = score_match(user_text, q)
        candidates.append((q, s, idx))

    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]

    high, medium = 85, 70
    if len(user_text.split()) <= 3:
        high, medium = 90, 80

    best_q, best_score, best_idx = candidates[0]

    if best_score >= high:
        return qa_pairs[best_idx][1], []
    elif best_score >= medium:
        return None, candidates
    else:
        return None, []


# ---------------- Gemini Wrappers ----------------
qa_chat = client.chats.create(
    model="gemini-1.5-flash",
    config=types.GenerateContentConfig(system_instruction=SYS_PROMPT_QA),
)

def query_gemini(user_text: str) -> str:
    try:
        response = qa_chat.send_message(
            [f"User asked: {user_text}\n\nAnswer in clear spoken style."],
            config=types.GenerateContentConfig(
                system_instruction=SYS_PROMPT_QA,
                temperature=0.7,
                max_output_tokens=200,
            ),
        )
        return (response.text or "").strip() or "I don’t have a clear answer for that."
    except Exception as e:
        return f"⚠️ Gemini API error: {e}"


def transcribe_audio(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
            part = types.Part(inline_data=types.Blob(mime_type="audio/wav", data=audio_bytes))

            resp = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[part],
                config=types.GenerateContentConfig(
                    system_instruction="You are a transcription engine. Return only the transcribed text."
                ),
            )
        return (resp.text or "").strip()
    except Exception as e:
        return f"⚠️ STT error: {e}"


# ---------------- Short Speech Summary ----------------
def summarize_for_speech(answer: str) -> str:
    sentences = re.split(r'(?<=[.!?]) +', answer.strip())
    if not sentences:
        return answer

    key_sentences = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        key_sentences.append(s)
        if len(key_sentences) >= 4:
            break

    if not key_sentences:
        key_sentences = sentences[:2]

    bullets = [f"• {s}" for s in key_sentences]
    return "Here are the main points:\n" + "\n".join(bullets)


# ---------------- Speech ----------------
def speak(tts, text: str):
    if not text:
        return
    try:
        clean_text = " ".join(text.split())
        threading.Thread(target=lambda: (tts.say(clean_text), tts.runAndWait())).start()
    except Exception as e:
        print(f"🔇 TTS error: {e}")


# ---------------- Handle Turn ----------------
def handle_turn(user_text, tts, qa_pairs, rec):
    local_answer, candidates = find_answer_local(user_text, qa_pairs)

    if local_answer:
        print(f"\n🤖 Full Answer:\n{local_answer}\n")
        short_reply = summarize_for_speech(local_answer)
        speak(tts, short_reply)
        return

    if candidates:
        print("\n🤖 Clarification needed. Possible matches:\n")
        for i, (q, s, idx) in enumerate(candidates, 1):
            print(f"{i}. {q}")

        clarifying = "I found some similar questions. Please say option number like one, two, or three."
        print("\n🤖 Spoken clarification:\n", clarifying)
        speak(tts, clarifying)

        # Wait for choice
        while True:
            input("Press ENTER and speak your choice...")
            t = threading.Thread(target=rec.record)
            t.start()
            input()
            rec.stop()
            t.join()
            wav_file = rec.save_wav("recorded.wav")

            choice_text = transcribe_audio(wav_file).lower().strip()
            print(f"📝 Choice STT: {choice_text}")

            match = re.search(r"(option\s*)?(\d+|one|two|three|four|five)", choice_text)
            if match:
                val = match.group(2)
                num_map = {"one":1,"two":2,"three":3,"four":4,"five":5}
                choice_num = int(num_map.get(val, val))

                if 1 <= choice_num <= len(candidates):
                    chosen_idx = candidates[choice_num-1][2]
                    chosen_answer = qa_pairs[chosen_idx][1]
                    print(f"\n🤖 Selected Answer:\n{chosen_answer}\n")
                    short_reply = summarize_for_speech(chosen_answer)
                    speak(tts, short_reply)
                    return

            speak(tts, "Sorry, I did not understand. Please say option number again.")

        return

    ai_answer = query_gemini(user_text)
    print(f"\n🤖 Full Answer (Gemini):\n{ai_answer}\n")
    short_reply = summarize_for_speech(ai_answer)
    speak(tts, short_reply)


# ---------------- Main Loop ----------------
def main():
    qa_pairs = load_csv(CSV_FILE)

    tts = pyttsx3.init()
    tts.setProperty("rate", 165)
    tts.setProperty("volume", 1.0)

    rec = Recorder()

    print("\n🎙️ Gemini Voice Assistant (CSV + Gemini + Options)")
    print("Press ENTER to start recording, ENTER again to stop. Type 'q' + ENTER to quit.")

    greeting = "Hello! I am your M C B voice assistant. How can I help you today?"
    print(f"\n🤖 Greeting:\n{greeting}\n")
    speak(tts, greeting)

    while True:
        cmd = input("Press ENTER to record (or 'q' to quit): ")
        if cmd.strip().lower() == "q":
            break

        print("🎤 Recording... Press ENTER to stop.")
        t = threading.Thread(target=rec.record)
        t.start()
        input()
        rec.stop()
        t.join()
        print("⏹️ Stopped recording.")

        wav_file = rec.save_wav("recorded.wav")

        print("📝 Transcribing...")
        user_text = transcribe_audio(wav_file)
        print(f"📝 STT: {user_text}")
        if not user_text:
            continue

        handle_turn(user_text, tts, qa_pairs, rec)


# ---------------- Run ----------------
if __name__ == "__main__":
    main()
