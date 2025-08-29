#!/usr/bin/env python3
"""
🎙️ Gemini Voice Assistant (CSV Q&A + Options Selection)
Web-safe version: no sounddevice, no pyttsx3
Frontend handles recording/playback.
Backend handles:
- CSV lookup
- Gemini fallback
- gTTS for audio response (returned as Base64)
"""

import os
import sys
import re
import tempfile
import base64
import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import fuzz
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS

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

# ---------------- FastAPI app ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- CSV Loader ----------------
def load_csv(csv_file):
    if not os.path.exists(csv_file):
        return []
    df = pd.read_csv(csv_file)
    return [(str(q).strip(), str(a).strip()) for q, a in zip(df.iloc[:, 0], df.iloc[:, 1])]

qa_pairs = load_csv(CSV_FILE)

# ---------------- Fuzzy Match ----------------
def score_match(user_text: str, candidate: str) -> float:
    tsr = fuzz.token_sort_ratio(user_text, candidate)
    pr = fuzz.partial_ratio(user_text, candidate)
    wr = fuzz.WRatio(user_text, candidate)
    len_ratio = min(len(user_text), len(candidate)) / max(len(user_text), len(candidate))
    len_score = 100 * len_ratio
    return (0.4 * tsr + 0.3 * pr + 0.2 * wr + 0.1 * len_score)

def find_answer_local(user_text: str, qa_pairs: list, top_k: int = 5):
    user_text = user_text.strip()
    if not user_text:
        return None
    candidates = [(q, score_match(user_text, q), idx) for idx, (q, _) in enumerate(qa_pairs)]
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
    if not candidates:
        return None
    best_q, best_score, best_idx = candidates[0]
    return qa_pairs[best_idx][1] if best_score >= 80 else None

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

# ---------------- API Endpoints ----------------
@app.get("/")
def root():
    return {"status": "Voice Assistant Backend is running"}

@app.post("/ask/")
async def ask(query: str = Form(...)):
    """Handle text query, return text + speech (Base64)"""
    local_answer = find_answer_local(query, qa_pairs)
    if local_answer:
        answer = local_answer
    else:
        answer = query_gemini(query)

    # Generate speech as Base64
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts = gTTS(answer)
        tts.save(tmpfile.name)
        with open(tmpfile.name, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

    return {"answer": answer, "audio_base64": audio_base64}

@app.post("/stt/")
async def stt(file: UploadFile):
    """Accept audio file, return transcription"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(await file.read())
        tmpfile_path = tmpfile.name
    text = transcribe_audio(tmpfile_path)
    return {"transcription": text}
