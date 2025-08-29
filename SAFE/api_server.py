import os, sys, threading, subprocess
from collections import deque
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = ROOT / "backend"

# Ensure backend is unpacked
if not BACKEND_DIR.exists():
    print("Unpacking backend from SAFE_original.zip ...")
    import subprocess as sp
    sp.check_call([sys.executable, str(ROOT / "unpack_backend.py")])

def find_main_py(root: Path):
    # Order of preference
    names = ("main.py", "app.py", "run.py")
    candidates = []
    for name in names:
        for p in root.rglob(name):
            candidates.append(p)
    if candidates:
        candidates.sort(key=lambda p: len(p.parts))
        return candidates[0]
    # fallback: any .py containing typical prompt lines
    for p in root.rglob("*.py"):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
            if "Press ENTER to start recording" in t or "Type 'q' + ENTER to quit" in t:
                return p
        except Exception:
            pass
    return None

MAIN_FILE = find_main_py(BACKEND_DIR)
if MAIN_FILE is None:
    raise SystemExit("Could not locate backend main file in 'backend/'. Please ensure SAFE_original.zip contains your project.")

PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)

app = FastAPI(title="Voice Agent Wrapper API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

process = None
reader_thread = None
lines = deque(maxlen=3000)
lock = threading.Lock()

class Command(BaseModel):
    text: str

def _reader(proc):
    for raw in iter(proc.stdout.readline, ''):
        with lock:
            lines.append(raw.rstrip("\n"))
    rest = proc.stdout.read()
    if rest:
        for ln in rest.splitlines():
            with lock:
                lines.append(ln)

def _ensure_process():
    global process, reader_thread
    if process is None or process.poll() is not None:
        process = subprocess.Popen(
            [PYTHON_BIN, str(MAIN_FILE)],
            cwd=str(MAIN_FILE.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        reader_thread = threading.Thread(target=_reader, args=(process,), daemon=True)
        reader_thread.start()
    return process

def _send_stdin(text: str):
    if process is None or process.poll() is not None:
        raise RuntimeError("Process not running")
    process.stdin.write(text)
    process.stdin.flush()

@app.get("/status")
def status():
    running = process is not None and process.poll() is None
    return {"running": running, "main": str(MAIN_FILE)}

@app.post("/start")
def start():
    _ensure_process()
    _send_stdin("\n")
    return {"ok": True}

@app.post("/stop")
def stop():
    _ensure_process()
    _send_stdin("\n")
    return {"ok": True}

@app.post("/quit")
def quit():
    global process
    if process is not None and process.poll() is None:
        try:
            _send_stdin("q\n")
        except Exception:
            pass
        try:
            process.terminate()
        except Exception:
            pass
    process = None
    return {"ok": True}

@app.get("/logs")
def logs(n: int = 200):
    with lock:
        out = list(lines)[-int(n):]
    return {"lines": out}

@app.post("/send")
def send(cmd: Command):
    _ensure_process()
    _send_stdin(cmd.text + ("\n" if not cmd.text.endswith("\n") else ""))
    return {"ok": True}
