import zipfile, sys, os
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
ZIP_PATH = ROOT / "SAFE_original.zip"
DEST = ROOT / "backend"

if DEST.exists():
    print("Backend already present at", DEST)
    sys.exit(0)

if not ZIP_PATH.exists():
    print("SAFE_original.zip not found!", file=sys.stderr)
    sys.exit(1)

print("Extracting backend to", DEST)
DEST.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    zf.extractall(DEST)
print("Done.")
