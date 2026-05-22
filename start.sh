#!/bin/bash
# NeuroCalc Startup Script (Linux/macOS)
set -e

echo ""
echo " NeuroCalc — CPU-Only Handwritten Math Solver"
echo " ─────────────────────────────────────────────"

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating virtual environment..."
    python3 -m venv venv
    echo "[SETUP] Installing dependencies..."
    venv/bin/pip install --upgrade pip -q
    venv/bin/pip install -r requirements.txt -q
    echo "[SETUP] Done!"
fi

source venv/bin/activate
mkdir -p models

echo "[START] Launching server at http://localhost:8000"
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
