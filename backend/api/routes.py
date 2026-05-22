"""
NeuroCalc API Routes
POST /predict  - Recognize equation from image
POST /solve    - Solve recognized equation
POST /camera-frame - Process webcam frame
GET  /history  - Retrieve solution history
GET  /health   - Health check
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import base64
import io
import time
import numpy as np
from PIL import Image

from recognition.recognizer import MathRecognizer
from solver.equation_solver import EquationSolver
from database.db import save_history, get_history

router = APIRouter()

# Lazy-loaded singletons
_recognizer: Optional[MathRecognizer] = None
_solver: Optional[EquationSolver] = None


def get_recognizer() -> MathRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = MathRecognizer()
    return _recognizer


def get_solver() -> EquationSolver:
    global _solver
    if _solver is None:
        _solver = EquationSolver()
    return _solver


class CameraFrameRequest(BaseModel):
    frame: str  # base64 encoded image
    skip_frames: int = 3


class SolveRequest(BaseModel):
    equation: str
    show_steps: bool = True


def decode_image(data: bytes) -> np.ndarray:
    """Decode bytes to numpy array via PIL."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


def decode_base64_image(b64: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
    if "," in b64:
        b64 = b64.split(",")[1]
    img_bytes = base64.b64decode(b64)
    return decode_image(img_bytes)


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "mode": "CPU-only",
        "model_loaded": _recognizer is not None,
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Recognize handwritten equation from uploaded image."""
    t0 = time.time()
    try:
        data = await file.read()
        img_array = decode_image(data)
        recognizer = get_recognizer()
        result = recognizer.recognize(img_array)
        elapsed = round(time.time() - t0, 3)
        return {
            "equation": result["equation"],
            "tokens": result["tokens"],
            "confidence": result["confidence"],
            "inference_ms": elapsed * 1000,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/solve")
async def solve(request: SolveRequest):
    """Parse and solve an equation string step-by-step."""
    t0 = time.time()
    try:
        solver = get_solver()
        result = solver.solve(request.equation, show_steps=request.show_steps)
        elapsed = round(time.time() - t0, 3)

        # Persist to history
        save_history(
            equation=request.equation,
            result=str(result.get("result", "")),
            steps=result.get("steps", []),
        )

        return {
            "equation": request.equation,
            "result": result.get("result"),
            "steps": result.get("steps", []),
            "latex": result.get("latex", ""),
            "solve_ms": elapsed * 1000,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not solve: {str(e)}")


@router.post("/predict-and-solve")
async def predict_and_solve(file: UploadFile = File(...)):
    """One-shot: recognize then solve."""
    t0 = time.time()
    try:
        data = await file.read()
        img_array = decode_image(data)

        recognizer = get_recognizer()
        rec_result = recognizer.recognize(img_array)

        solver = get_solver()
        solve_result = solver.solve(rec_result["equation"])

        save_history(
            equation=rec_result["equation"],
            result=str(solve_result.get("result", "")),
            steps=solve_result.get("steps", []),
        )

        elapsed = round(time.time() - t0, 3)
        return {
            "equation": rec_result["equation"],
            "confidence": rec_result["confidence"],
            "result": solve_result.get("result"),
            "steps": solve_result.get("steps", []),
            "latex": solve_result.get("latex", ""),
            "total_ms": elapsed * 1000,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_frame_counter = 0


@router.post("/camera-frame")
async def camera_frame(request: CameraFrameRequest):
    """Process a webcam frame (with frame-skipping for performance)."""
    global _frame_counter
    _frame_counter += 1

    # Frame skipping: only process every N-th frame
    if _frame_counter % max(request.skip_frames, 1) != 0:
        return {"equation": None, "confidence": 0, "skipped": True}

    try:
        img_array = decode_base64_image(request.frame)
        recognizer = get_recognizer()
        result = recognizer.recognize(img_array)
        return {
            "equation": result["equation"],
            "confidence": result["confidence"],
            "skipped": False,
        }
    except Exception as e:
        return {"equation": None, "confidence": 0, "error": str(e), "skipped": False}


@router.get("/history")
async def history(limit: int = 20):
    """Retrieve recent solution history."""
    rows = get_history(limit=limit)
    return {"history": rows}
