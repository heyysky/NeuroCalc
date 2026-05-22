"""
NeuroCalc Recognizer
Combines preprocessing pipeline + ONNX inference + token assembly
"""

import numpy as np
import re
import logging
from typing import Dict, Any, List

from preprocessing.image_preprocessor import ImagePreprocessor, preprocess_drawing_canvas
from recognition.onnx_engine import ONNXInferenceEngine

logger = logging.getLogger(__name__)

# Operator correction rules: common misclassifications
SYMBOL_CORRECTIONS = {
    "l": "1",   # lowercase l → 1
    "I": "1",   # uppercase I → 1
    "O": "0",   # uppercase O → 0
    "o": "0",
    "S": "5",
    "Z": "2",
    "B": "8",
    "G": "6",
    "b": "6",
    "q": "9",
    "g": "9",
}


class MathRecognizer:
    """
    End-to-end math recognizer:
    image → preprocess → segment → classify → assemble equation string
    """

    def __init__(self):
        self._preprocessor = ImagePreprocessor(target_size=28)
        self._engine = ONNXInferenceEngine()
        logger.info(f"MathRecognizer ready (ONNX={self._engine.is_using_onnx})")

    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Full recognition pipeline.
        Returns: {equation, tokens, confidence}
        """
        # Preprocess + segment
        segments = self._preprocessor.segment_symbols(image)

        if not segments:
            return {"equation": "", "tokens": [], "confidence": 0.0}

        # Classify each symbol
        tokens = []
        confidences = []
        for symbol_img, bbox in segments:
            cnn_input = self._preprocessor.prepare_for_cnn(symbol_img)
            symbol, conf = self._engine.predict(cnn_input)

            # Apply correction map
            symbol = SYMBOL_CORRECTIONS.get(symbol, symbol)
            tokens.append({"symbol": symbol, "bbox": bbox, "confidence": round(conf, 3)})
            confidences.append(conf)

        # Assemble equation string
        equation = self._assemble_equation([t["symbol"] for t in tokens])
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            "equation": equation,
            "tokens": tokens,
            "confidence": round(avg_confidence, 3),
        }

    def recognize_from_canvas(self, canvas_array: np.ndarray) -> Dict[str, Any]:
        """Special pipeline for drawing canvas input."""
        normalized = preprocess_drawing_canvas(canvas_array)
        # Convert back to 3-channel for standard pipeline
        rgb = np.stack([normalized, normalized, normalized], axis=2)
        return self.recognize(rgb)

    def _assemble_equation(self, symbols: List[str]) -> str:
        """
        Join symbols into a valid equation string.
        Handles spacing, implicit multiplication, etc.
        """
        if not symbols:
            return ""

        result = []
        for i, sym in enumerate(symbols):
            if sym in ("+", "-", "*", "/", "=", "^", "<", ">"):
                # Operators: surround with spaces
                result.append(f" {sym} ")
            elif sym == "sqrt":
                result.append("sqrt(")
            elif sym == "(":
                result.append("(")
            elif sym == ")":
                # Close sqrt if needed
                result.append(")")
            else:
                result.append(sym)

        equation = "".join(result).strip()
        # Collapse multiple spaces
        equation = re.sub(r"\s{2,}", " ", equation)
        # Fix patterns like "3 x" → "3*x" (implicit multiply)
        equation = re.sub(r"(\d)\s+([a-zA-Z])", r"\1*\2", equation)
        equation = re.sub(r"([a-zA-Z])\s+(\d)", r"\1*\2", equation)

        return equation
