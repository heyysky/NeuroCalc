"""
NeuroCalc Lightweight CNN Model
< 15MB, CPU-only, ONNX-exportable
Trained on EMNIST digits + custom math operators
"""

import numpy as np
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)

# ── Symbol classes ─────────────────────────────────────────────────────────────
# Index → symbol mapping for the CNN classifier
SYMBOL_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "-", "*", "/", "=",
    "x", "y", "z",
    "(", ")",
    "^", "sqrt",
    ".", ",",
    "<", ">",
]

NUM_CLASSES = len(SYMBOL_CLASSES)
CLASS_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOL_CLASSES)}
SYMBOL_TO_CLASS = {s: i for i, s in enumerate(SYMBOL_CLASSES)}


def get_model_definition():
    """
    Returns the lightweight CNN architecture spec.
    Architecture: 3 conv blocks + 2 FC layers
    ~600K parameters — fits in <15MB after quantization.
    
    Input:  (batch, 1, 28, 28)
    Output: (batch, num_classes) logits
    """
    return {
        "name": "NeuroCalcCNN",
        "input_shape": [1, 1, 28, 28],
        "output_shape": [1, NUM_CLASSES],
        "layers": [
            # Block 1
            {"type": "conv2d", "in": 1,  "out": 16, "kernel": 3, "padding": 1},
            {"type": "bn2d",   "features": 16},
            {"type": "relu"},
            {"type": "maxpool", "kernel": 2},          # 14x14

            # Block 2
            {"type": "conv2d", "in": 16, "out": 32, "kernel": 3, "padding": 1},
            {"type": "bn2d",   "features": 32},
            {"type": "relu"},
            {"type": "maxpool", "kernel": 2},          # 7x7

            # Block 3
            {"type": "conv2d", "in": 32, "out": 64, "kernel": 3, "padding": 1},
            {"type": "bn2d",   "features": 64},
            {"type": "relu"},
            {"type": "adaptavgpool", "output": 4},     # 4x4

            # Head
            {"type": "flatten"},
            {"type": "linear", "in": 64*4*4, "out": 128},
            {"type": "relu"},
            {"type": "dropout", "p": 0.3},
            {"type": "linear", "in": 128, "out": NUM_CLASSES},
        ]
    }


class FallbackClassifier:
    """
    Rule-based fallback when no ONNX model is available.
    Uses pixel density heuristics for rough digit recognition.
    Good enough for demos and testing.
    """

    def predict(self, image_array: np.ndarray) -> tuple[int, float]:
        """
        image_array: (1, 1, 28, 28) float32
        Returns (class_index, confidence)
        """
        img = image_array[0, 0]  # (28, 28)
        density = np.mean(img)
        
        # Very rough heuristic based on pixel density patterns
        # In production: replaced by ONNX model
        if density < 0.05:
            return 0, 0.3  # probably "0" or empty
        
        # Compute basic shape features
        top_half = np.mean(img[:14, :])
        bottom_half = np.mean(img[14:, :])
        left_half = np.mean(img[:, :14])
        right_half = np.mean(img[:, 14:])
        
        ratio_tb = top_half / (bottom_half + 1e-6)
        ratio_lr = left_half / (right_half + 1e-6)
        
        # Heuristic classification
        if ratio_tb > 1.5:
            return 7, 0.4  # top-heavy → "7"
        elif ratio_tb < 0.7:
            return 6, 0.4  # bottom-heavy → "6" 
        elif ratio_lr > 1.3:
            return 4, 0.4  # left-heavy → "4"
        elif density > 0.3:
            return 8, 0.4  # dense → "8"
        else:
            return 1, 0.35  # default → "1"

    def predict_batch(self, images: np.ndarray) -> list:
        return [self.predict(images[i:i+1]) for i in range(len(images))]
