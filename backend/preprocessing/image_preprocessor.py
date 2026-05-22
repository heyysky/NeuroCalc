"""
NeuroCalc Preprocessing Pipeline
Handles: grayscale → denoise → threshold → contour → segment
CPU-only, no GPU deps, optimized for low RAM.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Lightweight preprocessing pipeline for handwritten math images.
    Steps: BGR→Gray → Denoise → Threshold → Contour → Crop → Resize
    """

    def __init__(self, target_size: int = 28):
        self.target_size = target_size

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline. Returns normalized float32 array."""
        gray = self._to_grayscale(image)
        denoised = self._denoise(gray)
        binary = self._threshold(denoised)
        cleaned = self._morphological_clean(binary)
        return cleaned

    def segment_symbols(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple]]:
        """
        Segment image into individual symbol crops.
        Returns list of (symbol_image, bounding_box) sorted left-to-right.
        """
        preprocessed = self.preprocess(image)
        contours, _ = cv2.findContours(
            preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        segments = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter noise: skip tiny bounding boxes
            if w < 5 or h < 5:
                continue
            # Add small padding
            pad = 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(preprocessed.shape[1], x + w + pad)
            y2 = min(preprocessed.shape[0], y + h + pad)
            crop = preprocessed[y1:y2, x1:x2]
            resized = self._resize_symbol(crop)
            segments.append((resized, (x1, y1, x2 - x1, y2 - y1)))

        # Sort left-to-right by x coordinate
        segments.sort(key=lambda s: s[1][0])
        return segments

    def prepare_for_cnn(self, symbol_img: np.ndarray) -> np.ndarray:
        """
        Prepare a single symbol image for CNN inference.
        Returns shape (1, 1, 28, 28) float32 normalized.
        """
        if symbol_img.shape != (self.target_size, self.target_size):
            symbol_img = cv2.resize(
                symbol_img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA
            )
        # Normalize to [0, 1]
        normalized = symbol_img.astype(np.float32) / 255.0
        # Shape: (batch=1, channels=1, H, W)
        return normalized.reshape(1, 1, self.target_size, self.target_size)

    # ── Private helpers ──────────────────────────────────────────────────

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def _threshold(self, gray: np.ndarray) -> np.ndarray:
        # Adaptive threshold handles uneven lighting well
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=8,
        )
        return binary

    def _morphological_clean(self, binary: np.ndarray) -> np.ndarray:
        kernel = np.ones((2, 2), np.uint8)
        # Remove small noise
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return opened

    def _resize_symbol(self, crop: np.ndarray) -> np.ndarray:
        """Resize symbol preserving aspect ratio into target_size square."""
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((self.target_size, self.target_size), dtype=np.uint8)

        scale = (self.target_size - 4) / max(h, w)
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center on black canvas
        canvas = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        y_off = (self.target_size - new_h) // 2
        x_off = (self.target_size - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas


def preprocess_drawing_canvas(canvas_data: np.ndarray) -> np.ndarray:
    """
    Special handling for frontend drawing canvas input (white on black or
    black on white). Normalizes to black-on-white convention.
    """
    if len(canvas_data.shape) == 3:
        gray = cv2.cvtColor(canvas_data, cv2.COLOR_RGB2GRAY)
    else:
        gray = canvas_data

    # If mean is dark, assume white-on-black → invert
    if np.mean(gray) < 128:
        gray = 255 - gray

    return gray
