"""
NeuroCalc ONNX Runtime Inference Engine
CPU-only, quantized model support, lazy loading
"""

import numpy as np
import os
import logging
from typing import Optional, Tuple, List

from .cnn_model import SYMBOL_CLASSES, CLASS_TO_SYMBOL, FallbackClassifier

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


class ONNXInferenceEngine:
    """
    Runs ONNX CNN model on CPU for symbol classification.
    Falls back to rule-based classifier if no model found.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._session = None
        self._fallback = FallbackClassifier()
        self._use_onnx = False
        self._model_path = model_path or self._find_model()
        self._load_model()

    def _find_model(self) -> Optional[str]:
        """Search for ONNX model in models directory."""
        candidates = [
            os.path.join(MODELS_DIR, "neurocalc_cnn.onnx"),
            os.path.join(MODELS_DIR, "neurocalc_cnn_quantized.onnx"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _load_model(self):
        """Lazy-load ONNX model session."""
        if not self._model_path or not os.path.exists(self._model_path):
            logger.info("No ONNX model found — using fallback classifier")
            return

        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2       # Keep CPU usage low
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            self._session = ort.InferenceSession(
                self._model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],  # No GPU
            )
            self._use_onnx = True
            logger.info(f"✓ ONNX model loaded: {self._model_path}")
        except ImportError:
            logger.warning("onnxruntime not installed — using fallback")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Classify a single symbol image.
        image: (1, 1, 28, 28) float32
        Returns (symbol_str, confidence_float)
        """
        if self._use_onnx and self._session is not None:
            return self._onnx_predict(image)
        else:
            idx, conf = self._fallback.predict(image)
            return CLASS_TO_SYMBOL.get(idx, "?"), conf

    def predict_symbols(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Batch predict (but run one at a time for low memory)."""
        return [self.predict(img) for img in images]

    def _onnx_predict(self, image: np.ndarray) -> Tuple[str, float]:
        """Run ONNX inference."""
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: image})
        logits = outputs[0][0]  # (num_classes,)

        # Softmax
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        symbol = CLASS_TO_SYMBOL.get(idx, "?")
        return symbol, confidence

    @property
    def is_using_onnx(self) -> bool:
        return self._use_onnx
