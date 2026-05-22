"""
NeuroCalc Unit Tests
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

# ── Solver tests ───────────────────────────────────────────────────────────────

class TestEquationSolver:
    def setup_method(self):
        from backend.solver.equation_solver import EquationSolver
        self.solver = EquationSolver()

    def test_arithmetic_addition(self):
        r = self.solver.solve("2 + 3")
        assert r["result"] == 5 or str(r["result"]) == "5"

    def test_arithmetic_multiplication(self):
        r = self.solver.solve("6 * 7")
        assert str(r["result"]) == "42"

    def test_linear_equation(self):
        r = self.solver.solve("2*x + 4 = 10")
        assert "3" in str(r["result"])

    def test_quadratic_equation(self):
        r = self.solver.solve("x**2 - 5*x + 6 = 0")
        solutions = r.get("solutions", [])
        vals = sorted([float(s) for s in solutions])
        assert abs(vals[0] - 2.0) < 0.01
        assert abs(vals[1] - 3.0) < 0.01

    def test_fraction_simplification(self):
        r = self.solver.solve("12/8")
        result_str = str(r["result"])
        assert "3/2" in result_str or "1.5" in result_str

    def test_power_expression(self):
        r = self.solver.solve("2**10")
        assert str(r["result"]) == "1024"

    def test_square_root(self):
        r = self.solver.solve("sqrt(16)")
        assert str(r["result"]) == "4"

    def test_steps_generated(self):
        r = self.solver.solve("x + 5 = 12", show_steps=True)
        assert len(r.get("steps", [])) > 0

    def test_error_graceful(self):
        r = self.solver.solve("???invalid???")
        assert r.get("error") is not None or r.get("result") is None


# ── Parser tests ───────────────────────────────────────────────────────────────

class TestExpressionParser:
    def setup_method(self):
        from backend.parsing.expression_parser import ExpressionParser
        self.parser = ExpressionParser()

    def test_simple_expression(self):
        expr, t, _ = self.parser.parse("2 + 3")
        assert t == "expression"

    def test_equation_detected(self):
        expr, t, _ = self.parser.parse("x = 5")
        assert t == "equation"

    def test_xor_conversion(self):
        # x^2 should become x**2
        expr, t, cleaned = self.parser.parse("x^2")
        import sympy as sp
        assert expr is not None

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            self.parser.parse("")

    def test_implicit_multiplication(self):
        expr, _, _ = self.parser.parse("2x")
        import sympy as sp
        x = sp.Symbol("x")
        assert expr == 2 * x


# ── Preprocessing tests ────────────────────────────────────────────────────────

class TestImagePreprocessor:
    def setup_method(self):
        from backend.preprocessing.image_preprocessor import ImagePreprocessor
        self.preprocessor = ImagePreprocessor(target_size=28)

    def _make_test_image(self, with_marks=True):
        """Create a synthetic test image with some marks."""
        img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        if with_marks:
            # Draw a fake digit
            img[20:80, 20:60] = 0  # Black rectangle = "symbol"
            img[20:80, 120:160] = 0
            img[20:80, 220:260] = 0
        return img

    def test_preprocess_returns_binary(self):
        img = self._make_test_image()
        result = self.preprocessor.preprocess(img)
        assert result.ndim == 2
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_segment_finds_symbols(self):
        img = self._make_test_image(with_marks=True)
        segments = self.preprocessor.segment_symbols(img)
        assert len(segments) >= 1

    def test_segment_empty_image(self):
        img = np.ones((100, 300, 3), dtype=np.uint8) * 255  # All white
        segments = self.preprocessor.segment_symbols(img)
        assert len(segments) == 0

    def test_prepare_for_cnn_shape(self):
        img = self._make_test_image()
        segments = self.preprocessor.segment_symbols(img)
        if segments:
            sym_img, _ = segments[0]
            tensor = self.preprocessor.prepare_for_cnn(sym_img)
            assert tensor.shape == (1, 1, 28, 28)
            assert tensor.dtype == np.float32
            assert tensor.min() >= 0.0 and tensor.max() <= 1.0


# ── Database tests ─────────────────────────────────────────────────────────────

class TestDatabase:
    def setup_method(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        # Patch DB_PATH
        import backend.database.db as db_module
        db_module.DB_PATH = os.path.join(self._tmpdir, "test.db")
        db_module.init_db()
        self.db = db_module

    def test_save_and_retrieve(self):
        self.db.save_history("x+1=2", "1", [{"step": 1, "description": "test"}])
        rows = self.db.get_history(limit=10)
        assert len(rows) >= 1
        assert rows[0]["equation"] == "x+1=2"

    def test_history_limit(self):
        for i in range(5):
            self.db.save_history(f"eq{i}", str(i), [])
        rows = self.db.get_history(limit=3)
        assert len(rows) == 3

    def test_clear_history(self):
        self.db.save_history("test", "1", [])
        self.db.clear_history()
        rows = self.db.get_history()
        assert len(rows) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
