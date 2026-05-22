"""
NeuroCalc Expression Parser
Converts recognized equation strings → SymPy expressions
Handles: arithmetic, algebra, quadratics, fractions, powers, sqrt, basic calculus
"""

import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# SymPy parsing transformations
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Common symbols
x, y, z, t = sp.symbols("x y z t")
SYMBOL_MAP = {"x": x, "y": y, "z": z, "t": t}


class ExpressionParser:
    """
    Parses equation/expression strings into SymPy objects.
    """

    def parse(self, equation: str) -> Tuple[Optional[sp.Basic], str, str]:
        """
        Parse equation string.
        Returns: (sympy_expr_or_equation, equation_type, cleaned_str)
        equation_type: 'expression' | 'equation' | 'calculus'
        """
        cleaned = self._clean(equation)

        if not cleaned:
            raise ValueError("Empty equation")

        eq_type = self._detect_type(cleaned)

        try:
            if eq_type == "equation":
                return self._parse_equation(cleaned), "equation", cleaned
            elif eq_type == "calculus":
                return self._parse_calculus(cleaned), "calculus", cleaned
            else:
                expr = parse_expr(cleaned, local_dict=SYMBOL_MAP, transformations=TRANSFORMATIONS)
                return expr, "expression", cleaned
        except Exception as e:
            # Try harder with additional cleanup
            try:
                fallback = self._aggressive_clean(cleaned)
                expr = parse_expr(fallback, local_dict=SYMBOL_MAP, transformations=TRANSFORMATIONS)
                return expr, "expression", fallback
            except Exception:
                raise ValueError(f"Cannot parse: '{equation}' → {e}")

    def _clean(self, eq: str) -> str:
        """Normalize equation string."""
        eq = eq.strip()
        # Replace × with *
        eq = eq.replace("×", "*").replace("÷", "/")
        # Replace √ with sqrt
        eq = eq.replace("√", "sqrt")
        # Replace superscript-like patterns: 2^2 is fine, also handle x² → x**2
        eq = re.sub(r"(\w)\^(\w)", r"\1**\2", eq)
        # Normalize spaces around operators
        eq = re.sub(r"\s*([+\-*/=<>])\s*", r" \1 ", eq)
        eq = eq.replace("  ", " ").strip()
        return eq

    def _aggressive_clean(self, eq: str) -> str:
        """More aggressive cleanup for tough cases."""
        # Remove stray characters
        eq = re.sub(r"[^0-9a-zA-Z+\-*/=().^√\s]", "", eq)
        return eq.strip()

    def _detect_type(self, eq: str) -> str:
        """Detect equation type from string."""
        if "diff(" in eq or "integrate(" in eq or "d/d" in eq:
            return "calculus"
        if "=" in eq:
            return "equation"
        return "expression"

    def _parse_equation(self, eq: str) -> sp.Eq:
        """Parse equation with '=' sign."""
        parts = eq.split("=", 1)
        if len(parts) != 2:
            raise ValueError("Invalid equation: multiple = signs")
        lhs = parse_expr(parts[0].strip(), local_dict=SYMBOL_MAP, transformations=TRANSFORMATIONS)
        rhs = parse_expr(parts[1].strip(), local_dict=SYMBOL_MAP, transformations=TRANSFORMATIONS)
        return sp.Eq(lhs, rhs)

    def _parse_calculus(self, eq: str) -> sp.Basic:
        """Handle calculus expressions."""
        # d/dx(expr) pattern
        m = re.match(r"d/d([a-z])\s*\((.+)\)", eq)
        if m:
            var = sp.Symbol(m.group(1))
            expr = parse_expr(m.group(2), local_dict=SYMBOL_MAP, transformations=TRANSFORMATIONS)
            return sp.diff(expr, var)
        return parse_expr(eq, local_dict=SYMBOL_MAP, transformations=TRANSFORMATIONS)


# Module-level singleton
_parser = ExpressionParser()

def parse_equation(equation: str):
    return _parser.parse(equation)
