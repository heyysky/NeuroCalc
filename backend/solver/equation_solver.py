"""
NeuroCalc Equation Solver
Step-by-step solutions using SymPy — no heavy LLM required.
Supports: arithmetic, linear, quadratic, fractions, powers, roots, calculus
"""

import sympy as sp
from sympy import (
    solve, simplify, expand, factor, latex,
    diff, integrate, sqrt, Rational, pi, E,
    symbols, Eq, Symbol
)
from typing import Dict, Any, List, Optional
import logging
import re

from parsing.expression_parser import parse_equation

logger = logging.getLogger(__name__)

x, y, z, t = symbols("x y z t")


class EquationSolver:
    """
    Solves math equations with step-by-step explanations.
    Pure SymPy — zero external API calls, zero token usage.
    """

    def solve(self, equation: str, show_steps: bool = True) -> Dict[str, Any]:
        """
        Main solve entry point.
        Returns: {result, steps, latex, equation_type}
        """
        try:
            expr, eq_type, cleaned = parse_equation(equation)
            
            if eq_type == "equation":
                return self._solve_equation(cleaned, expr, show_steps)
            elif eq_type == "calculus":
                return self._solve_calculus(cleaned, expr, show_steps)
            else:
                return self._evaluate_expression(cleaned, expr, show_steps)

        except Exception as e:
            logger.error(f"Solver error for '{equation}': {e}")
            return {
                "result": None,
                "steps": [{"step": 1, "description": f"Error: {str(e)}", "expression": equation}],
                "latex": "",
                "error": str(e),
            }

    # ── Expression evaluation ────────────────────────────────────────────

    def _evaluate_expression(self, eq_str: str, expr: sp.Basic, show_steps: bool) -> Dict:
        steps = []
        step_n = 1

        if show_steps:
            steps.append({
                "step": step_n,
                "description": "Write the expression",
                "expression": str(expr),
                "latex": latex(expr),
            })
            step_n += 1

        # Check if purely numeric
        if expr.is_number:
            result = float(expr.evalf())
            if show_steps:
                simplified = simplify(expr)
                steps.append({
                    "step": step_n,
                    "description": "Evaluate numerically",
                    "expression": f"= {result}",
                    "latex": f"= {latex(simplified)}",
                })
            return {"result": result, "steps": steps, "latex": latex(expr)}

        # Algebraic simplification
        simplified = simplify(expr)
        if show_steps and simplified != expr:
            steps.append({
                "step": step_n,
                "description": "Simplify",
                "expression": str(simplified),
                "latex": latex(simplified),
            })
            step_n += 1

        # Try to expand
        expanded = expand(expr)
        if show_steps and expanded != simplified:
            steps.append({
                "step": step_n,
                "description": "Expand",
                "expression": str(expanded),
                "latex": latex(expanded),
            })
            step_n += 1

        # Try to factor
        factored = factor(expr)
        if show_steps and factored != expanded and str(factored) != str(expr):
            steps.append({
                "step": step_n,
                "description": "Factor",
                "expression": str(factored),
                "latex": latex(factored),
            })

        return {
            "result": str(simplified),
            "steps": steps,
            "latex": latex(simplified),
        }

    # ── Equation solving ─────────────────────────────────────────────────

    def _solve_equation(self, eq_str: str, sympy_eq: sp.Eq, show_steps: bool) -> Dict:
        steps = []
        step_n = 1

        lhs = sympy_eq.lhs
        rhs = sympy_eq.rhs

        if show_steps:
            steps.append({
                "step": step_n,
                "description": "Write the equation",
                "expression": f"{lhs} = {rhs}",
                "latex": f"{latex(lhs)} = {latex(rhs)}",
            })
            step_n += 1

        # Detect equation degree
        free_syms = sympy_eq.free_symbols
        if not free_syms:
            # Numeric equation — evaluate both sides
            lval = float(lhs.evalf())
            rval = float(rhs.evalf())
            is_true = abs(lval - rval) < 1e-9
            return {
                "result": str(is_true),
                "steps": steps,
                "latex": f"{lval} = {rval} \\Rightarrow {is_true}",
            }

        solve_var = list(free_syms)[0]  # Pick first variable

        if show_steps:
            steps.append({
                "step": step_n,
                "description": f"Rearrange to isolate {solve_var}",
                "expression": f"{lhs} - ({rhs}) = 0",
                "latex": f"{latex(lhs)} - ({latex(rhs)}) = 0",
            })
            step_n += 1

        # Solve
        solutions = solve(sympy_eq, solve_var)

        if not solutions:
            return {
                "result": "No real solutions",
                "steps": steps,
                "latex": "\\text{No real solutions}",
            }

        # Degree-specific steps
        diff_expr = sp.expand(lhs - rhs)
        degree = sp.degree(diff_expr, solve_var) if diff_expr.is_polynomial(solve_var) else 1

        if degree == 1 and show_steps:
            steps.extend(self._linear_steps(lhs, rhs, solve_var, step_n))
            step_n += len(steps) - step_n + 1

        elif degree == 2 and show_steps:
            steps.extend(self._quadratic_steps(diff_expr, solve_var, solutions, step_n))
            step_n += 3

        if show_steps:
            sol_str = ", ".join([f"{solve_var} = {s}" for s in solutions])
            sol_latex = ", ".join([f"{latex(solve_var)} = {latex(s)}" for s in solutions])
            steps.append({
                "step": len(steps) + 1,
                "description": "Solution",
                "expression": sol_str,
                "latex": sol_latex,
            })

        result_str = ", ".join([str(s) for s in solutions])
        return {
            "result": result_str,
            "steps": steps,
            "latex": ", ".join([latex(s) for s in solutions]),
            "solutions": [str(s) for s in solutions],
        }

    def _linear_steps(self, lhs, rhs, var, step_n) -> List[Dict]:
        """Generate steps for linear equation."""
        steps = []
        # Move all terms with var to left, constants to right
        lhs_expanded = sp.expand(lhs)
        rhs_expanded = sp.expand(rhs)

        coeff = lhs_expanded.coeff(var) - rhs_expanded.coeff(var)
        const = (rhs_expanded - lhs_expanded).subs(var, 0)

        steps.append({
            "step": step_n,
            "description": f"Collect terms with {var} on the left",
            "expression": f"({coeff}){var} = {const}",
            "latex": f"({latex(coeff)}){latex(var)} = {latex(const)}",
        })

        if coeff != 1 and coeff != 0:
            val = sp.simplify(const / coeff)
            steps.append({
                "step": step_n + 1,
                "description": f"Divide both sides by {coeff}",
                "expression": f"{var} = {val}",
                "latex": f"{latex(var)} = {latex(val)}",
            })
        return steps

    def _quadratic_steps(self, poly, var, solutions, step_n) -> List[Dict]:
        """Generate steps for quadratic using the formula."""
        steps = []
        try:
            a = poly.coeff(var, 2)
            b = poly.coeff(var, 1)
            c = poly.coeff(var, 0)
            disc = b**2 - 4*a*c

            steps.append({
                "step": step_n,
                "description": f"Identify coefficients: a={a}, b={b}, c={c}",
                "expression": f"{a}x² + {b}x + {c} = 0",
                "latex": f"{latex(a)}x^2 + {latex(b)}x + {latex(c)} = 0",
            })
            steps.append({
                "step": step_n + 1,
                "description": "Calculate discriminant Δ = b² - 4ac",
                "expression": f"Δ = {b}² - 4·{a}·{c} = {sp.simplify(disc)}",
                "latex": f"\\Delta = {latex(b)}^2 - 4 \\cdot {latex(a)} \\cdot {latex(c)} = {latex(sp.simplify(disc))}",
            })
            steps.append({
                "step": step_n + 2,
                "description": "Apply quadratic formula: x = (-b ± √Δ) / 2a",
                "expression": f"x = ({-b} ± √{sp.simplify(disc)}) / {2*a}",
                "latex": f"x = \\frac{{{latex(-b)} \\pm \\sqrt{{{latex(sp.simplify(disc))}}}}}{{2 \\cdot {latex(a)}}}",
            })
        except Exception:
            pass
        return steps

    # ── Calculus ─────────────────────────────────────────────────────────

    def _solve_calculus(self, eq_str: str, expr: sp.Basic, show_steps: bool) -> Dict:
        steps = []

        # Check if already differentiated
        if isinstance(expr, sp.Derivative) or "Derivative" in str(type(expr)):
            result = sp.simplify(expr.doit())
        else:
            result = expr

        if show_steps:
            steps = [
                {"step": 1, "description": "Original expression", "expression": str(expr), "latex": latex(expr)},
                {"step": 2, "description": "Apply calculus rules", "expression": str(result), "latex": latex(result)},
                {"step": 3, "description": "Simplify", "expression": str(sp.simplify(result)), "latex": latex(sp.simplify(result))},
            ]

        return {
            "result": str(sp.simplify(result)),
            "steps": steps,
            "latex": latex(sp.simplify(result)),
        }
