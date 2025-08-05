"""
Math Engine Module
Handles mathematical calculations and formula evaluation using SymPy
"""
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import re
from decimal import Decimal

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from loguru import logger


@dataclass
class CalculationResult:
    """Result of a mathematical calculation"""
    expression: str
    symbolic_result: sp.Expr
    numeric_result: Union[float, complex, None]
    units: Optional[str] = None
    variables_used: Dict[str, float] = None
    steps: List[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.variables_used is None:
            self.variables_used = {}
        if self.steps is None:
            self.steps = []
            
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary representation"""
        return {
            'expression': self.expression,
            'symbolic_result': str(self.symbolic_result),
            'numeric_result': self.numeric_result,
            'units': self.units,
            'variables_used': self.variables_used,
            'steps': self.steps,
            'error': self.error
        }


class MathEngine:
    """Mathematical calculation engine using SymPy"""
    
    def __init__(self):
        # Common mathematical constants
        self.constants = {
            'pi': sp.pi,
            'e': sp.E,
            'inf': sp.oo,
            'infinity': sp.oo
        }
        
        # Common functions
        self.functions = {
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'asin': sp.asin,
            'acos': sp.acos,
            'atan': sp.atan,
            'sinh': sp.sinh,
            'cosh': sp.cosh,
            'tanh': sp.tanh,
            'exp': sp.exp,
            'log': sp.log,
            'ln': sp.ln,
            'sqrt': sp.sqrt,
            'abs': sp.Abs,
            'floor': sp.floor,
            'ceil': sp.ceiling,
            'min': sp.Min,
            'max': sp.Max
        }
        
        # Parsing transformations
        self.transformations = standard_transformations + (implicit_multiplication_application,)
        
    def evaluate(self, 
                expression: str, 
                variables: Optional[Dict[str, float]] = None,
                return_steps: bool = True) -> CalculationResult:
        """
        Evaluate a mathematical expression
        
        Args:
            expression: Mathematical expression to evaluate
            variables: Dictionary of variable values
            return_steps: Whether to return calculation steps
            
        Returns:
            CalculationResult object
        """
        try:
            # Initialize result
            result = CalculationResult(
                expression=expression,
                symbolic_result=None,
                numeric_result=None
            )
            
            # Clean and parse expression
            cleaned_expr = self._clean_expression(expression)
            if return_steps:
                result.steps.append(f"Original expression: {expression}")
                result.steps.append(f"Cleaned expression: {cleaned_expr}")
                
            # Parse to SymPy expression
            sympy_expr = self._parse_expression(cleaned_expr)
            result.symbolic_result = sympy_expr
            
            if return_steps:
                result.steps.append(f"Parsed expression: {sympy_expr}")
                
            # If variables provided, substitute them
            if variables:
                result.variables_used = variables.copy()
                
                # Create symbol substitutions
                substitutions = {}
                for var_name, value in variables.items():
                    var_symbol = sp.Symbol(var_name)
                    substitutions[var_symbol] = value
                    
                if return_steps:
                    result.steps.append(f"Substituting variables: {substitutions}")
                    
                # Substitute variables
                substituted_expr = sympy_expr.subs(substitutions)
                
                if return_steps:
                    result.steps.append(f"After substitution: {substituted_expr}")
                    
                # Evaluate numerically
                numeric_result = self._evaluate_numeric(substituted_expr)
                result.numeric_result = numeric_result
                
                if return_steps:
                    result.steps.append(f"Numeric result: {numeric_result}")
            else:
                # Try to simplify symbolically
                simplified = sp.simplify(sympy_expr)
                result.symbolic_result = simplified
                
                if return_steps:
                    result.steps.append(f"Simplified expression: {simplified}")
                    
                # If expression is numeric, evaluate it
                if sympy_expr.is_number:
                    result.numeric_result = float(sympy_expr.evalf())
                    
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {e}")
            return CalculationResult(
                expression=expression,
                symbolic_result=None,
                numeric_result=None,
                error=str(e)
            )
            
    def solve_equation(self, 
                      equation: str, 
                      solve_for: str,
                      known_values: Optional[Dict[str, float]] = None) -> CalculationResult:
        """
        Solve an equation for a specific variable
        
        Args:
            equation: Equation string (e.g., "A = B * C")
            solve_for: Variable to solve for
            known_values: Known variable values
            
        Returns:
            CalculationResult with solution
        """
        try:
            result = CalculationResult(
                expression=equation,
                symbolic_result=None,
                numeric_result=None
            )
            
            # Parse equation
            if '=' not in equation:
                raise ValueError("Equation must contain '=' sign")
                
            left_str, right_str = equation.split('=', 1)
            
            # Parse both sides
            left_expr = self._parse_expression(left_str.strip())
            right_expr = self._parse_expression(right_str.strip())
            
            # Create equation
            eq = sp.Eq(left_expr, right_expr)
            result.steps.append(f"Parsed equation: {eq}")
            
            # Solve for variable
            solve_var = sp.Symbol(solve_for)
            solutions = sp.solve(eq, solve_var)
            
            if not solutions:
                result.error = f"No solution found for {solve_for}"
                return result
                
            # Take first solution (handle multiple solutions later)
            solution = solutions[0] if isinstance(solutions, list) else solutions
            result.symbolic_result = solution
            result.steps.append(f"Symbolic solution: {solve_for} = {solution}")
            
            # If known values provided, substitute and evaluate
            if known_values:
                result.variables_used = known_values.copy()
                
                # Substitute known values
                substitutions = {sp.Symbol(k): v for k, v in known_values.items()}
                numeric_solution = solution.subs(substitutions)
                
                result.steps.append(f"After substitution: {solve_for} = {numeric_solution}")
                
                # Evaluate numerically
                if numeric_solution.is_number:
                    result.numeric_result = float(numeric_solution.evalf())
                    result.steps.append(f"Numeric result: {solve_for} = {result.numeric_result}")
                    
            return result
            
        except Exception as e:
            logger.error(f"Error solving equation: {e}")
            return CalculationResult(
                expression=equation,
                symbolic_result=None,
                numeric_result=None,
                error=str(e)
            )
            
    def validate_calculation(self, 
                           expression: str,
                           expected_result: float,
                           variables: Optional[Dict[str, float]] = None,
                           tolerance: float = 0.01) -> Tuple[bool, str]:
        """
        Validate a calculation against expected result
        
        Args:
            expression: Expression to evaluate
            expected_result: Expected numeric result
            variables: Variable values
            tolerance: Acceptable difference ratio
            
        Returns:
            Tuple of (is_valid, message)
        """
        result = self.evaluate(expression, variables, return_steps=False)
        
        if result.error:
            return False, f"Calculation error: {result.error}"
            
        if result.numeric_result is None:
            return False, "No numeric result obtained"
            
        # Check relative difference
        diff = abs(result.numeric_result - expected_result)
        rel_diff = diff / abs(expected_result) if expected_result != 0 else diff
        
        if rel_diff <= tolerance:
            return True, f"Result {result.numeric_result} matches expected {expected_result}"
        else:
            return False, f"Result {result.numeric_result} differs from expected {expected_result} by {rel_diff*100:.1f}%"
            
    def _clean_expression(self, expr: str) -> str:
        """Clean and normalize expression string"""
        # Replace common notation
        replacements = {
            '×': '*',
            '÷': '/',
            '−': '-',
            '–': '-',
            '^': '**',
            '²': '**2',
            '³': '**3',
            '√': 'sqrt',
        }
        
        cleaned = expr
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
            
        # Handle implicit multiplication (e.g., "2x" -> "2*x")
        # This is handled by implicit_multiplication_application transform
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
        
    def _parse_expression(self, expr: str) -> sp.Expr:
        """Parse string expression to SymPy expression"""
        # Add constants and functions to local dict
        local_dict = {}
        local_dict.update(self.constants)
        local_dict.update(self.functions)
        
        try:
            # Parse with transformations
            parsed = parse_expr(
                expr,
                local_dict=local_dict,
                transformations=self.transformations
            )
            return parsed
        except Exception as e:
            logger.error(f"Error parsing expression '{expr}': {e}")
            raise
            
    def _evaluate_numeric(self, expr: sp.Expr) -> Optional[float]:
        """Evaluate SymPy expression numerically"""
        try:
            if expr.is_number:
                # Use evalf for precise evaluation
                result = expr.evalf()
                
                # Convert to float if possible
                if result.is_real:
                    return float(result)
                elif result.is_complex:
                    # Return complex number
                    return complex(result)
                else:
                    # Return as is (might be infinity, etc.)
                    return result
            else:
                # Expression still contains symbols
                return None
        except Exception as e:
            logger.error(f"Error evaluating expression numerically: {e}")
            return None
            
    def extract_variables(self, expression: str) -> List[str]:
        """Extract all variables from an expression"""
        try:
            cleaned = self._clean_expression(expression)
            expr = self._parse_expression(cleaned)
            
            # Get free symbols (variables)
            symbols = expr.free_symbols
            
            # Convert to sorted list of strings
            return sorted([str(s) for s in symbols])
            
        except Exception as e:
            logger.error(f"Error extracting variables: {e}")
            return []
            
    def create_formula_function(self, formula: str, output_var: str) -> Optional[callable]:
        """
        Create a callable function from a formula
        
        Args:
            formula: Formula string (e.g., "V = n * R")
            output_var: Output variable name
            
        Returns:
            Callable function or None
        """
        try:
            # Parse formula
            if '=' not in formula:
                raise ValueError("Formula must contain '='")
                
            parts = formula.split('=')
            if len(parts) != 2:
                raise ValueError("Formula must have exactly one '='")
                
            # Find which side has the output variable
            left = parts[0].strip()
            right = parts[1].strip()
            
            if left == output_var:
                expr_str = right
            elif right == output_var:
                expr_str = left
            else:
                raise ValueError(f"Output variable {output_var} not found in formula")
                
            # Parse expression
            expr = self._parse_expression(expr_str)
            
            # Get variables
            variables = sorted([str(s) for s in expr.free_symbols])
            
            # Create lambda function
            if variables:
                # Create SymPy lambda function
                var_symbols = [sp.Symbol(v) for v in variables]
                func = sp.lambdify(var_symbols, expr, modules=['numpy'])
                
                # Wrap to accept dictionary
                def wrapped_func(**kwargs):
                    # Extract values in correct order
                    values = [kwargs.get(v, 0) for v in variables]
                    return func(*values)
                    
                wrapped_func.variables = variables
                wrapped_func.expression = str(expr)
                
                return wrapped_func
            else:
                # Constant expression
                value = float(expr.evalf())
                
                def const_func(**kwargs):
                    return value
                    
                const_func.variables = []
                const_func.expression = str(expr)
                
                return const_func
                
        except Exception as e:
            logger.error(f"Error creating formula function: {e}")
            return None