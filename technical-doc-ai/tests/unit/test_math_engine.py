"""
Unit tests for Math Engine
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from reasoning.math_engine import MathEngine, CalculationResult


class TestMathEngine:
    """Test cases for MathEngine"""
    
    @pytest.fixture
    def math_engine(self):
        """Create a MathEngine instance for testing"""
        return MathEngine()
        
    def test_simple_arithmetic(self, math_engine):
        """Test basic arithmetic operations"""
        result = math_engine.evaluate("2 + 3")
        assert result.numeric_result == 5
        assert result.error is None
        
    def test_formula_with_variables(self, math_engine):
        """Test formula evaluation with variables"""
        variables = {"x": 5, "y": 3}
        result = math_engine.evaluate("x * y + 2", variables)
        assert result.numeric_result == 17
        assert result.variables_used == variables
        
    def test_ventilation_calculation(self, math_engine):
        """Test the ventilation calculation example"""
        variables = {"num_cars": 6, "rate_per_car": 300}
        result = math_engine.evaluate("num_cars * rate_per_car", variables)
        assert result.numeric_result == 1800
        
    def test_solve_equation(self, math_engine):
        """Test equation solving"""
        known_values = {"B": 300, "C": 6}
        result = math_engine.solve_equation("A = B * C", "A", known_values)
        assert result.numeric_result == 1800
        
    def test_validation(self, math_engine):
        """Test calculation validation"""
        variables = {"x": 10, "y": 5}
        is_valid, message = math_engine.validate_calculation(
            "x + y", 15, variables
        )
        assert is_valid
        assert "matches expected" in message
        
    def test_variable_extraction(self, math_engine):
        """Test variable extraction from expressions"""
        variables = math_engine.extract_variables("A = B * C + D")
        assert set(variables) == {"A", "B", "C", "D"}
        
    def test_error_handling(self, math_engine):
        """Test error handling for invalid expressions"""
        result = math_engine.evaluate("invalid expression @@")
        assert result.error is not None
        assert result.numeric_result is None


if __name__ == "__main__":
    pytest.main([__file__])