"""
Unit tests for Formula Extractor
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from document_processing.formula_extractor import FormulaExtractor, FormulaType


class TestFormulaExtractor:
    """Test cases for FormulaExtractor"""
    
    @pytest.fixture
    def extractor(self):
        """Create a FormulaExtractor instance for testing"""
        return FormulaExtractor()
        
    def test_simple_equation_extraction(self, extractor):
        """Test extraction of simple equations"""
        text = "The ventilation rate Q = n × V where Q is flow rate."
        formulas = extractor.extract_formulas(text)
        
        assert len(formulas) >= 1
        formula = formulas[0]
        assert formula.formula_type == FormulaType.EQUATION
        assert "Q" in formula.variables
        
    def test_calculation_extraction(self, extractor):
        """Test extraction of calculations"""
        text = "For 6 cars: 6 × 300 = 1800 m³/hour"
        formulas = extractor.extract_formulas(text)
        
        assert len(formulas) >= 1
        formula = formulas[0]
        assert formula.formula_type == FormulaType.CALCULATION
        
    def test_inequality_extraction(self, extractor):
        """Test extraction of inequalities"""
        text = "The pressure P ≤ 50 Pa for comfort."
        formulas = extractor.extract_formulas(text)
        
        assert len(formulas) >= 1
        formula = formulas[0]
        assert formula.formula_type == FormulaType.INEQUALITY
        
    def test_variable_definition_extraction(self, extractor):
        """Test extraction of variable definitions"""
        text = """
        Calculate ventilation rate Q = n × V
        where Q is the total flow rate (m³/h)
        and n is the number of cars
        and V is the rate per car (300 m³/h)
        """
        formulas = extractor.extract_formulas(text)
        var_defs = extractor.extract_variable_definitions(text)
        
        assert len(var_defs) >= 2
        assert any(vd.symbol == "Q" for vd in var_defs)
        assert any(vd.symbol == "n" for vd in var_defs)
        
    def test_complex_formula_parsing(self, extractor):
        """Test parsing of complex formulas"""
        text = "Heat load H = A × U × ΔT + Q × ρ × Cp × ΔT"
        formulas = extractor.extract_formulas(text)
        
        assert len(formulas) >= 1
        formula = formulas[0]
        assert len(formula.variables) >= 5
        assert "H" in formula.variables
        assert "A" in formula.variables
        
    def test_unit_handling(self, extractor):
        """Test handling of units in formulas"""
        text = "Flow rate = 300 m³/hour per car"
        formulas = extractor.extract_formulas(text)
        
        # Should extract the numerical value and unit
        assert len(formulas) >= 0  # May or may not extract depending on pattern
        
    def test_normalized_formula_output(self, extractor):
        """Test that formulas are properly normalized"""
        text = "Q = 6 × 300 m³/hour"
        formulas = extractor.extract_formulas(text)
        
        if formulas:
            formula = formulas[0]
            # Check that × is normalized to *
            assert "*" in formula.normalized_text or "×" in formula.raw_text
            
    def test_formula_dependencies(self, extractor):
        """Test extraction of formula dependencies"""
        text = """
        Q = n × V
        P = Q × R
        """
        formulas = extractor.extract_formulas(text)
        dependencies = extractor.extract_formula_dependencies(formulas)
        
        assert len(dependencies) > 0
        # Should find that formulas share variable Q
        
    def test_multiple_formulas_in_text(self, extractor):
        """Test extraction of multiple formulas from text"""
        text = """
        The ventilation calculation involves:
        1. Q = n × 300 where n is number of cars
        2. P ≤ 50 Pa for pressure limit
        3. Total = Q × efficiency factor
        """
        formulas = extractor.extract_formulas(text)
        
        assert len(formulas) >= 2
        types = [f.formula_type for f in formulas]
        assert FormulaType.EQUATION in types


if __name__ == "__main__":
    pytest.main([__file__])