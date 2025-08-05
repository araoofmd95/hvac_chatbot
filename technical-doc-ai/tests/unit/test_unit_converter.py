"""
Unit tests for Unit Converter
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from reasoning.unit_converter import UnitConverter, ConversionResult


class TestUnitConverter:
    """Test cases for UnitConverter"""
    
    @pytest.fixture
    def converter(self):
        """Create a UnitConverter instance for testing"""
        return UnitConverter()
        
    def test_basic_length_conversion(self, converter):
        """Test basic length conversion"""
        result = converter.convert(100, "meter", "foot")
        assert result.error is None
        assert abs(result.converted_value - 328.084) < 0.1
        
    def test_ventilation_units(self, converter):
        """Test building-specific ventilation units"""
        result = converter.convert(1800, "m³/hour", "L/s")
        assert result.error is None
        assert abs(result.converted_value - 500) < 1
        
    def test_area_conversion(self, converter):
        """Test area conversion"""
        result = converter.convert(100, "sqm", "sqft")
        assert result.error is None
        assert result.converted_value > 1000  # 100 sqm > 1000 sqft
        
    def test_volume_flow_rate(self, converter):
        """Test volume flow rate conversions"""
        result = converter.convert(300, "m³/h/car", "L/s")
        assert result.error is None
        # Should handle the /car unit
        
    def test_dimensional_consistency(self, converter):
        """Test dimensional consistency checking"""
        quantities = [
            (100, "meter"),
            (328, "foot"),
            (1000, "millimeter")
        ]
        is_consistent = converter.check_dimensional_consistency(*quantities)
        assert is_consistent
        
    def test_incompatible_units(self, converter):
        """Test conversion between incompatible units"""
        result = converter.convert(100, "meter", "kilogram")
        assert result.error is not None
        assert "incompatible dimensions" in result.error
        
    def test_quantity_parsing(self, converter):
        """Test parsing quantity strings"""
        result = converter.parse_quantity("100 m³/hour")
        assert result is not None
        value, unit = result
        assert value == 100
        assert "m³/hour" in unit or "m3/hour" in unit
        
    def test_si_normalization(self, converter):
        """Test conversion to SI units"""
        si_value, si_unit = converter.normalize_to_si(1, "mile")
        assert abs(si_value - 1609.34) < 1
        assert "meter" in si_unit
        
    def test_custom_building_units(self, converter):
        """Test custom building industry units"""
        # Test air changes per hour
        result = converter.convert(5, "ACH", "1/hour")
        assert result.error is None
        
    def test_suggest_compatible_units(self, converter):
        """Test unit suggestions"""
        suggestions = converter.suggest_compatible_units("meter")
        assert "foot" in suggestions
        assert "inch" in suggestions
        assert "kilometer" in suggestions


if __name__ == "__main__":
    pytest.main([__file__])