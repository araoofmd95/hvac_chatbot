"""
Unit Converter Module
Handles unit conversions and dimensional analysis using Pint
"""
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import re

import pint
from pint import UnitRegistry, DimensionalityError, UndefinedUnitError
from loguru import logger


@dataclass
class ConversionResult:
    """Result of a unit conversion"""
    original_value: float
    original_unit: str
    converted_value: float
    converted_unit: str
    magnitude_change: float
    dimension: str
    steps: List[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []
            
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary representation"""
        return {
            'original': f"{self.original_value} {self.original_unit}",
            'converted': f"{self.converted_value} {self.converted_unit}",
            'magnitude_change': self.magnitude_change,
            'dimension': self.dimension,
            'steps': self.steps,
            'error': self.error
        }


class UnitConverter:
    """Unit conversion and dimensional analysis engine"""
    
    def __init__(self):
        # Initialize Pint unit registry
        self.ureg = UnitRegistry()
        
        # Add custom units common in building/construction
        self._add_custom_units()
        
        # Common unit aliases
        self.unit_aliases = {
            'cfm': 'cubic_feet/minute',
            'gpm': 'gallons/minute',
            'psi': 'pound_force/inch**2',
            'psf': 'pound_force/foot**2',
            'mph': 'miles/hour',
            'kph': 'kilometers/hour',
            'sqft': 'foot**2',
            'sqm': 'meter**2',
            'cuft': 'foot**3',
            'cum': 'meter**3',
            'L/s': 'liter/second',
            'm³/h': 'meter**3/hour',
            'm³/hour': 'meter**3/hour',
            'm3/h': 'meter**3/hour',
            'm3/hour': 'meter**3/hour'
        }
        
        # Dimension categories
        self.dimension_categories = {
            'length': '[length]',
            'area': '[length]**2',
            'volume': '[length]**3',
            'mass': '[mass]',
            'time': '[time]',
            'temperature': '[temperature]',
            'velocity': '[length]/[time]',
            'acceleration': '[length]/[time]**2',
            'force': '[mass]*[length]/[time]**2',
            'pressure': '[mass]/[length]/[time]**2',
            'energy': '[mass]*[length]**2/[time]**2',
            'power': '[mass]*[length]**2/[time]**3',
            'flow_rate_volume': '[length]**3/[time]',
            'flow_rate_mass': '[mass]/[time]'
        }
        
    def _add_custom_units(self):
        """Add custom units common in technical documents"""
        # Building-specific units
        self.ureg.define('air_changes_per_hour = 1/hour = ACH')
        self.ureg.define('people = [occupancy] = person = persons')
        self.ureg.define('car = [vehicle] = cars = vehicle = vehicles')
        
        # Ventilation rates
        self.ureg.define('liter_per_second_per_person = liter/second/person = L/s/person')
        self.ureg.define('cubic_meter_per_hour_per_car = meter**3/hour/car = m³/h/car')
        
    def convert(self, 
               value: float, 
               from_unit: str, 
               to_unit: str,
               show_steps: bool = True) -> ConversionResult:
        """
        Convert a value from one unit to another
        
        Args:
            value: Numeric value to convert
            from_unit: Source unit
            to_unit: Target unit
            show_steps: Whether to include conversion steps
            
        Returns:
            ConversionResult object
        """
        try:
            # Normalize units
            from_unit_norm = self._normalize_unit(from_unit)
            to_unit_norm = self._normalize_unit(to_unit)
            
            if show_steps:
                steps = [f"Converting {value} {from_unit} to {to_unit}"]
                if from_unit != from_unit_norm or to_unit != to_unit_norm:
                    steps.append(f"Normalized units: {from_unit_norm} → {to_unit_norm}")
            else:
                steps = []
                
            # Create quantity with source unit
            quantity = self.ureg.Quantity(value, from_unit_norm)
            
            if show_steps:
                steps.append(f"Created quantity: {quantity}")
                
            # Get dimension
            dimension = str(quantity.dimensionality)
            
            # Convert to target unit
            converted = quantity.to(to_unit_norm)
            
            if show_steps:
                steps.append(f"Converted to: {converted}")
                
            # Calculate magnitude change
            magnitude_change = converted.magnitude / value if value != 0 else 0
            
            result = ConversionResult(
                original_value=value,
                original_unit=from_unit,
                converted_value=float(converted.magnitude),
                converted_unit=to_unit,
                magnitude_change=magnitude_change,
                dimension=dimension,
                steps=steps
            )
            
            return result
            
        except DimensionalityError as e:
            error_msg = f"Cannot convert between {from_unit} and {to_unit}: incompatible dimensions"
            logger.error(error_msg)
            return ConversionResult(
                original_value=value,
                original_unit=from_unit,
                converted_value=0,
                converted_unit=to_unit,
                magnitude_change=0,
                dimension="",
                error=error_msg
            )
            
        except UndefinedUnitError as e:
            error_msg = f"Undefined unit: {e}"
            logger.error(error_msg)
            return ConversionResult(
                original_value=value,
                original_unit=from_unit,
                converted_value=0,
                converted_unit=to_unit,
                magnitude_change=0,
                dimension="",
                error=error_msg
            )
            
        except Exception as e:
            error_msg = f"Conversion error: {e}"
            logger.error(error_msg)
            return ConversionResult(
                original_value=value,
                original_unit=from_unit,
                converted_value=0,
                converted_unit=to_unit,
                magnitude_change=0,
                dimension="",
                error=error_msg
            )
            
    def normalize_to_si(self, value: float, unit: str) -> Tuple[float, str]:
        """
        Convert any unit to its SI equivalent
        
        Args:
            value: Numeric value
            unit: Source unit
            
        Returns:
            Tuple of (si_value, si_unit)
        """
        try:
            # Create quantity
            unit_norm = self._normalize_unit(unit)
            quantity = self.ureg.Quantity(value, unit_norm)
            
            # Convert to base SI units
            si_quantity = quantity.to_base_units()
            
            return float(si_quantity.magnitude), str(si_quantity.units)
            
        except Exception as e:
            logger.error(f"Error normalizing to SI: {e}")
            return value, unit
            
    def check_dimensional_consistency(self, *quantities: Tuple[float, str]) -> bool:
        """
        Check if multiple quantities have the same dimensions
        
        Args:
            quantities: Variable number of (value, unit) tuples
            
        Returns:
            True if all quantities have the same dimensions
        """
        if len(quantities) < 2:
            return True
            
        try:
            dimensions = []
            
            for value, unit in quantities:
                unit_norm = self._normalize_unit(unit)
                q = self.ureg.Quantity(value, unit_norm)
                dimensions.append(q.dimensionality)
                
            # Check if all dimensions are the same
            return all(d == dimensions[0] for d in dimensions)
            
        except Exception as e:
            logger.error(f"Error checking dimensional consistency: {e}")
            return False
            
    def parse_quantity(self, quantity_str: str) -> Optional[Tuple[float, str]]:
        """
        Parse a quantity string into value and unit
        
        Args:
            quantity_str: String like "100 m³/hour" or "6 cars"
            
        Returns:
            Tuple of (value, unit) or None
        """
        # Patterns to match quantities
        patterns = [
            # Standard: "100 m³/hour"
            r'([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z°³²¹⁰⁴⁵⁶⁷⁸⁹/\-\*\^]+(?:\s*/\s*[a-zA-Z]+)*)',
            
            # With parentheses: "100 (m³/hour)"
            r'([+-]?\d+(?:\.\d+)?)\s*\(([^)]+)\)',
            
            # Compact: "100m³/h"
            r'([+-]?\d+(?:\.\d+)?)([a-zA-Z°³²¹⁰⁴⁵⁶⁷⁸⁹/\-\*\^]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, quantity_str)
            if match:
                value = float(match.group(1))
                unit = match.group(2).strip()
                return value, unit
                
        return None
        
    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit string for Pint"""
        # Check aliases first
        if unit in self.unit_aliases:
            return self.unit_aliases[unit]
            
        # Handle superscript numbers
        superscript_map = {
            '²': '**2', '³': '**3', '⁴': '**4',
            '⁵': '**5', '⁶': '**6', '⁷': '**7',
            '⁸': '**8', '⁹': '**9', '⁰': '**0',
            '¹': '**1'
        }
        
        normalized = unit
        for sup, replacement in superscript_map.items():
            normalized = normalized.replace(sup, replacement)
            
        # Handle special characters
        normalized = normalized.replace('°', 'degree_')
        
        return normalized
        
    def get_dimension_category(self, unit: str) -> Optional[str]:
        """Get the dimension category for a unit"""
        try:
            unit_norm = self._normalize_unit(unit)
            quantity = self.ureg.Quantity(1, unit_norm)
            dim_str = str(quantity.dimensionality)
            
            # Find matching category
            for category, dimension in self.dimension_categories.items():
                if dim_str == dimension:
                    return category
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting dimension category: {e}")
            return None
            
    def suggest_compatible_units(self, unit: str) -> List[str]:
        """Suggest units compatible with the given unit"""
        try:
            unit_norm = self._normalize_unit(unit)
            quantity = self.ureg.Quantity(1, unit_norm)
            dimension = quantity.dimensionality
            
            # Common compatible units by dimension
            suggestions = {
                '[length]': ['meter', 'foot', 'inch', 'mile', 'kilometer'],
                '[length]**2': ['meter**2', 'foot**2', 'acre', 'hectare'],
                '[length]**3': ['meter**3', 'liter', 'gallon', 'foot**3'],
                '[mass]': ['kilogram', 'pound', 'gram', 'ton'],
                '[time]': ['second', 'minute', 'hour', 'day'],
                '[length]**3/[time]': ['meter**3/hour', 'liter/second', 'gallon/minute', 'cubic_feet/minute'],
                '[mass]/[length]/[time]**2': ['pascal', 'psi', 'bar', 'atmosphere']
            }
            
            dim_str = str(dimension)
            return suggestions.get(dim_str, [])
            
        except Exception as e:
            logger.error(f"Error suggesting compatible units: {e}")
            return []
            
    def calculate_with_units(self, 
                           expression: str,
                           variables: Dict[str, Tuple[float, str]]) -> Optional[Tuple[float, str]]:
        """
        Calculate an expression with units
        
        Args:
            expression: Mathematical expression
            variables: Dictionary of variable names to (value, unit) tuples
            
        Returns:
            Tuple of (result_value, result_unit) or None
        """
        try:
            # Create quantities for all variables
            quantities = {}
            for var_name, (value, unit) in variables.items():
                unit_norm = self._normalize_unit(unit)
                quantities[var_name] = self.ureg.Quantity(value, unit_norm)
                
            # Parse and evaluate expression
            # This is a simplified version - full implementation would use SymPy
            result = eval(expression, {"__builtins__": {}}, quantities)
            
            if isinstance(result, self.ureg.Quantity):
                return float(result.magnitude), str(result.units)
            else:
                return float(result), "dimensionless"
                
        except Exception as e:
            logger.error(f"Error calculating with units: {e}")
            return None