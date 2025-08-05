"""
Formula Extractor Module
Identifies and extracts mathematical formulas and expressions from text
"""
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class FormulaType(Enum):
    """Types of formulas that can be extracted"""
    EQUATION = "equation"
    INEQUALITY = "inequality"
    DEFINITION = "definition"
    CALCULATION = "calculation"
    REFERENCE = "reference"


@dataclass
class ExtractedFormula:
    """Represents an extracted mathematical formula"""
    formula_id: str
    raw_text: str
    normalized_text: str
    formula_type: FormulaType
    variables: List[str]
    operators: List[str]
    context: str
    source_location: Dict[str, any]
    metadata: Dict[str, any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    def to_dict(self) -> Dict[str, any]:
        """Convert formula to dictionary"""
        return {
            'formula_id': self.formula_id,
            'raw_text': self.raw_text,
            'normalized_text': self.normalized_text,
            'formula_type': self.formula_type.value,
            'variables': self.variables,
            'operators': self.operators,
            'context': self.context,
            'source_location': self.source_location,
            'metadata': self.metadata
        }


@dataclass
class VariableDefinition:
    """Definition of a variable found in text"""
    symbol: str
    name: str
    unit: Optional[str] = None
    description: Optional[str] = None
    constraints: List[str] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []


class FormulaExtractor:
    """Extracts mathematical formulas and expressions from text"""
    
    def __init__(self):
        # Common mathematical operators and symbols
        self.operators = {
            '+', '-', '*', '/', '=', '≈', '≤', '≥', '<', '>', '≠',
            '×', '÷', '±', '∑', '∏', '∫', '√', '^', '²', '³'
        }
        
        # Common formula patterns
        self.formula_patterns = [
            # Standard equations: A = B * C
            r'([A-Za-z_]\w*)\s*=\s*([^,\.\n]+)',
            
            # Inequalities: A ≤ B
            r'([A-Za-z_]\w*)\s*[<>≤≥]\s*([^,\.\n]+)',
            
            # Definitions with parentheses: Q (flow rate) = ...
            r'([A-Za-z_]\w*)\s*\([^)]+\)\s*=\s*([^,\.\n]+)',
            
            # LaTeX style: $formula$
            r'\$([^\$]+)\$',
            
            # Inline calculations: 6 × 300 = 1800
            r'(\d+(?:\.\d+)?)\s*[×\*]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
            
            # Unit expressions: m³/hour, L/s
            r'(\w+[²³]?)\s*/\s*(\w+)'
        ]
        
        # Variable definition patterns
        self.variable_patterns = [
            # "where X is the ..."
            r'where\s+([A-Za-z_]\w*)\s+(?:is|=)\s+(?:the\s+)?([^,\.\n]+)',
            
            # "X = variable name (unit)"
            r'([A-Za-z_]\w*)\s*=\s*([^(]+)\s*\(([^)]+)\)',
            
            # "Let X be ..."
            r'[Ll]et\s+([A-Za-z_]\w*)\s+be\s+(?:the\s+)?([^,\.\n]+)',
            
            # "X: description"
            r'([A-Za-z_]\w*)\s*:\s*([^,\.\n]+)',
            
            # "X (description) = ..."
            r'([A-Za-z_]\w*)\s*\(([^)]+)\)\s*='
        ]
        
    def extract_formulas(self, text: str, section_info: Optional[Dict[str, any]] = None) -> List[ExtractedFormula]:
        """
        Extract all formulas from text
        
        Args:
            text: Text to extract formulas from
            section_info: Optional section/location information
            
        Returns:
            List of extracted formulas
        """
        formulas = []
        formula_count = 0
        
        # Split text into lines for line-by-line processing
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue
                
            # Try each pattern
            for pattern in self.formula_patterns:
                matches = re.finditer(pattern, line)
                
                for match in matches:
                    formula_count += 1
                    
                    # Extract formula components
                    raw_text = match.group(0)
                    normalized = self._normalize_formula(raw_text)
                    
                    # Determine formula type
                    formula_type = self._determine_formula_type(raw_text)
                    
                    # Extract variables and operators
                    variables = self._extract_variables(normalized)
                    operators = self._extract_operators(normalized)
                    
                    # Get context (surrounding text)
                    context = self._get_context(lines, line_num)
                    
                    # Create source location
                    source_location = {
                        'line_number': line_num + 1,
                        'char_start': match.start(),
                        'char_end': match.end()
                    }
                    if section_info:
                        source_location.update(section_info)
                        
                    # Create unique formula ID
                    import uuid
                    unique_id = str(uuid.uuid4())[:8]
                    section_prefix = section_info.get('section_number', 'unknown') if section_info else 'unknown'
                    formula_id = f"formula_{section_prefix}_{unique_id}"
                        
                    # Create formula object
                    formula = ExtractedFormula(
                        formula_id=formula_id,
                        raw_text=raw_text,
                        normalized_text=normalized,
                        formula_type=formula_type,
                        variables=variables,
                        operators=operators,
                        context=context,
                        source_location=source_location
                    )
                    
                    formulas.append(formula)
                    
        # Extract variable definitions
        variable_defs = self.extract_variable_definitions(text)
        
        # Link variables to their definitions
        self._link_variables_to_definitions(formulas, variable_defs)
        
        logger.info(f"Extracted {len(formulas)} formulas")
        return formulas
        
    def extract_variable_definitions(self, text: str) -> List[VariableDefinition]:
        """Extract variable definitions from text"""
        definitions = []
        seen_symbols = set()
        
        for pattern in self.variable_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    symbol = groups[0].strip()
                    description = groups[1].strip()
                    
                    # Skip if already seen
                    if symbol in seen_symbols:
                        continue
                        
                    seen_symbols.add(symbol)
                    
                    # Extract unit if present
                    unit = None
                    if len(groups) >= 3:
                        unit = groups[2].strip()
                    else:
                        # Try to extract unit from description
                        unit_match = re.search(r'\(([^)]+)\)$', description)
                        if unit_match:
                            unit = unit_match.group(1)
                            description = description[:unit_match.start()].strip()
                            
                    # Create definition
                    var_def = VariableDefinition(
                        symbol=symbol,
                        name=description,
                        unit=unit
                    )
                    
                    definitions.append(var_def)
                    
        return definitions
        
    def _normalize_formula(self, formula: str) -> str:
        """Normalize formula text for processing"""
        # Remove extra whitespace
        normalized = ' '.join(formula.split())
        
        # Replace Unicode math symbols with standard ones
        replacements = {
            '×': '*',
            '÷': '/',
            '≤': '<=',
            '≥': '>=',
            '≠': '!=',
            '≈': '~=',
            '²': '^2',
            '³': '^3',
            '√': 'sqrt'
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
            
        return normalized
        
    def _determine_formula_type(self, formula: str) -> FormulaType:
        """Determine the type of formula"""
        if any(op in formula for op in ['<', '>', '≤', '≥', '<=', '>=']):
            return FormulaType.INEQUALITY
        elif '=' in formula and re.search(r'\d+\s*[*×]\s*\d+', formula):
            return FormulaType.CALCULATION
        elif '=' in formula:
            return FormulaType.EQUATION
        elif ':=' in formula or 'is defined as' in formula.lower():
            return FormulaType.DEFINITION
        else:
            return FormulaType.REFERENCE
            
    def _extract_variables(self, formula: str) -> List[str]:
        """Extract variable names from formula"""
        # Pattern for variable names (letters followed by optional subscript)
        pattern = r'[A-Za-z]\w*(?:_\w+)?'
        
        # Find all matches
        matches = re.findall(pattern, formula)
        
        # Filter out common function names
        functions = {'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'min', 'max'}
        variables = [m for m in matches if m.lower() not in functions]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_vars = []
        for var in variables:
            if var not in seen:
                seen.add(var)
                unique_vars.append(var)
                
        return unique_vars
        
    def _extract_operators(self, formula: str) -> List[str]:
        """Extract mathematical operators from formula"""
        found_operators = []
        
        for op in self.operators:
            if op in formula:
                found_operators.append(op)
                
        return found_operators
        
    def _get_context(self, lines: List[str], line_num: int, context_size: int = 2) -> str:
        """Get surrounding context for a formula"""
        start = max(0, line_num - context_size)
        end = min(len(lines), line_num + context_size + 1)
        
        context_lines = lines[start:end]
        return '\n'.join(context_lines)
        
    def _link_variables_to_definitions(self, 
                                     formulas: List[ExtractedFormula], 
                                     definitions: List[VariableDefinition]):
        """Link variables in formulas to their definitions"""
        # Create lookup dictionary
        def_lookup = {d.symbol: d for d in definitions}
        
        # Add definitions to formula metadata
        for formula in formulas:
            var_defs = {}
            for var in formula.variables:
                if var in def_lookup:
                    var_def = def_lookup[var]
                    var_defs[var] = {
                        'name': var_def.name,
                        'unit': var_def.unit,
                        'description': var_def.description
                    }
                    
            if var_defs:
                formula.metadata['variable_definitions'] = var_defs
                
    def parse_calculation(self, formula: ExtractedFormula) -> Optional[Dict[str, any]]:
        """Parse a calculation formula into components"""
        if formula.formula_type != FormulaType.CALCULATION:
            return None
            
        # Try to parse simple calculations
        calc_pattern = r'(\d+(?:\.\d+)?)\s*([*×/+-])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'
        match = re.search(calc_pattern, formula.normalized_text)
        
        if match:
            return {
                'operand1': float(match.group(1)),
                'operator': match.group(2),
                'operand2': float(match.group(3)),
                'result': float(match.group(4))
            }
            
        return None
        
    def extract_formula_dependencies(self, formulas: List[ExtractedFormula]) -> Dict[str, Set[str]]:
        """Extract dependencies between formulas based on shared variables"""
        dependencies = {}
        
        # Build variable to formula mapping
        var_to_formulas = {}
        for formula in formulas:
            for var in formula.variables:
                if var not in var_to_formulas:
                    var_to_formulas[var] = set()
                var_to_formulas[var].add(formula.formula_id)
                
        # Find dependencies
        for formula in formulas:
            deps = set()
            for var in formula.variables:
                # Add all formulas that share this variable
                deps.update(var_to_formulas.get(var, set()))
                
            # Remove self-reference
            deps.discard(formula.formula_id)
            
            if deps:
                dependencies[formula.formula_id] = deps
                
        return dependencies