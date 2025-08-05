"""
Rule Engine Module
Extracts and manages conditional logic and business rules from documents
"""
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

from loguru import logger


class RuleType(Enum):
    """Types of rules that can be extracted"""
    CONDITIONAL = "conditional"  # if-then rules
    REQUIREMENT = "requirement"  # must/shall statements
    CONSTRAINT = "constraint"    # limits and bounds
    CALCULATION = "calculation"  # computational rules
    VALIDATION = "validation"    # compliance checks
    REFERENCE = "reference"      # lookup rules


class ConditionOperator(Enum):
    """Logical operators for conditions"""
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IN = "IN"
    CONTAINS = "CONTAINS"


@dataclass
class Condition:
    """Represents a logical condition"""
    variable: str
    operator: ConditionOperator
    value: Union[str, float, List[str]]
    unit: Optional[str] = None
    
    def to_dict(self) -> Dict[str, any]:
        return {
            'variable': self.variable,
            'operator': self.operator.value,
            'value': self.value,
            'unit': self.unit
        }
        
    def evaluate(self, variables: Dict[str, any]) -> bool:
        """Evaluate condition against variable values"""
        if self.variable not in variables:
            return False
            
        var_value = variables[self.variable]
        
        try:
            if self.operator == ConditionOperator.EQUAL:
                return var_value == self.value
            elif self.operator == ConditionOperator.NOT_EQUAL:
                return var_value != self.value
            elif self.operator == ConditionOperator.GREATER:
                return float(var_value) > float(self.value)
            elif self.operator == ConditionOperator.GREATER_EQUAL:
                return float(var_value) >= float(self.value)
            elif self.operator == ConditionOperator.LESS:
                return float(var_value) < float(self.value)
            elif self.operator == ConditionOperator.LESS_EQUAL:
                return float(var_value) <= float(self.value)
            elif self.operator == ConditionOperator.IN:
                return var_value in self.value
            elif self.operator == ConditionOperator.CONTAINS:
                return str(self.value).lower() in str(var_value).lower()
            else:
                return False
        except (ValueError, TypeError):
            return False


@dataclass
class Action:
    """Represents an action to be taken"""
    action_type: str  # 'assign', 'calculate', 'reference', 'validate'
    target: str
    expression: str
    unit: Optional[str] = None
    
    def to_dict(self) -> Dict[str, any]:
        return {
            'action_type': self.action_type,
            'target': self.target,
            'expression': self.expression,
            'unit': self.unit
        }


@dataclass
class ExtractedRule:
    """Represents an extracted business rule"""
    rule_id: str
    rule_type: RuleType
    description: str
    conditions: List[Condition] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    source_section: Optional[str] = None
    source_text: str = ""
    priority: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, any]:
        return {
            'rule_id': self.rule_id,
            'rule_type': self.rule_type.value,
            'description': self.description,
            'conditions': [c.to_dict() for c in self.conditions],
            'actions': [a.to_dict() for a in self.actions],
            'source_section': self.source_section,
            'source_text': self.source_text,
            'priority': self.priority,
            'tags': self.tags,
            'metadata': self.metadata
        }
        
    def evaluate(self, variables: Dict[str, any]) -> Tuple[bool, List[Action]]:
        """
        Evaluate rule against given variables
        
        Returns:
            Tuple of (conditions_met, applicable_actions)
        """
        if not self.conditions:
            return True, self.actions
            
        # Evaluate all conditions (AND logic for now)
        conditions_met = all(condition.evaluate(variables) for condition in self.conditions)
        
        return conditions_met, self.actions if conditions_met else []


class RuleEngine:
    """Extracts and manages business rules from technical documents"""
    
    def __init__(self):
        # Patterns for different rule types
        self.rule_patterns = {
            RuleType.CONDITIONAL: [
                # "If X then Y"
                r'[Ii]f\s+([^,]+),?\s+then\s+([^.]+)',
                # "When X, Y"
                r'[Ww]hen\s+([^,]+),\s*([^.]+)',
                # "Where X, Y shall be"
                r'[Ww]here\s+([^,]+),\s*([^.]+)\s+shall\s+be\s+([^.]+)'
            ],
            
            RuleType.REQUIREMENT: [
                # "X shall/must be Y"
                r'([^.]+)\s+(?:shall|must)\s+(?:be\s+)?([^.]+)',
                # "It is required that X"
                r'[Ii]t\s+is\s+required\s+that\s+([^.]+)',
                # "X is mandatory"
                r'([^.]+)\s+is\s+mandatory'
            ],
            
            RuleType.CONSTRAINT: [
                # "X shall not exceed Y"
                r'([^.]+)\s+shall\s+not\s+exceed\s+([^.]+)',
                # "Maximum X is Y"
                r'[Mm]aximum\s+([^.]+)\s+is\s+([^.]+)',
                # "Minimum X is Y"
                r'[Mm]inimum\s+([^.]+)\s+is\s+([^.]+)',
                # "X ≤ Y" or "X <= Y"
                r'([^≤<]+)\s*[≤<]\s*([^.]+)',
                # "X ≥ Y" or "X >= Y"
                r'([^≥>]+)\s*[≥>]\s*([^.]+)'
            ],
            
            RuleType.CALCULATION: [
                # "X = Y * Z"
                r'([A-Za-z_]\w*)\s*=\s*([^.]+)',
                # "Calculate X as Y"
                r'[Cc]alculate\s+([^.]+)\s+as\s+([^.]+)',
                # "X is determined by Y"
                r'([^.]+)\s+is\s+determined\s+by\s+([^.]+)'
            ],
            
            RuleType.VALIDATION: [
                # "Verify that X"
                r'[Vv]erify\s+that\s+([^.]+)',
                # "Check whether X"
                r'[Cc]heck\s+(?:whether|that)\s+([^.]+)',
                # "Ensure X"
                r'[Ee]nsure\s+(?:that\s+)?([^.]+)'
            ],
            
            RuleType.REFERENCE: [
                # "Refer to Table X for Y"
                r'[Rr]efer\s+to\s+([^.]+)\s+for\s+([^.]+)',
                # "See Section X"
                r'[Ss]ee\s+[Ss]ection\s+([^.]+)',
                # "In accordance with X"
                r'[Ii]n\s+accordance\s+with\s+([^.]+)'
            ]
        }
        
        # Condition patterns
        self.condition_patterns = [
            # "X > Y"
            (r'([^>]+)\s*>\s*([^,\s]+)', ConditionOperator.GREATER),
            # "X >= Y"
            (r'([^≥]+)\s*≥\s*([^,\s]+)', ConditionOperator.GREATER_EQUAL),
            # "X < Y"
            (r'([^<]+)\s*<\s*([^,\s]+)', ConditionOperator.LESS),
            # "X <= Y"
            (r'([^≤]+)\s*≤\s*([^,\s]+)', ConditionOperator.LESS_EQUAL),
            # "X equals Y"
            (r'([^.]+)\s+equals?\s+([^,\s]+)', ConditionOperator.EQUAL),
            # "X is Y"
            (r'([^.]+)\s+is\s+([^,\s]+)', ConditionOperator.EQUAL)
        ]
        
        self.extracted_rules = []
        
    def extract_rules(self, text: str, section_info: Optional[Dict[str, any]] = None) -> List[ExtractedRule]:
        """
        Extract rules from text
        
        Args:
            text: Text to extract rules from
            section_info: Optional section information
            
        Returns:
            List of extracted rules
        """
        rules = []
        rule_count = 0
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            # Try each rule type
            for rule_type, patterns in self.rule_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    
                    for match in matches:
                        rule_count += 1
                        
                        # Extract rule components
                        rule = self._parse_rule_match(
                            match, rule_type, sentence, rule_count, section_info
                        )
                        
                        if rule:
                            rules.append(rule)
                            
        # Store extracted rules
        self.extracted_rules.extend(rules)
        
        logger.info(f"Extracted {len(rules)} rules from text")
        return rules
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for rule extraction"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
        
    def _parse_rule_match(self, 
                         match: re.Match,
                         rule_type: RuleType,
                         sentence: str,
                         rule_count: int,
                         section_info: Optional[Dict[str, any]]) -> Optional[ExtractedRule]:
        """Parse a regex match into a rule"""
        try:
            groups = match.groups()
            
            if rule_type == RuleType.CONDITIONAL:
                return self._parse_conditional_rule(groups, sentence, rule_count, section_info)
            elif rule_type == RuleType.REQUIREMENT:
                return self._parse_requirement_rule(groups, sentence, rule_count, section_info)
            elif rule_type == RuleType.CONSTRAINT:
                return self._parse_constraint_rule(groups, sentence, rule_count, section_info)
            elif rule_type == RuleType.CALCULATION:
                return self._parse_calculation_rule(groups, sentence, rule_count, section_info)
            elif rule_type == RuleType.VALIDATION:
                return self._parse_validation_rule(groups, sentence, rule_count, section_info)
            elif rule_type == RuleType.REFERENCE:
                return self._parse_reference_rule(groups, sentence, rule_count, section_info)
                
        except Exception as e:
            logger.warning(f"Error parsing rule: {e}")
            
        return None
        
    def _parse_conditional_rule(self, groups: Tuple[str, ...], sentence: str, 
                               rule_count: int, section_info: Optional[Dict]) -> ExtractedRule:
        """Parse conditional (if-then) rule"""
        if len(groups) >= 2:
            condition_text = groups[0].strip()
            action_text = groups[1].strip()
            
            # Parse condition
            conditions = self._parse_conditions(condition_text)
            
            # Parse action
            actions = self._parse_actions(action_text)
            
            rule = ExtractedRule(
                rule_id=f"rule_{rule_count}",
                rule_type=RuleType.CONDITIONAL,
                description=f"If {condition_text}, then {action_text}",
                conditions=conditions,
                actions=actions,
                source_section=section_info.get('section_number') if section_info else None,
                source_text=sentence
            )
            
            return rule
            
        return None
        
    def _parse_requirement_rule(self, groups: Tuple[str, ...], sentence: str,
                               rule_count: int, section_info: Optional[Dict]) -> ExtractedRule:
        """Parse requirement rule"""
        if len(groups) >= 1:
            subject = groups[0].strip() if len(groups) > 1 else "system"
            requirement = groups[-1].strip()
            
            action = Action(
                action_type="validate",
                target=subject,
                expression=requirement
            )
            
            rule = ExtractedRule(
                rule_id=f"rule_{rule_count}",
                rule_type=RuleType.REQUIREMENT,
                description=f"{subject} shall {requirement}",
                actions=[action],
                source_section=section_info.get('section_number') if section_info else None,
                source_text=sentence,
                tags=["mandatory"]
            )
            
            return rule
            
        return None
        
    def _parse_constraint_rule(self, groups: Tuple[str, ...], sentence: str,
                              rule_count: int, section_info: Optional[Dict]) -> ExtractedRule:
        """Parse constraint rule"""
        if len(groups) >= 2:
            variable = groups[0].strip()
            constraint_value = groups[1].strip()
            
            # Determine operator from sentence
            operator = ConditionOperator.LESS_EQUAL
            if "maximum" in sentence.lower() or "not exceed" in sentence.lower():
                operator = ConditionOperator.LESS_EQUAL
            elif "minimum" in sentence.lower():
                operator = ConditionOperator.GREATER_EQUAL
                
            # Extract numeric value and unit
            value, unit = self._extract_value_and_unit(constraint_value)
            
            condition = Condition(
                variable=variable,
                operator=operator,
                value=value,
                unit=unit
            )
            
            rule = ExtractedRule(
                rule_id=f"rule_{rule_count}",
                rule_type=RuleType.CONSTRAINT,
                description=f"{variable} constraint: {constraint_value}",
                conditions=[condition],
                source_section=section_info.get('section_number') if section_info else None,
                source_text=sentence,
                tags=["constraint"]
            )
            
            return rule
            
        return None
        
    def _parse_calculation_rule(self, groups: Tuple[str, ...], sentence: str,
                               rule_count: int, section_info: Optional[Dict]) -> ExtractedRule:
        """Parse calculation rule"""
        if len(groups) >= 2:
            target = groups[0].strip()
            expression = groups[1].strip()
            
            action = Action(
                action_type="calculate",
                target=target,
                expression=expression
            )
            
            rule = ExtractedRule(
                rule_id=f"rule_{rule_count}",
                rule_type=RuleType.CALCULATION,
                description=f"Calculate {target} = {expression}",
                actions=[action],
                source_section=section_info.get('section_number') if section_info else None,
                source_text=sentence,
                tags=["calculation"]
            )
            
            return rule
            
        return None
        
    def _parse_validation_rule(self, groups: Tuple[str, ...], sentence: str,
                              rule_count: int, section_info: Optional[Dict]) -> ExtractedRule:
        """Parse validation rule"""
        if len(groups) >= 1:
            validation_text = groups[0].strip()
            
            action = Action(
                action_type="validate",
                target="system",
                expression=validation_text
            )
            
            rule = ExtractedRule(
                rule_id=f"rule_{rule_count}",
                rule_type=RuleType.VALIDATION,
                description=f"Validate: {validation_text}",
                actions=[action],
                source_section=section_info.get('section_number') if section_info else None,
                source_text=sentence,
                tags=["validation"]
            )
            
            return rule
            
        return None
        
    def _parse_reference_rule(self, groups: Tuple[str, ...], sentence: str,
                             rule_count: int, section_info: Optional[Dict]) -> ExtractedRule:
        """Parse reference rule"""
        if len(groups) >= 1:
            reference = groups[0].strip()
            purpose = groups[1].strip() if len(groups) > 1 else "information"
            
            action = Action(
                action_type="reference",
                target=purpose,
                expression=reference
            )
            
            rule = ExtractedRule(
                rule_id=f"rule_{rule_count}",
                rule_type=RuleType.REFERENCE,
                description=f"Reference {reference} for {purpose}",
                actions=[action],
                source_section=section_info.get('section_number') if section_info else None,
                source_text=sentence,
                tags=["reference"]
            )
            
            return rule
            
        return None
        
    def _parse_conditions(self, condition_text: str) -> List[Condition]:
        """Parse conditions from text"""
        conditions = []
        
        for pattern, operator in self.condition_patterns:
            matches = re.finditer(pattern, condition_text, re.IGNORECASE)
            
            for match in matches:
                variable = match.group(1).strip()
                value_text = match.group(2).strip()
                
                value, unit = self._extract_value_and_unit(value_text)
                
                condition = Condition(
                    variable=variable,
                    operator=operator,
                    value=value,
                    unit=unit
                )
                conditions.append(condition)
                
        return conditions
        
    def _parse_actions(self, action_text: str) -> List[Action]:
        """Parse actions from text"""
        # Simple action parsing - can be enhanced
        action = Action(
            action_type="assign",
            target="result",
            expression=action_text
        )
        
        return [action]
        
    def _extract_value_and_unit(self, text: str) -> Tuple[Union[float, str], Optional[str]]:
        """Extract numeric value and unit from text"""
        # Pattern for number with optional unit
        pattern = r'([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z°³²¹⁰⁴⁵⁶⁷⁸⁹/\-\*\^]+)?'
        match = re.search(pattern, text)
        
        if match:
            try:
                value = float(match.group(1))
                unit = match.group(2).strip() if match.group(2) else None
                return value, unit
            except ValueError:
                pass
                
        # Return as string if not numeric
        return text.strip(), None
        
    def evaluate_rules(self, variables: Dict[str, any]) -> List[Tuple[ExtractedRule, List[Action]]]:
        """
        Evaluate all rules against given variables
        
        Args:
            variables: Dictionary of variable values
            
        Returns:
            List of (rule, applicable_actions) tuples
        """
        applicable_rules = []
        
        for rule in self.extracted_rules:
            conditions_met, actions = rule.evaluate(variables)
            
            if conditions_met:
                applicable_rules.append((rule, actions))
                
        return applicable_rules
        
    def find_rules_by_type(self, rule_type: RuleType) -> List[ExtractedRule]:
        """Find all rules of a specific type"""
        return [rule for rule in self.extracted_rules if rule.rule_type == rule_type]
        
    def find_rules_by_variable(self, variable: str) -> List[ExtractedRule]:
        """Find all rules that involve a specific variable"""
        matching_rules = []
        
        for rule in self.extracted_rules:
            # Check conditions
            for condition in rule.conditions:
                if condition.variable.lower() == variable.lower():
                    matching_rules.append(rule)
                    break
                    
            # Check actions
            for action in rule.actions:
                if action.target.lower() == variable.lower():
                    matching_rules.append(rule)
                    break
                    
        return matching_rules
        
    def get_rule_statistics(self) -> Dict[str, any]:
        """Get statistics about extracted rules"""
        total_rules = len(self.extracted_rules)
        
        type_counts = {}
        for rule_type in RuleType:
            type_counts[rule_type.value] = len(self.find_rules_by_type(rule_type))
            
        return {
            'total_rules': total_rules,
            'rule_types': type_counts,
            'rules_with_conditions': len([r for r in self.extracted_rules if r.conditions]),
            'rules_with_actions': len([r for r in self.extracted_rules if r.actions])
        }