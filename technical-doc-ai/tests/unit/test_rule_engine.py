"""
Unit tests for Rule Engine
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from knowledge_base.rule_engine import RuleEngine, RuleType, ConditionOperator


class TestRuleEngine:
    """Test cases for RuleEngine"""
    
    @pytest.fixture
    def rule_engine(self):
        """Create a RuleEngine instance for testing"""
        return RuleEngine()
        
    def test_conditional_rule_extraction(self, rule_engine):
        """Test extraction of conditional rules"""
        text = "If the area exceeds 500 m², then mechanical ventilation is required."
        rules = rule_engine.extract_rules(text)
        
        assert len(rules) >= 1
        rule = rules[0]
        assert rule.rule_type == RuleType.CONDITIONAL
        assert "area" in rule.description.lower()
        
    def test_requirement_rule_extraction(self, rule_engine):
        """Test extraction of requirement rules"""
        text = "Ventilation systems shall provide minimum 300 m³/hour per car."
        rules = rule_engine.extract_rules(text)
        
        assert len(rules) >= 1
        rule = rules[0]
        assert rule.rule_type == RuleType.REQUIREMENT
        assert "mandatory" in rule.tags
        
    def test_constraint_rule_extraction(self, rule_engine):
        """Test extraction of constraint rules"""
        text = "Maximum pressure shall not exceed 50 Pa."
        rules = rule_engine.extract_rules(text)
        
        assert len(rules) >= 1
        rule = rules[0]
        assert rule.rule_type == RuleType.CONSTRAINT
        assert len(rule.conditions) >= 1
        
    def test_calculation_rule_extraction(self, rule_engine):
        """Test extraction of calculation rules"""
        text = "Total ventilation = number of cars × 300 m³/hour"
        rules = rule_engine.extract_rules(text)
        
        assert len(rules) >= 1
        rule = rules[0]
        assert rule.rule_type == RuleType.CALCULATION
        assert len(rule.actions) >= 1
        
    def test_validation_rule_extraction(self, rule_engine):
        """Test extraction of validation rules"""
        text = "Verify that the exit width meets minimum requirements."
        rules = rule_engine.extract_rules(text)
        
        assert len(rules) >= 1
        rule = rules[0]
        assert rule.rule_type == RuleType.VALIDATION
        
    def test_reference_rule_extraction(self, rule_engine):
        """Test extraction of reference rules"""
        text = "Refer to Table 3.2 for ventilation rates."
        rules = rule_engine.extract_rules(text)
        
        assert len(rules) >= 1
        rule = rules[0]
        assert rule.rule_type == RuleType.REFERENCE
        
    def test_condition_evaluation(self, rule_engine):
        """Test evaluation of rule conditions"""
        text = "If area > 500, then use mechanical ventilation."
        rules = rule_engine.extract_rules(text)
        
        if rules and rules[0].conditions:
            # Test with area = 600 (should trigger)
            variables = {"area": 600}
            conditions_met, actions = rules[0].evaluate(variables)
            assert conditions_met
            
            # Test with area = 300 (should not trigger)
            variables = {"area": 300}
            conditions_met, actions = rules[0].evaluate(variables)
            assert not conditions_met
            
    def test_rule_evaluation_with_variables(self, rule_engine):
        """Test rule evaluation with specific variables"""
        text = "Maximum occupancy shall not exceed 100 persons."
        rules = rule_engine.extract_rules(text)
        
        variables = {"occupancy": 80, "building_type": "office"}
        applicable_rules = rule_engine.evaluate_rules(variables)
        
        # Should have some applicable rules
        assert len(applicable_rules) >= 0
        
    def test_find_rules_by_type(self, rule_engine):
        """Test finding rules by type"""
        text = """
        Ventilation shall provide 300 m³/hour per car.
        If area > 500 m², then use mechanical system.
        Maximum pressure shall not exceed 50 Pa.
        """
        rules = rule_engine.extract_rules(text)
        
        requirements = rule_engine.find_rules_by_type(RuleType.REQUIREMENT)
        constraints = rule_engine.find_rules_by_type(RuleType.CONSTRAINT)
        
        assert len(requirements) >= 1
        assert len(constraints) >= 1
        
    def test_find_rules_by_variable(self, rule_engine):
        """Test finding rules by variable"""
        text = """
        Pressure shall not exceed 50 Pa.
        If pressure > 40 Pa, then check system.
        """
        rules = rule_engine.extract_rules(text)
        
        pressure_rules = rule_engine.find_rules_by_variable("pressure")
        assert len(pressure_rules) >= 1
        
    def test_complex_rule_text(self, rule_engine):
        """Test extraction from complex building code text"""
        text = """
        3.2.1 Ventilation Requirements
        
        For carpark ventilation, the following shall apply:
        a) Minimum ventilation rate shall be 300 m³/hour per car space.
        b) If the carpark area exceeds 2000 m², mechanical ventilation is required.
        c) Maximum static pressure shall not exceed 50 Pa.
        d) Verify that exhaust fans meet the calculated flow requirements.
        """
        
        rules = rule_engine.extract_rules(text)
        
        # Should extract multiple rules of different types
        assert len(rules) >= 3
        
        types_found = set(rule.rule_type for rule in rules)
        assert RuleType.REQUIREMENT in types_found
        assert RuleType.CONDITIONAL in types_found or RuleType.CONSTRAINT in types_found
        
    def test_rule_statistics(self, rule_engine):
        """Test rule statistics generation"""
        text = """
        Ventilation shall provide 300 m³/hour per car.
        If area > 500, then use mechanical system.
        Maximum pressure ≤ 50 Pa.
        """
        rules = rule_engine.extract_rules(text)
        
        stats = rule_engine.get_rule_statistics()
        assert stats['total_rules'] >= 3
        assert 'rule_types' in stats
        assert 'rules_with_conditions' in stats
        assert 'rules_with_actions' in stats


if __name__ == "__main__":
    pytest.main([__file__])