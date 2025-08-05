"""
Pytest configuration and shared fixtures
"""
import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data files"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_technical_text():
    """Sample technical document text for testing"""
    return """
    3.2.1 Ventilation Requirements for Carparks
    
    The ventilation system shall provide adequate air movement to prevent 
    accumulation of vehicle exhaust gases.
    
    a) Minimum ventilation rate shall be 300 m³/hour per car space.
    b) If the carpark area exceeds 2000 m², mechanical ventilation is required.
    c) Maximum static pressure shall not exceed 50 Pa.
    d) For areas where Q = n × 300, where Q is total flow rate and n is number of cars.
    
    Table 3.1: Ventilation Rates
    Space Type | Rate (m³/h per car)
    Small cars | 250
    Large cars | 350
    Mixed      | 300
    
    Example calculation:
    For 6 cars: 6 × 300 = 1800 m³/hour
    """


@pytest.fixture
def sample_formulas():
    """Sample formulas for testing"""
    return [
        "Q = n × V",
        "P ≤ 50 Pa",
        "A = π × r²",
        "6 × 300 = 1800"
    ]


@pytest.fixture
def sample_rules():
    """Sample business rules for testing"""
    return [
        "If area exceeds 2000 m², then mechanical ventilation is required.",
        "Ventilation rate shall be minimum 300 m³/hour per car.",
        "Maximum pressure shall not exceed 50 Pa.",
        "Verify that exit widths meet minimum requirements."
    ]


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    return {
        "choices": [
            {
                "message": {
                    "content": '{"query_type": "calculation", "entities": {"subject": "6-car carpark"}, "parameters": {"num_cars": 6}}'
                }
            }
        ]
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Set test environment variables
    os.environ['TESTING'] = 'true'
    
    yield
    
    # Cleanup
    if 'TESTING' in os.environ:
        del os.environ['TESTING']