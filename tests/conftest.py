"""
pytest configuration for salk_toolkit tests.
"""

import pytest
import sys
from pathlib import Path

# Add the parent directory to sys.path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

def pytest_addoption(parser):
    parser.addoption("--recompute", action="store_true", default=False, 
                        help="Recompute reference data for plots etc")

@pytest.fixture
def recompute(request):
    """Fixture to provide the recompute flag value."""
    return request.config.getoption("--recompute")

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
