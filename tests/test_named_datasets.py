"""Integration tests for named dataset handling."""

import json
import logging
import re

import pytest
import requests

from salk_toolkit.data_server import LocalDataServer
from salk_toolkit.utils import plot_matrix_html

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Mock the altair context
class MockAltairChart:
    """Mock for Altair chart object."""

    def __init__(self, data=None, spec=None):
        """Initialize mock chart."""
        self.data = data
        self.spec = spec or {}

    def to_json(self):
        """Convert chart to JSON representation."""
        # Allow passing full dict as "spec" for testing
        if isinstance(self.data, dict):
            return json.dumps(self.data)

        # Simple structure
        return json.dumps({"data": {"values": self.data.to_dict(orient="records")}, "spec": self.spec, "width": 100})


def test_named_datasets():
    """Test that named datasets are correctly extracted and served via LocalDataServer."""
    logger.info("Testing Named Datasets handling...")

    LocalDataServer.get_instance(port=8003)

    # Simulate an Altair chart with named datasets
    # Structure: top level datasets, layer referencing it
    chart_json = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "datasets": {"data-12345": [{"category": "A", "val": 28}, {"category": "B", "val": 55}]},
        "layer": [{"data": {"name": "data-12345"}, "mark": "bar"}],
        "width": 200,
    }

    chart_mock = MockAltairChart(data=chart_json)  # Passing dict triggers direct dump

    logger.info("Generating HTML...")
    html = plot_matrix_html(chart_mock, uid="test_viz", serve_data=True)

    assert html is not None, "Error: HTML generation failed"

    logger.info("HTML Generated. analyzing...")

    # Extract the spec from HTML
    # The template has: var specs = [...];
    match = re.search(r"var specs = (\[.*\]);", html, re.DOTALL)
    assert match is not None, "Could not find specs in HTML"

    specs_json = match.group(1)
    specs = json.loads(specs_json)
    spec = specs[0]

    logger.info("Spec content:")
    logger.info(json.dumps(spec, indent=2))

    # Verification checks

    # 1. "datasets" key should be gone (or empty if we didn't remove it, but we did del)
    assert "datasets" not in spec, "FAILURE: 'datasets' key still present in spec"
    logger.info("SUCCESS: 'datasets' key removed")

    # 2. Layer data should reference URL
    layer_data = spec["layer"][0]["data"]

    if "name" in layer_data:
        pytest.fail(f"FAILURE: Data still references name: {layer_data['name']}")

    assert "url" in layer_data, f"FAILURE: Data has unexpected format: {layer_data}"

    url = layer_data["url"]
    logger.info(f"SUCCESS: Data references URL: {url}")

    # Verify URL content
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Fetched {len(data)} records from URL")
        assert len(data) == 2
        assert data[0]["category"] == "A"
        logger.info("SUCCESS: Data content verified using URL")
    except Exception as e:
        pytest.fail(f"FAILURE: Could not fetch data from URL: {e}")


if __name__ == "__main__":
    test_named_datasets()
