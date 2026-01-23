import requests
import json
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Mock the altair context
class MockAltairChart:
    def __init__(self, data=None, spec=None):
        self.data = data
        self.spec = spec or {}

    def to_json(self):
        # Allow passing full dict as "spec" for testing
        if isinstance(self.data, dict):
            return json.dumps(self.data)

        # Simple structure
        return json.dumps({"data": {"values": self.data.to_dict(orient="records")}, "spec": self.spec, "width": 100})


# Import via pkg path to test actual integration
# We need to hack sys.path to run this from root
import sys
import os

sys.path.append(os.getcwd())

from salk_toolkit.data_server import LocalDataServer
from salk_toolkit.utils import plot_matrix_html


def test_named_datasets():
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

    if not html:
        logger.error("Error: HTML generation failed")
        return

    logger.info("HTML Generated. analyzing...")

    # Extract the spec from HTML
    # The template has: var specs = [...];
    match = re.search(r"var specs = (\[.*\]);", html, re.DOTALL)
    if not match:
        logger.error("Could not find specs in HTML")
        logger.error(html[:500])
        return

    specs_json = match.group(1)
    specs = json.loads(specs_json)
    spec = specs[0]

    logger.info("Spec content:")
    logger.info(json.dumps(spec, indent=2))

    # Verification checks

    # 1. "datasets" key should be gone (or empty if we didn't remove it, but we did del)
    if "datasets" in spec:
        logger.error("FAILURE: 'datasets' key still present in spec")
    else:
        logger.info("SUCCESS: 'datasets' key removed")

    # 2. Layer data should reference URL
    layer_data = spec["layer"][0]["data"]
    if "url" in layer_data:
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
            logger.error(f"FAILURE: Could not fetch data from URL: {e}")

    elif "name" in layer_data:
        logger.error(f"FAILURE: Data still references name: {layer_data['name']}")
    else:
        logger.error(f"FAILURE: Data has unexpected format: {layer_data}")


if __name__ == "__main__":
    test_named_datasets()
