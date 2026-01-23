"""Test config injection"""

import os
import sys
import logging
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Mock streamlit and utils
mock_st = MagicMock()
sys.modules["streamlit"] = mock_st
sys.modules["streamlit.components"] = MagicMock()
sys.modules["streamlit.components.v1"] = MagicMock()
sys.modules["streamlit.components.v1.components"] = MagicMock()
sys.modules["extra_streamlit_components"] = MagicMock()
sys.modules["streamlit_authenticator"] = MagicMock()
sys.modules["streamlit.logger"] = MagicMock()

import streamlit as st
import salk_toolkit.utils as utils
from salk_toolkit import dashboard

# Setup mock utils with a known config
sys.path.append(os.getcwd())
utils.altair_default_config = {"mock_global_setting": "global_value", "shared_setting": "global_shared"}

# Import dashboard to test draw_plot_matrix
# We need to mock sdb/st to avoid initialization issues if dashboard.py executes code


def test_dashboard_config_injection():
    """Testing dashboard.draw_plot_matrix config injection..."""
    logger.info("Testing dashboard.draw_plot_matrix config injection...")

    # Mock Altair chart object
    class MockChart:
        def to_dict(self):
            return {
                "mark": "bar",
                "config": {"chart_specific_setting": "chart_value", "shared_setting": "chart_override"},
            }

    mock_chart = MockChart()

    # Capture the call to vega_lite_chart
    mock_col = MagicMock()
    st.columns = MagicMock(return_value=[mock_col])

    # Run function
    dashboard.draw_plot_matrix([[mock_chart]])

    # Inspect arguments passed to vega_lite_chart
    # NOTE: If 1 column, dashboard uses [st] directly, not st.columns result
    call_args = mock_col.vega_lite_chart.call_args
    if not call_args:
        # Fallback to checking st.vega_lite_chart (mock_st)
        call_args = mock_st.vega_lite_chart.call_args

    if not call_args:
        logger.error("FAILURE: vega_lite_chart was not called on st or st.columns output")
        return

    spec = call_args.kwargs["spec"]
    config = spec.get("config", {})

    logger.info(f"Final config: {config}")

    # Verification checks
    if config.get("mock_global_setting") == "global_value":
        logger.info("SUCCESS: Global setting injected")
    else:
        logger.error("FAILURE: Global setting MISSING")

    if config.get("chart_specific_setting") == "chart_value":
        logger.info("SUCCESS: Chart specific setting preserved")
    else:
        logger.error("FAILURE: Chart specific setting LOST")

    if config.get("shared_setting") == "chart_override":
        logger.info("SUCCESS: Chart specific setting overrides global")
    else:
        logger.error(f"FAILURE: Shared setting is {config.get('shared_setting')}, expected 'chart_override'")


if __name__ == "__main__":
    test_dashboard_config_injection()
