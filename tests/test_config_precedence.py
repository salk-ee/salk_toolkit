#!/usr/bin/env python3
"""Simple test to verify config precedence without full imports"""

import logging
from copy import deepcopy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def deep_merge(target, source):
    """Recursively merge source dict into target dict."""
    for k, v in source.items():
        if isinstance(v, dict) and k in target and isinstance(target[k], dict):
            deep_merge(target[k], v)
        else:
            target[k] = v
    return target


# Simulate the configs
_altair_base_config = {
    "legend": {
        "columns": None,  # No default
        "labelFontSize": 14,
    }
}

# This is what gets loaded from altair_custom_config.json
altair_custom_config = {
    "legend": {
        "columns": 1,  # User wants 1 column
        "labelFontSize": 10,
    }
}

# This is what comes from a plot (e.g., from plot-specific legend config)
plot_config = {
    "legend": {
        "columns": 2,  # Plot specifies 2 columns
    }
}

logger.info("Testing config precedence...")
logger.info(f"Base config legend: {_altair_base_config['legend']}")
logger.info(f"Plot config legend: {plot_config['legend']}")
logger.info(f"Custom config legend: {altair_custom_config['legend']}")
logger.info("")

# OLD WAY (WRONG) - merge custom into default at module load
logger.info("=== OLD WAY (WRONG) ===")
altair_default_config_old = deepcopy(_altair_base_config)
deep_merge(altair_default_config_old, altair_custom_config)
logger.info(f"After module load, altair_default_config: {altair_default_config_old['legend']}")

# Then in plot_matrix_html, start with default and merge plot config
full_config_old = deepcopy(altair_default_config_old)
deep_merge(full_config_old, plot_config)
logger.info(f"After merging plot config: {full_config_old['legend']}")
logger.info(f"Result: columns = {full_config_old['legend']['columns']} (WRONG! Should be 1)")
logger.info("")

# NEW WAY (CORRECT) - keep custom separate, apply last
logger.info("=== NEW WAY (CORRECT) ===")
altair_default_config_new = deepcopy(_altair_base_config)
logger.info(f"altair_default_config (no custom merged): {altair_default_config_new['legend']}")

# In plot_matrix_html: base -> plot -> custom
full_config_new = deepcopy(_altair_base_config)
deep_merge(full_config_new, plot_config)
logger.info(f"After merging plot config: {full_config_new['legend']}")
deep_merge(full_config_new, altair_custom_config)
logger.info(f"After merging custom config: {full_config_new['legend']}")
logger.info(f"Result: columns = {full_config_new['legend']['columns']} (CORRECT! Custom overrides plot)")
logger.info("")

# Verify
assert full_config_old["legend"]["columns"] == 2, "Old way should have columns=2"
assert full_config_new["legend"]["columns"] == 1, "New way should have columns=1"
assert full_config_new["legend"]["labelFontSize"] == 10, "Custom labelFontSize should be applied"

logger.info("✅ All assertions passed! The fix is correct.")
