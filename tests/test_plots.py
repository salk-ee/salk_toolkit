"""
Comprehensive tests for all plot types in salk_toolkit.plots module.
Tests all e2e_plot configurations found in 03_plots.ipynb notebook.
"""

import pytest
import pandas as pd
import altair as alt
import os
import json
import sys
from pathlib import Path
import numpy as np
import random

# Add tests directory to path for utility imports
sys.path.insert(0, str(Path(__file__).parent))

# Disable Altair max rows limit for testing
alt.data_transformers.disable_max_rows()

# Import the plotting functions
from salk_toolkit.pp import e2e_plot
from utils.plot_comparison import compare_json_with_tolerance, normalize_chart_json, pretty_print_json_differences
from salk_toolkit.io import read_json


class TestPlots:
    """Test class for all plot types in the salk_toolkit plotting system."""
    
    @classmethod
    def setup_class(cls):
        """Set up test data and common configurations."""
        # Get the path to the test data file
        cls.data_file = Path(__file__).parent / 'data' / 'master_bootstrap.parquet'
        cls.data_meta_file = Path(__file__).parent / 'data' / 'master_meta.json'

        # Reference files directory
        cls.reference_dir = Path(__file__).parent / 'reference_plots'
        cls.reference_dir.mkdir(exist_ok=True)
        
        # Load metadata if available
        cls.data_meta = None
        if cls.data_meta_file.exists():
            cls.data_meta = read_json(str(cls.data_meta_file))
        
        # Common test parameters
        cls.common_kwargs = {
            'width': 400,  # Smaller width for faster testing
        }

    def _run_plot_test(self, test_name, config, data_file=None, recompute=False, float_tolerance=5e-4, **kwargs):
        """Run a plot test and compare against reference JSON."""
        if data_file is None:
            data_file = str(self.data_file)
        
        # Merge common kwargs with test-specific kwargs
        test_kwargs = {**self.common_kwargs, **kwargs}
        
        # Get reference file path
        reference_file = self.reference_dir / f"{test_name}.json"
        
        # Run the plot
        result = e2e_plot(config, data_file, **test_kwargs)
        
        # Verify the result is an Altair chart
        assert isinstance(result, (alt.Chart, alt.LayerChart, alt.VConcatChart, alt.HConcatChart, alt.FacetChart)), \
            f"Expected Altair chart, got {type(result)}"
        
        # Normalize the chart JSON for comparison
        normalized_result = normalize_chart_json(result.to_json())

        already_exists = reference_file.exists()
        if recompute:
            # Save/update the reference
            with open(reference_file, 'w') as f:
                json.dump(normalized_result, f, indent=2, sort_keys=True)
            print(f"{'Updated' if already_exists else 'Created'} reference for test {test_name}")
        elif not already_exists:
            raise ValueError(f"Reference file for {test_name} not found")
        else:
            # Compare against reference
            with open(reference_file, 'r') as f:
                reference_json = json.load(f)
            
            # Compare with floating point tolerance
            if not compare_json_with_tolerance(reference_json, normalized_result, float_tolerance):
                # Generate detailed difference report
                diff_report = pretty_print_json_differences(reference_json, normalized_result, float_tolerance)
                assert False, \
                    f"Plot output differs from reference. Test: {test_name}\n" \
                    f"Reference file: {reference_file}\n\n" \
                    f"{diff_report}"
        
        return result

    def test_boxplots_basic(self, recompute):
        """Test basic boxplots."""
        config = {
            'res_col': 'party_preference',
            'factor_cols': ['age_group'],
            'filter': {},
            'plot': 'boxplots',
            'internal_facet': True,
        }
        self._run_plot_test("test_boxplots_basic", config, recompute=recompute)

    def test_boxplots_raw(self, recompute):
        """Test raw boxplots."""
        config = {
            'res_col': 'EKRE',
            'factor_cols': ['age_group'],
            'filter': {},
            'plot': 'boxplots-raw',
            'internal_facet': True
        }
        self._run_plot_test("test_boxplots_raw", config, recompute=recompute)

    def test_columns_basic(self, recompute):
        """Test basic column plots."""
        config = {
            'res_col': 'e-valimised',
            'factor_cols': ['nationality'],
            'filter': {},
            'plot': 'columns',
            'internal_facet': True
        }
        self._run_plot_test("test_columns_basic", config, recompute=recompute)

    def test_columns_thermometer(self, recompute):
        """Test column plots with thermometer data."""
        config = {
            'res_col': 'thermometer',
            'factor_cols': ['nationality'],
            'filter': {},
            'plot': 'columns',
            'internal_facet': True
        }
        self._run_plot_test("test_columns_thermometer", config, recompute=recompute)

    def test_stacked_columns(self, recompute):
        """Test stacked column plots."""
        config = {
            "res_col": "party_preference",
            "factor_cols": ["EKRE"],
            "internal_facet": True,
            "plot": "stacked_columns",
            "plot_args": {
                "normalized": False
            },
            "filter": {}
        }
        self._run_plot_test("test_stacked_columns", config, recompute=recompute)

    def test_diff_columns(self, recompute):
        """Test difference column plots."""
        config = {
            'res_col': 'thermometer',
            'factor_cols': ['age_group'],
            'filter': {'age_group': [None, '25-34', '35-44']},
            'plot': 'diff_columns',
            'internal_facet': True,
            'plot_args': {'sort_descending': True}
        }
        self._run_plot_test("test_diff_columns", config, recompute=recompute)

    def test_massplot(self, recompute):
        """Test mass plot."""
        config = {
            'res_col': 'trust',
            'factor_cols': ['party_preference'],
            'filter': {},
            'plot': 'massplot',
            'internal_facet': True,
            'convert_res': 'continuous'
        }
        self._run_plot_test("test_massplot", config, recompute=recompute)

    def test_likert_bars_basic(self, recompute):
        """Test basic Likert bars."""
        config = {
            'res_col': 'trust',
            'factor_cols': ['party_preference'],
            'filter': {},
            'plot': 'likert_bars',
            'internal_facet': True
        }
        self._run_plot_test("test_likert_bars_basic", config, recompute=recompute)

    def test_likert_bars_no_factors(self, recompute):
        """Test Likert bars with no factor columns."""
        config = {
            'res_col': 'valitsus',
            'factor_cols': [],
            'filter': {},
            'plot': 'likert_bars',
            'internal_facet': True
        }
        self._run_plot_test("test_likert_bars_no_factors", config, recompute=recompute)

    def test_density_raw_stacked(self, recompute):
        """Test raw density plot with stacking."""
        config = {
            'res_col': 'thermometer',
            'factor_cols': ['party_preference'],
            'filter': {},
            'plot': 'density-raw',
            'plot_args': {'stacked': True},
            'internal_facet': True
        }
        self._run_plot_test("test_density_raw_stacked", config, recompute=recompute, width=500)

    def test_violin_raw(self, recompute):
        """Test raw violin plot."""
        config = {
            'res_col': 'thermometer',
            'factor_cols': ['party_preference'],
            'filter': {},
            'plot': 'violin-raw',
            'internal_facet': True
        }
        self._run_plot_test("test_violin_raw", config, recompute=recompute, width=650)

    def test_matrix_basic(self, recompute):
        """Test basic matrix plot."""
        config = {
            'res_col': 'party_preference',
            'factor_cols': ['age_group'],
            'filter': {},
            'plot': 'matrix',
            'internal_facet': True
        }
        self._run_plot_test("test_matrix_basic", config, recompute=recompute)

    def test_matrix_thermometer(self, recompute):
        """Test matrix plot with thermometer data."""
        config = {
            'res_col': 'thermometer',
            'factor_cols': ['party_preference'],
            'filter': {},
            'plot': 'matrix',
            'internal_facet': True
        }
        self._run_plot_test("test_matrix_thermometer", config, recompute=recompute)

    def test_matrix_with_reorder(self, recompute):
        """Test matrix plot with reordering."""
        config = {
            'res_col': 'issues',
            'factor_cols': ['party_preference'],
            'filter': {},
            'plot': 'matrix',
            'plot_args': {'reorder': True},
            'internal_facet': True,
            'convert_res': 'continuous',
            'cont_transform': 'center'
        }
        self._run_plot_test("test_matrix_with_reorder", config, recompute=recompute)

    def test_corr_matrix(self, recompute):
        """Test correlation matrix plot."""
        config = {
            'res_col': 'thermometer',
            'factor_cols': [],
            'filter': {},
            'plot': 'corr_matrix',
            'internal_facet': True
        }
        self._run_plot_test("test_corr_matrix", config, recompute=recompute)

    def test_lines_smooth(self, recompute):
        """Test line plot with smoothing."""
        config = {
            'res_col': 'thermometer',
            'factor_cols': ['education'],
            'filter': {},
            'plot': 'lines',
            'internal_facet': True,
            'plot_args': {'smooth': True}
        }
        self._run_plot_test("test_lines_smooth", config, recompute=recompute)

    def test_lines_hdi_basic(self, recompute):
        """Test HDI line plot."""
        config = {
            'res_col': 'party_preference',
            'factor_cols': ['age_group', 'nationality'],
            'filter': {},
            'plot': 'lines_hdi',
            'internal_facet': True,
        }
        self._run_plot_test("test_lines_hdi_basic", config, recompute=recompute)

    def test_area_smooth(self, recompute):
        """Test smooth area plot."""
        config = {
            'res_col': 'party_preference',
            'factor_cols': ['education', 'gender'],
            'filter': {},
            'plot': 'area_smooth',
            'internal_facet': True
        }
        self._run_plot_test("test_area_smooth", config, recompute=recompute)

    def test_likert_rad_pol(self, recompute):
        """Test Likert radicalization/polarization plot."""
        config = {
            "res_col": "referendum",
            "factor_cols": [
                "electoral_district"
            ],
            "internal_facet": True,
            "plot": "likert_rad_pol",
            "filter": {}
        }
        # This test uses a different data file
        self._run_plot_test("test_likert_rad_pol", config, recompute=recompute, width=600)

    def test_barbell(self, recompute):
        """Test barbell plot."""
        config = {
            'res_col': 'trust',
            'factor_cols': ['party_preference'],
            'filter': {},
            'plot': 'barbell',
            'internal_facet': True,
            'convert_res': 'continuous'
        }
        self._run_plot_test("test_barbell", config, recompute=recompute)

    def test_geoplot(self, recompute):
        """Test geographic plot."""
        if self.data_meta is None:
            pytest.skip("Data metadata not available for geoplot test")
        
        config = {
            'res_col': 'EKRE',
            'factor_cols': ['electoral_district'],
            'filter': {},
            'plot': 'geoplot',
            'internal_facet': True
        }
        self._run_plot_test("test_geoplot", config, recompute=recompute, data_meta=self.data_meta, width=400)

    def test_facet_dist(self, recompute):
        """Test facet distribution plot."""
        config = {
            'res_col': 'thermometer',
            'factor_cols': ['party_preference'],
            'filter': {},
            'plot': 'facet_dist',
            'internal_facet': True
        }
        self._run_plot_test("test_facet_dist", config, recompute=recompute, float_tolerance=3e-2)

    def test_max_diff(self, recompute):
        """Test max_diff plot."""
        config = {
                'factor_cols': ['question'],
                'filter': {},
                'internal_facet': True,
                'plot': 'maxdiff',
                'plot_args': {},
                'res_col': 'thermometer'
        }
        self._run_plot_test("test_maxdiff", config, recompute=recompute, float_tolerance=3e-2)

    @pytest.mark.skip(reason="Very hard to make deterministic (tried but failed)")
    def test_ordered_population(self, recompute):
        """Test ordered population plot."""
        config = {
            'res_col': 'thermometer',
            'factor_cols': ['party_preference'],
            'filter': {},
            'plot': 'ordered_population',
            'internal_facet': True,
            'plot_args': {'full_data': True} # required for consistency
        }
        self._run_plot_test("test_ordered_population", config, recompute=recompute, width=800)

    def test_marimekko(self, recompute):
        """Test Marimekko plot."""
        config = {
            'res_col': 'trust',
            'factor_cols': ['question', 'party_preference'],
            'filter': {},
            'plot': 'marimekko',
            'internal_facet': True,
        }
        self._run_plot_test("test_marimekko", config, recompute=recompute)

    def test_boxplots_convert_res(self, recompute):
        """Test boxplots with result conversion."""
        config = {
            'res_col': 'age_group',
            'factor_cols': ['gender'],
            'filter': {},
            'plot': 'boxplots',
            'internal_facet': True,
            'convert_res': 'continuous'
        }
        self._run_plot_test("test_boxplots_convert_res", config, recompute=recompute)

    def test_lines_hdi_with_num_values(self, recompute):
        """Test HDI lines with numerical values and custom formatting."""
        config = {
            'res_col': 'valitsus',
            'factor_cols': ['party_preference', 'age_group'],
            # Isamaa HDI has multiple close options so excluding it
            'filter': { 'party_preference': ['Keskerakond','Reformierakond', 'SDE', 'EKRE']}, 
            'convert_res': 'continuous',
            'num_values': [0, 0, 0, 1, 1],
            'value_name': 'Pr[valitsus==Agree]',
            'value_format': '.0%',
            'plot': 'lines_hdi',
            'internal_facet': True,
        }
        self._run_plot_test("test_lines_hdi_with_num_values", config, recompute=recompute, width=800)

    def test_likert_bars_with_metadata(self, recompute):
        """Test Likert bars with metadata."""
        config = {
            'res_col': 'trust',
            'factor_cols': ['gender', 'age_group'],
            'filter': {},
            'plot': 'likert_bars',
            'internal_facet': True
        }
        kwargs = {'width': 800}
        if self.data_meta is not None:
            kwargs['data_meta'] = self.data_meta
        
        self._run_plot_test("test_likert_bars_with_metadata", config, recompute=recompute, **kwargs)

    def test_plot_with_missing_data_file(self, recompute):
        """Test that appropriate error is raised for missing data file."""
        config = {
            'res_col': 'party_preference',
            'factor_cols': ['age_group'],
            'filter': {},
            'plot': 'boxplots',
            'internal_facet': True,
        }
        
        with pytest.raises((FileNotFoundError, OSError)):
            self._run_plot_test('test_missing_data_file', config, data_file='nonexistent_file.parquet', recompute=recompute)

    def test_plot_with_invalid_column(self, recompute):
        """Test behavior with invalid column names."""
        config = {
            'res_col': 'nonexistent_column',
            'factor_cols': ['age_group'],
            'filter': {},
            'plot': 'boxplots',
            'internal_facet': True,
        }
        
        # This should either raise an error or handle gracefully
        # The exact behavior depends on the implementation
        try:
            self._run_plot_test('test_plot_with_invalid_column', config, recompute=recompute)
        except (KeyError, ValueError, Exception):
            # Expected behavior for invalid columns
            pass

class TestPlotUtilities:
    """Test utility functions used in plotting."""
    
    def test_estimate_legend_columns_horiz_naive(self):
        """Test the naive legend column estimation function."""
        from salk_toolkit.plots import estimate_legend_columns_horiz_naive
        
        cats = ["Keskerakond", "EKRE", "Reformierakond", "Isamaa", "SDE", "Rohelised"]
        result = estimate_legend_columns_horiz_naive(cats, 800)
        assert isinstance(result, int)
        assert result > 0
        assert result <= len(cats)

    def test_estimate_legend_columns_horiz(self):
        """Test the sophisticated legend column estimation function."""
        from salk_toolkit.plots import estimate_legend_columns_horiz
        
        cats = ["Keskerakond", "EKRE", "Reformierakond", "Isamaa", "SDE", 
                "Rohelised", "Eesti 200", "Parempoolsed", "Other", 
                "None of the parties", "No opinion"]
        result = estimate_legend_columns_horiz(cats, 800)
        assert isinstance(result, int)
        assert result > 0
        assert result <= len(cats)
        # Test the specific assertion from the notebook
        assert result == 6


if __name__ == "__main__":
    pytest.main([__file__])