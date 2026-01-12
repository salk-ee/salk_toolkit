"""
Comprehensive tests for all plot types in salk_toolkit.plots module.
Tests all e2e_plot configurations defined in `salk_toolkit/plots.py`.
"""

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import altair as alt
import pandas as pd
import pytest

from salk_toolkit.election_models import mandate_plot
from salk_toolkit.io import read_json, read_parquet_with_metadata
from salk_toolkit.pp import AltairChart, FacetMeta, e2e_plot, matching_plots, PlotInput
from salk_toolkit.validation import (
    ColumnMeta,
    DataMeta,
    ElectoralSystem,
    soft_validate,
)

# Add tests directory to path for utility imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.plot_comparison import (
    compare_json_with_tolerance,
    normalize_chart_json,
    pretty_print_json_differences,
    save_plot_comparison_html,
    save_plot_matrix_comparison_html,
)

# Disable Altair max rows limit for testing
alt.data_transformers.disable_max_rows()


class TestPlots:
    """Test class for all plot types in the salk_toolkit plotting system."""

    @classmethod
    def setup_class(cls) -> None:
        """Set up test data and common configurations."""
        # Get the path to the test data file
        cls.data_file = Path(__file__).parent / "data" / "master_bootstrap.parquet"
        cls.data_meta_file = Path(__file__).parent / "data" / "master_meta.json"

        # Reference files directory
        cls.reference_dir = Path(__file__).parent / "reference_plots"
        cls.reference_dir.mkdir(exist_ok=True)

        # Load metadata if available
        cls.data_meta: DataMeta | None = None
        if cls.data_meta_file.exists():
            cls.data_meta = soft_validate(read_json(str(cls.data_meta_file)), DataMeta)

        # Common test parameters
        cls.common_kwargs = {
            "width": 400,  # Smaller width for faster testing
        }

    def _run_plot_test(
        self,
        test_name: str,
        config: Mapping[str, Any],
        data_file: str | Path | None = None,
        full_df: Any | None = None,
        data_meta: DataMeta | None = None,
        recompute: bool = False,
        float_tolerance: float = 5e-4,
        **kwargs: Any,
    ) -> alt.TopLevelMixin | list[list[AltairChart]]:
        """Run a plot test and compare against reference JSON."""
        if data_file is None and full_df is None:
            data_file = str(self.data_file)

        # Merge common kwargs with test-specific kwargs
        test_kwargs: dict[str, Any] = {**self.common_kwargs, **kwargs}

        # Load data and metadata for matching_plots call
        if full_df is not None:
            if data_meta is None:
                raise ValueError("data_meta must be provided when full_df is supplied")
            test_full_df = full_df
            test_data_meta = data_meta
        else:
            test_data_file = str(data_file)
            test_full_df, full_meta = read_parquet_with_metadata(test_data_file, lazy=True)
            if full_meta is None or full_meta.data is None:
                raise ValueError(f"Parquet file {test_data_file} has no data metadata")
            test_data_meta = full_meta.data

        # Get matching plots (details=True returns a dict)
        matches_dict = matching_plots(config, test_full_df, test_data_meta, details=True, list_hidden=True)

        if full_df is not None:
            result = e2e_plot(config, data_file=None, full_df=full_df, data_meta=data_meta, **test_kwargs)
        else:
            result = e2e_plot(config, data_file, **test_kwargs)

        self._assert_chart_matches_reference(test_name, result, matches_dict, recompute, float_tolerance)
        return result

    def _assert_chart_matches_reference(
        self,
        test_name: str,
        chart: alt.TopLevelMixin | list[list[AltairChart]],
        matching_plots_dict: dict[str, tuple[int, list[str]]],
        recompute: bool,
        float_tolerance: float = 5e-4,
    ) -> None:
        """Compare chart JSON against stored reference (updating when requested)."""

        reference_file = self.reference_dir / f"{test_name}.json"
        reference_matching_plots_file = self.reference_dir / f"{test_name}_matching_plots.json"

        # Handle matrix of plots
        if isinstance(chart, list):
            # Validate it's a list of lists of charts
            assert all(isinstance(row, list) for row in chart), f"Expected list of lists, got {type(chart)}"
            assert all(
                isinstance(item, (alt.Chart, alt.LayerChart, alt.VConcatChart, alt.HConcatChart, alt.FacetChart))
                for row in chart
                for item in row
            ), f"Expected list of lists of Altair charts, got {type(chart)}"
            # Convert matrix to list of lists of JSON specs
            result_spec = [[item.to_dict() for item in row] for row in chart]
            normalized_result = normalize_chart_json(result_spec)
        else:
            # Single chart
            assert isinstance(
                chart,
                (
                    alt.Chart,
                    alt.LayerChart,
                    alt.VConcatChart,
                    alt.HConcatChart,
                    alt.FacetChart,
                ),
            ), f"Expected Altair chart or list of lists, got {type(chart)}"
            result_spec = chart.to_dict()
            normalized_result = normalize_chart_json(result_spec)

        already_exists = reference_file.exists()
        if recompute:
            with open(reference_file, "w") as f:
                json.dump(normalized_result, f, indent=2, sort_keys=True)
            # Save matching_plots dict (only if not empty)
            if matching_plots_dict:
                with open(reference_matching_plots_file, "w") as f:
                    json.dump(matching_plots_dict, f, indent=2, sort_keys=True)
            print(f"{'Updated' if already_exists else 'Created'} reference for test {test_name}")
            return

        if not already_exists:
            # Generate and save the plot HTML for viewing
            diff_dir = self.reference_dir / "diff_html"
            diff_dir.mkdir(parents=True, exist_ok=True)
            plot_html_path = diff_dir / f"{test_name}_missing_reference.html"

            if isinstance(chart, list):
                # Matrix of plots - use plot_matrix_html
                from salk_toolkit.utils import plot_matrix_html

                actual_html = plot_matrix_html(chart, uid="actual", width=400, responsive=False)
                if actual_html:
                    # Wrap in a simple HTML page
                    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{test_name} - Missing Reference</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #fafafa; }}
    h1 {{ font-size: 1.4rem; margin-bottom: 16px; color: #d00; }}
    .plot-container {{ padding: 16px; background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  </style>
</head>
<body>
  <h1>Missing Reference - {test_name}</h1>
  <p>Reference file not found. This is the actual plot output.</p>
  <div class="plot-container">
    {actual_html}
  </div>
</body>
</html>"""
                    plot_html_path.write_text(full_html, encoding="utf-8")
            else:
                # Single chart - use save_plot_comparison_html with dummy reference
                # save_plot_comparison_html imported at module scope (avoid shadowing it here)
                # Create a dummy reference spec for display (same as actual)
                dummy_reference = result_spec.copy()
                save_plot_comparison_html(
                    dummy_reference,
                    result_spec,
                    plot_html_path,
                    page_title=f"{test_name} - Missing Reference",
                    reference_title="Missing Reference (showing actual only)",
                    actual_title="Actual",
                )

            plot_uri = plot_html_path.resolve().as_uri()
            recompute_cmd = f"pytest tests/test_plots.py::TestPlots::{test_name} --recompute"
            raise ValueError(
                f"Reference file for {test_name} not found.\n"
                f"Plot HTML saved to: {plot_uri}\n"
                f"Recompute with: {recompute_cmd}\n"
                f"Run with --recompute to create the reference file."
            )

        with open(reference_file, "r") as f:
            reference_spec = json.load(f)
        reference_json = normalize_chart_json(reference_spec)

        # Check matching_plots against reference (skip if empty, e.g., for tests that don't use e2e_plot)
        if matching_plots_dict:
            if not reference_matching_plots_file.exists():
                raise ValueError(f"Reference matching_plots file for {test_name} not found")

            with open(reference_matching_plots_file, "r") as f:
                reference_matching_plots = json.load(f)

            # Compare matching_plots (convert tuples to lists for JSON comparison)
            # JSON automatically converts tuples to lists, so normalize both to lists
            current_matching_plots = {k: [v[0], v[1]] for k, v in matching_plots_dict.items()}
            # Reference is already loaded as lists from JSON
            reference_matching_plots_normalized = reference_matching_plots

            if current_matching_plots != reference_matching_plots_normalized:
                assert False, (
                    f"Matching plots differ from reference. Test: {test_name}\n"
                    f"Reference file: {reference_matching_plots_file}\n"
                    f"Reference: {json.dumps(reference_matching_plots_normalized, indent=2, sort_keys=True)}\n"
                    f"Current: {json.dumps(current_matching_plots, indent=2, sort_keys=True)}"
                )

        if not compare_json_with_tolerance(reference_json, normalized_result, float_tolerance):
            diff_html_path = None
            try:
                diff_dir = self.reference_dir / "diff_html"
                if isinstance(chart, list):
                    # Matrix of plots - use matrix comparison
                    diff_html_path = save_plot_matrix_comparison_html(
                        reference_spec,
                        chart,
                        diff_dir / f"{test_name}.html",
                        page_title=f"{test_name} comparison",
                        width=400,
                    )
                    print(f"Matrix comparison HTML saved to {diff_html_path.resolve().as_uri()}")
                else:
                    # Single chart
                    diff_html_path = save_plot_comparison_html(
                        reference_spec,
                        result_spec,
                        diff_dir / f"{test_name}.html",
                        page_title=f"{test_name} comparison",
                    )
                    print(f"Comparison HTML saved to {diff_html_path.resolve().as_uri()}")
            except Exception as exc:  # pragma: no cover - best effort artifact
                print(f"Failed to save comparison HTML for {test_name}: {exc}")

            comparison_note = (
                f"\nComparison HTML: {diff_html_path.resolve().as_uri()}"
                if diff_html_path is not None
                else "\nComparison HTML: failed to generate (see log above)"
            )
            recompute_cmd = f"pytest tests/test_plots.py::TestPlots::{test_name} --recompute"

            diff_report = pretty_print_json_differences(reference_json, normalized_result, float_tolerance)
            assert False, (
                f"Plot output differs from reference. Test: {test_name}\n"
                f"Reference file: {reference_file}\n{comparison_note}\n"
                f"Recompute with: {recompute_cmd}\n\n"
                f"{diff_report}"
            )

    def test_boxplots_basic(self, recompute):
        """Test basic boxplots."""
        config = {
            "res_col": "party_preference",
            "factor_cols": ["age_group"],
            "filter": {},
            "plot": "boxplots",
            "internal_facet": True,
        }
        self._run_plot_test("test_boxplots_basic", config, recompute=recompute)

    def test_boxplots_raw(self, recompute):
        """Test raw boxplots."""
        config = {
            "res_col": "EKRE",
            "factor_cols": ["age_group"],
            "filter": {},
            "plot": "boxplots-raw",
            "internal_facet": True,
        }
        self._run_plot_test("test_boxplots_raw", config, recompute=recompute)

    def test_columns_basic(self, recompute):
        """Test basic column plots."""
        config = {
            "res_col": "e-valimised",
            "factor_cols": ["nationality"],
            "filter": {},
            "plot": "columns",
            "internal_facet": True,
        }
        self._run_plot_test("test_columns_basic", config, recompute=recompute)

    def test_columns_thermometer(self, recompute):
        """Test column plots with thermometer data."""
        config = {
            "res_col": "thermometer",
            "factor_cols": ["nationality"],
            "filter": {},
            "plot": "columns",
            "internal_facet": True,
        }
        self._run_plot_test("test_columns_thermometer", config, recompute=recompute)

    def test_columns_sorted_median(self, recompute):
        """Test column plot with custom aggregation and sorting."""
        config = {
            "res_col": "EKRE",
            "factor_cols": ["nationality"],
            "filter": {},
            "plot": "columns",
            "internal_facet": True,
            "agg_fn": "median",
            "sort": ["nationality"],
            "pl_filter": 'pl.col("age") > 40',
        }
        self._run_plot_test("test_columns_sorted_median", config, recompute=recompute, width=450)

    def test_stacked_columns(self, recompute):
        """Test stacked column plots."""
        config = {
            "res_col": "party_preference",
            "factor_cols": ["EKRE"],
            "internal_facet": True,
            "plot": "stacked_columns",
            "plot_args": {"normalized": False},
            "filter": {},
        }
        self._run_plot_test("test_stacked_columns", config, recompute=recompute)

    def test_diff_columns(self, recompute):
        """Test difference column plots."""
        config = {
            "res_col": "thermometer",
            "factor_cols": ["age_group"],
            "filter": {"age_group": [None, "25-34", "35-44"]},
            "plot": "diff_columns",
            "internal_facet": True,
            "plot_args": {"sort_descending": True},
        }
        self._run_plot_test("test_diff_columns", config, recompute=recompute)

    def test_massplot(self, recompute):
        """Test mass plot."""
        config = {
            "res_col": "trust",
            "factor_cols": ["party_preference"],
            "filter": {},
            "plot": "massplot",
            "internal_facet": True,
            "convert_res": "continuous",
        }
        self._run_plot_test("test_massplot", config, recompute=recompute)

    def test_likert_bars_basic(self, recompute):
        """Test basic Likert bars."""
        config = {
            "res_col": "trust",
            "factor_cols": ["party_preference"],
            "filter": {},
            "plot": "likert_bars",
            "internal_facet": True,
        }
        self._run_plot_test("test_likert_bars_basic", config, recompute=recompute)

    def test_likert_bars_no_factors(self, recompute):
        """Test Likert bars with no factor columns."""
        config = {
            "res_col": "valitsus",
            "factor_cols": [],
            "filter": {},
            "plot": "likert_bars",
            "internal_facet": True,
        }
        self._run_plot_test("test_likert_bars_no_factors", config, recompute=recompute)

    def test_density_raw_stacked(self, recompute):
        """Test raw density plot with stacking."""
        config = {
            "res_col": "thermometer",
            "factor_cols": ["party_preference"],
            "filter": {},
            "plot": "density-raw",
            "plot_args": {"stacked": True},
            "internal_facet": True,
        }
        self._run_plot_test("test_density_raw_stacked", config, recompute=recompute, width=500)

    def test_violin_raw(self, recompute):
        """Test raw violin plot."""
        config = {
            "res_col": "thermometer",
            "factor_cols": ["party_preference"],
            "filter": {},
            "plot": "violin-raw",
            "internal_facet": True,
        }
        self._run_plot_test("test_violin_raw", config, recompute=recompute, width=650)

    def test_matrix_basic(self, recompute):
        """Test basic matrix plot."""
        config = {
            "res_col": "party_preference",
            "factor_cols": ["age_group"],
            "filter": {},
            "plot": "matrix",
            "internal_facet": True,
        }
        self._run_plot_test("test_matrix_basic", config, recompute=recompute)

    def test_matrix_thermometer(self, recompute):
        """Test matrix plot with thermometer data."""
        config = {
            "res_col": "thermometer",
            "factor_cols": ["party_preference"],
            "filter": {},
            "plot": "matrix",
            "internal_facet": True,
        }
        self._run_plot_test("test_matrix_thermometer", config, recompute=recompute)

    def test_matrix_with_reorder(self, recompute):
        """Test matrix plot with reordering."""
        config = {
            "res_col": "issues",
            "factor_cols": ["party_preference"],
            "filter": {},
            "plot": "matrix",
            "plot_args": {"reorder": True},
            "internal_facet": True,
            "convert_res": "continuous",
            "cont_transform": "center",
        }
        self._run_plot_test("test_matrix_with_reorder", config, recompute=recompute)

    def test_corr_matrix(self, recompute):
        """Test correlation matrix plot."""
        config = {
            "res_col": "thermometer",
            "factor_cols": [],
            "filter": {},
            "plot": "corr_matrix",
            "internal_facet": True,
        }
        self._run_plot_test("test_corr_matrix", config, recompute=recompute)

    def test_lines_smooth(self, recompute):
        """Test line plot with smoothing."""
        config = {
            "res_col": "thermometer",
            "factor_cols": ["education"],
            "filter": {},
            "plot": "lines",
            "internal_facet": True,
            "plot_args": {"smooth": True},
            "n_facet_cols": 2,
        }
        self._run_plot_test("test_lines_smooth", config, recompute=recompute)

    def test_lines_date(self, recompute):
        """Test line plot with smoothing."""
        config = {
            "res_col": "party_preference",
            "factor_cols": ["td"],
            "filter": {},
            "plot": "lines",
            "internal_facet": True,
        }
        self._run_plot_test("test_lines_date", config, recompute=recompute)

    def test_lines_hdi_basic(self, recompute):
        """Test HDI line plot."""
        config = {
            "res_col": "party_preference",
            "factor_cols": ["age_group", "nationality"],
            "filter": {},
            "plot": "lines_hdi",
            "internal_facet": True,
        }
        self._run_plot_test("test_lines_hdi_basic", config, recompute=recompute)

    def test_area_smooth(self, recompute):
        """Test smooth area plot."""
        config = {
            "res_col": "party_preference",
            "factor_cols": ["education", "gender"],
            "filter": {},
            "plot": "area_smooth",
            "internal_facet": True,
        }
        self._run_plot_test("test_area_smooth", config, recompute=recompute)

    def test_likert_rad_pol(self, recompute):
        """Test Likert radicalization/polarization plot."""
        config = {
            "res_col": "referendum",
            "factor_cols": ["electoral_district"],
            "internal_facet": True,
            "plot": "likert_rad_pol",
            "filter": {},
        }
        # This test uses a different data file
        self._run_plot_test("test_likert_rad_pol", config, recompute=recompute, width=600)

    def test_barbell(self, recompute):
        """Test barbell plot."""
        config = {
            "res_col": "trust",
            "factor_cols": ["party_preference"],
            "filter": {},
            "plot": "barbell",
            "internal_facet": True,
            "convert_res": "continuous",
        }
        self._run_plot_test("test_barbell", config, recompute=recompute)

    def test_geoplot(self, recompute):
        """Test geographic plot."""
        if self.data_meta is None:
            pytest.skip("Data metadata not available for geoplot test")

        config = {
            "res_col": "EKRE",
            "factor_cols": ["electoral_district"],
            "filter": {},
            "plot": "geoplot",
            "internal_facet": True,
        }
        self._run_plot_test(
            "test_geoplot",
            config,
            recompute=recompute,
            data_meta=self.data_meta,
            width=400,
        )

    def test_geoplot_outer_colors_gradient(self, recompute):
        """Test geoplot with outer colors gradient."""
        if self.data_meta is None:
            pytest.skip("Data metadata not available for geoplot test")

        config = {
            "factor_cols": ["unit", "party_preference"],
            "internal_facet": True,
            "plot": "geoplot",
            "plot_args": {"separate_axes": True},
            "n_facet_cols": 2,
            "res_col": "party_preference",
        }
        self._run_plot_test(
            "test_geoplot_outer_colors_gradient",
            config,
            recompute=recompute,
            data_meta=self.data_meta,
            width=400,
        )

    def test_facet_dist(self, recompute):
        """Test facet distribution plot."""
        config = {
            "res_col": "thermometer",
            "factor_cols": ["party_preference"],
            "filter": {},
            "plot": "facet_dist",
            "internal_facet": True,
        }
        self._run_plot_test("test_facet_dist", config, recompute=recompute, float_tolerance=3e-2)

    def test_max_diff(self, recompute):
        """Test max_diff plot."""
        config = {
            "factor_cols": ["question"],
            "filter": {},
            "internal_facet": True,
            "plot": "maxdiff",
            "plot_args": {},
            "res_col": "thermometer",
        }
        self._run_plot_test("test_maxdiff", config, recompute=recompute, float_tolerance=3e-2)

    @pytest.mark.skip(reason="Very hard to make deterministic (tried but failed)")
    def test_ordered_population(self, recompute):
        """Test ordered population plot."""
        config = {
            "res_col": "thermometer",
            "factor_cols": ["party_preference"],
            "filter": {},
            "plot": "ordered_population",
            "internal_facet": True,
            "plot_args": {"full_data": True},  # required for consistency
        }
        self._run_plot_test("test_ordered_population", config, recompute=recompute, width=800)

    def test_marimekko(self, recompute):
        """Test Marimekko plot."""
        config = {
            "res_col": "trust",
            "factor_cols": ["question", "party_preference"],
            "filter": {},
            "plot": "marimekko",
            "internal_facet": True,
        }
        self._run_plot_test("test_marimekko", config, recompute=recompute)

    def test_mandate_plot_sim_done(self, recompute):
        """Test mandate plot with precomputed mandate data."""
        data = pd.DataFrame(
            {
                "draw": [0, 0, 0, 0],
                "party": ["Alpha", "Beta", "Alpha", "Beta"],
                "district": ["North", "North", "South", "South"],
                "mandates": [1, 0, 0, 1],
            }
        )
        facets = [
            FacetMeta(
                col="party",
                ocol="party",
                order=["Alpha", "Beta"],
                colors=alt.Scale(domain=["Alpha", "Beta"], range=["#c00", "#00c"]),
                neutrals=[],
                meta=ColumnMeta(categories=["Alpha", "Beta"]),
            ),
            FacetMeta(
                col="district",
                ocol="district",
                order=["North", "South"],
                colors=alt.Scale(domain=["North", "South"], range=["#090", "#990"]),
                neutrals=[],
                meta=ColumnMeta(categories=["North", "South"], mandates={"North": 1, "South": 1}),
            ),
        ]
        params = PlotInput(
            data=data,
            col_meta={},
            value_col="value",
            facets=facets,
            translate=lambda s: s,
            tooltip=[],
            alt_properties={},
            outer_factors=[],
            plot_args={},
        )
        chart = mandate_plot(
            p=params,
            mandates={"North": 1, "South": 1},
            electoral_system=ElectoralSystem(),
            sim_done=True,
        )
        # This test doesn't use e2e_plot, so matching_plots is not applicable
        self._assert_chart_matches_reference("test_mandate_plot_sim_done", chart, {}, recompute)

    def test_boxplots_convert_res(self, recompute):
        """Test boxplots with result conversion."""
        config = {
            "res_col": "age_group",
            "factor_cols": ["gender"],
            "filter": {},
            "plot": "boxplots",
            "internal_facet": True,
            "convert_res": "continuous",
            "val_range": (0.0, 3.0),
        }
        self._run_plot_test("test_boxplots_convert_res", config, recompute=recompute)

    def test_lines_hdi_with_num_values(self, recompute):
        """Test HDI lines with numerical values and custom formatting."""
        config = {
            "res_col": "valitsus",
            "factor_cols": ["party_preference", "age_group"],
            # Isamaa HDI has multiple close options so excluding it
            "filter": {"party_preference": ["Keskerakond", "Reformierakond", "SDE", "EKRE"]},
            "convert_res": "continuous",
            "num_values": [0, 0, 0, 1, 1],
            "val_name": "Pr[valitsus==Agree]",
            "val_format": ".0%",
            "plot": "lines_hdi",
            "internal_facet": True,
        }
        self._run_plot_test("test_lines_hdi_with_num_values", config, recompute=recompute, width=800)

    def test_likert_bars_with_metadata(self, recompute):
        """Test Likert bars with metadata."""
        config = {
            "res_col": "trust",
            "factor_cols": ["gender", "age_group"],
            "filter": {},
            "plot": "likert_bars",
            "internal_facet": True,
        }
        kwargs = {"width": 800}
        if self.data_meta is not None:
            kwargs["data_meta"] = self.data_meta

        self._run_plot_test("test_likert_bars_with_metadata", config, recompute=recompute, **kwargs)

    def test_plot_with_missing_data_file(self, recompute):
        """Test that appropriate error is raised for missing data file."""
        config = {
            "res_col": "party_preference",
            "factor_cols": ["age_group"],
            "filter": {},
            "plot": "boxplots",
            "internal_facet": True,
        }

        with pytest.raises((FileNotFoundError, OSError)):
            self._run_plot_test(
                "test_missing_data_file",
                config,
                data_file="nonexistent_file.parquet",
                recompute=recompute,
            )

    def test_plot_with_invalid_column(self, recompute):
        """Test behavior with invalid column names."""
        config = {
            "res_col": "nonexistent_column",
            "factor_cols": ["age_group"],
            "filter": {},
            "plot": "boxplots",
            "internal_facet": True,
        }

        # This should either raise an error or handle gracefully
        # The exact behavior depends on the implementation
        try:
            self._run_plot_test("test_plot_with_invalid_column", config, recompute=recompute)
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

        cats = [
            "Keskerakond",
            "EKRE",
            "Reformierakond",
            "Isamaa",
            "SDE",
            "Rohelised",
            "Eesti 200",
            "Parempoolsed",
            "Other",
            "None of the parties",
            "No opinion",
        ]
        result = estimate_legend_columns_horiz(cats, 800)
        assert isinstance(result, int)
        assert result > 0
        assert result <= len(cats)
        # Test the specific assertion from the notebook
        assert result == 6

    def test_cat_to_cont_axis_numeric_and_datetime(self):
        """Test _cat_to_cont_axis uses numeric/date string detection."""
        from salk_toolkit.plots import _cat_to_cont_axis

        # numeric-like categorical strings -> quantitative axis + float column
        df_num = pd.DataFrame({"x": ["1", "2.5", "3"]})
        x_axis, out = _cat_to_cont_axis(df_num.copy(), {"col": "x", "order": ["1", "2.5", "3"]})
        x_dict = x_axis.to_dict()
        assert x_dict["type"] == "quantitative"
        assert x_dict["field"] == "x_cont"
        assert out["x"].tolist() == ["1", "2.5", "3"]
        assert pd.api.types.is_float_dtype(out["x_cont"])

        # date-like categorical strings -> temporal axis + datetime column
        df_dt = pd.DataFrame({"x": ["2020-01-01", "2020-01-02"]})
        x_axis, out = _cat_to_cont_axis(df_dt.copy(), {"col": "x", "order": ["2020-01-01", "2020-01-02"]})
        x_dict = x_axis.to_dict()
        assert x_dict["type"] == "temporal"
        assert x_dict["field"] == "x_cont"
        assert len(x_dict["axis"]["values"]) == 1  # once per month
        # Ensure tick values are JSON-friendly (no pandas.Timestamp).
        assert not isinstance(x_dict["axis"]["values"][0], pd.Timestamp)
        assert out["x"].tolist() == ["2020-01-01", "2020-01-02"]
        assert out["x_cont"].tolist() == ["2020-01-01T00:00:00", "2020-01-02T00:00:00"]

        # neither -> nominal axis (keeps original values)
        df_nom = pd.DataFrame({"x": ["b", "a"]})
        x_axis, out = _cat_to_cont_axis(df_nom.copy(), {"col": "x", "order": ["b", "a"]})
        x_dict = x_axis.to_dict()
        assert x_dict["type"] == "nominal"
        assert out["x"].tolist() == ["b", "a"]

    def test_save_plot_comparison_html(self, tmp_path: Path) -> None:
        """Ensure helper writes an HTML diff to disk."""
        BASIC_SPEC = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": [{"category": "A", "value": 1}, {"category": "B", "value": 2}]},
            "mark": "bar",
            "encoding": {
                "x": {"field": "category", "type": "nominal"},
                "y": {"field": "value", "type": "quantitative"},
            },
        }

        ACTUAL_SPEC = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": [{"category": "A", "value": 2}, {"category": "B", "value": 3}]},
            "mark": "bar",
            "encoding": {
                "x": {"field": "category", "type": "nominal"},
                "y": {"field": "value", "type": "quantitative"},
            },
        }

        output_path = tmp_path / "comparison.html"

        result_path = save_plot_comparison_html(BASIC_SPEC, ACTUAL_SPEC, output_path)

        assert result_path.exists()
        content = result_path.read_text(encoding="utf-8")
        assert "vega-lite" in content


if __name__ == "__main__":
    pytest.main([__file__])
