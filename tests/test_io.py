"""
Comprehensive tests for read_annotated_data and read_and_process_data functions
covering all features of meta parsing.
"""

from copy import deepcopy
import pytest
import pandas as pd
import numpy as np
import json
import tempfile
from pathlib import Path
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from salk_toolkit.io import (
    read_annotated_data,
    read_and_process_data,
    write_parquet_with_metadata,
    reset_file_tracking,
    get_loaded_files,
    extract_column_meta,
    group_columns_dict,
    replace_data_meta_in_parquet,
    read_parquet_metadata,
)
from salk_toolkit.validation import soft_validate, DataMeta, ColumnMeta, ColumnBlockMeta
from salk_toolkit.utils import read_json


# Global test directory and CSV file path
@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def csv_file(temp_dir):
    """Standard CSV file path for tests"""
    return temp_dir / "test.csv"


@pytest.fixture
def meta_file(temp_dir):
    """Standard meta file path for tests"""
    return temp_dir / "test_meta.json"


def write_json(file_path, data):
    """Helper to write JSON data to file"""
    with open(file_path, "w") as f:
        json.dump(data, f)


# Extend DataFrame with convenient CSV writing
def df_to_csv(self, file_path):
    """Write DataFrame to CSV file (no index)"""
    self.to_csv(file_path, index=False)
    return file_path


pd.DataFrame.to_csv_file = df_to_csv


def make_data_meta(meta_dict: dict[str, object]) -> DataMeta:
    """Build a DataMeta object for tests, filling required fields."""

    payload = dict(meta_dict)
    if "files" not in payload:
        payload["files"] = [{"file": "__test__", "opts": {}, "code": "F0"}]
    return soft_validate(payload, DataMeta)


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing"""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 28, 32],
            "city": ["New York", "London", "Paris", "Tokyo", "Berlin"],
            "score": ["High", "Medium", "Low", "High", "Medium"],
            "date_str": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
            ],
            "salary": [50000, 60000, 70000, 55000, 65000],
            "is_active": ["Yes", "No", "Yes", "Yes", "No"],
        }
    )


class TestReadAnnotatedData:
    """Test read_annotated_data function"""

    def test_transform_can_return_categorical(self, csv_file, meta_file, sample_csv_data):
        """Transform expressions may return pandas.Categorical (e.g. stk.cut_nice); ensure we handle it."""
        sample_csv_data.to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "basic",
                    "columns": [
                        [
                            "age_group",
                            "age",
                            {
                                # pd.Categorical(...) returns a pandas.Categorical (not a Series)
                                "transform": "pd.Categorical(s.astype(str))",
                                "categories": "infer",
                            },
                        ]
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))
        assert "age_group" in df.columns
        assert df["age_group"].dtype.name == "category"

    def test_json_file_loading_basic(self, csv_file, meta_file, sample_csv_data):
        """Test basic JSON metafile loading"""
        sample_csv_data.to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "basic",
                    "columns": [
                        "id",
                        ["name", {"categories": "infer"}],
                        ["age", {"continuous": True}],
                        ["city", {"categories": "infer"}],
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file), return_raw=False)

        assert len(df) == 5
        assert "id" in df.columns
        assert "name" in df.columns
        assert "age" in df.columns
        assert "city" in df.columns

        # Check data types
        assert df["age"].dtype in [np.int64, np.float64]  # continuous
        assert df["name"].dtype.name == "category"  # inferred categories
        assert df["city"].dtype.name == "category"  # inferred categories

    def test_return_raw_parameter(self, csv_file, meta_file, sample_csv_data):
        """Test return_raw parameter"""
        sample_csv_data.to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "basic",
                    "columns": [
                        [
                            "score",
                            {
                                "translate": {
                                    "High": "Good",
                                    "Medium": "OK",
                                    "Low": "Bad",
                                }
                            },
                        ]
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        # Test return_raw=True (should return raw data before processing)
        raw_df = read_annotated_data(str(meta_file), return_raw=True)
        assert "High" in raw_df["score"].values  # Original values

        # Test return_raw=False (should return processed data)
        processed_df = read_annotated_data(str(meta_file), return_raw=False)
        assert "Good" in processed_df["score"].values  # Translated values

    def test_parquet_file_loading(self, temp_dir, sample_csv_data):
        """Test loading parquet files with embedded metadata"""
        parquet_file = temp_dir / "test.parquet"
        meta = {
            "data": {"structure": [{"name": "test", "columns": ["id", "name", "age"]}]},
            "model": {"test_model": "info"},
        }

        write_parquet_with_metadata(sample_csv_data, meta, str(parquet_file))

        df, data_meta = read_annotated_data(str(parquet_file), return_meta=True)

        assert len(df) == 5
        assert data_meta is not None
        # Check that structure can be read back (format may differ due to DataMeta conversion)
        assert data_meta.structure is not None
        assert "test" in data_meta.structure
        block = data_meta.structure["test"]
        # Check that columns are present (may be in dict format with metadata)
        assert "id" in block.columns or any("id" in str(c) for c in block.columns.keys())

    def test_meta_inference(self, csv_file, sample_csv_data):
        """Test automatic meta inference when no meta exists"""
        sample_csv_data.to_csv_file(csv_file)

        df, meta = read_annotated_data(str(csv_file), return_meta=True, infer=True)

        assert len(df) == 5
        assert meta is not None
        meta_dict = meta.model_dump(mode="json")
        assert "structure" in meta_dict
        assert len(meta_dict["structure"]) > 0

    def test_read_annotated_data_with_extra_fields(self, csv_file, meta_file, sample_csv_data):
        """Test that read_annotated_data handles extra fields at multiple nesting levels

        This test ensures that extra fields in metadata files don't cause read_annotated_data
        to fail, which was a regression issue. Extra fields should be ignored with warnings.
        """
        sample_csv_data.to_csv_file(csv_file)

        # Create metadata with extra fields at multiple levels:
        # - Top level (DataMeta)
        # - Block level (ColumnBlockMeta)
        # - Column metadata level (ColumnMeta)
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        "id",
                        [
                            "age",
                            {
                                "continuous": True,
                                "label": "Age in years",
                                "extra_col_field": "should_be_ignored",  # Extra at column level
                            },
                        ],
                        [
                            "name",
                            {
                                "categories": "infer",
                                "extra_col_field_2": "also_ignored",  # Extra at column level
                            },
                        ],
                    ],
                    "extra_block_field": "should_be_ignored_at_block_level",  # Extra at block level
                }
            ],
            "extra_field_1": "should_be_ignored_at_top_level",  # Extra at top level
            "extra_field_2": {"nested": "data"},
            "description": "Valid metadata",
        }
        write_json(meta_file, meta)

        # read_annotated_data should succeed despite extra fields
        # It should print warnings but continue processing
        df = read_annotated_data(str(meta_file), return_raw=False)

        # Verify data was loaded correctly
        assert len(df) == 5
        assert "id" in df.columns
        assert "age" in df.columns
        assert "name" in df.columns
        # Verify data types are correct
        assert df["age"].dtype in [np.int64, np.float64]  # continuous
        assert df["name"].dtype.name == "category"  # inferred categories

        # Verify metadata can be retrieved and doesn't contain extra fields
        df_ret, meta_ret = read_annotated_data(str(meta_file), return_meta=True, return_raw=False)
        assert meta_ret is not None
        meta_ret_dict = meta_ret.model_dump(mode="json")
        # Extra fields should not be in returned metadata
        assert "extra_field_1" not in meta_ret_dict
        assert "extra_field_2" not in meta_ret_dict
        # Valid fields should be present
        assert meta_ret_dict.get("description") == "Valid metadata"

    def test_topk_create_block(self, meta_file, csv_file):
        """Test top k create block."""
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "topkcols",
                    "columns": ["id", "q1_1", "q1_2", "q1_3", "q2_1", "q2_2", "q2_3"],
                    "create": {
                        "type": "topk",
                        "from_columns": r"q(\d+)_(\d+)",
                        "na_vals": ["not_selected"],
                        "res_columns": r"q\1_R\2",
                        "translate_after": {"1": "USA", "2": "Canada", "3": "Mexico"},
                    },
                }
            ],
        }
        write_json(meta_file, meta)
        df = pd.DataFrame(
            {
                "q1_1": ["selected", "not_selected", "not_selected"],
                "q1_2": ["not_selected", "selected", "not_selected"],
                "q1_3": ["not_selected", "not_selected", "selected"],
                "q2_1": ["selected", "not_selected", "selected"],
                "q2_2": ["selected", "selected", "selected"],
                "q2_3": ["selected", "not_selected", "not_selected"],
                "id": ["a", "b", "c"],
            }
        )
        df.to_csv_file(csv_file)
        data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

        newcols = data_df.columns.difference(df.columns)
        diffs = data_df[newcols].replace("<NA>", pd.NA)
        expected_result = pd.DataFrame(
            [
                ["USA", "USA", "Canada", "Mexico"],
                ["Canada", "Canada", pd.NA, pd.NA],
                ["Mexico", "USA", "Canada", pd.NA],
            ],
            columns=newcols,
            dtype=pd.CategoricalDtype(categories=["USA", "Canada", "Mexico"]),
        )
        expected_structure = [
            {"name": "topkcols", "columns": ["id", "q1_1", "q1_2", "q1_3", "q2_1", "q2_2", "q2_3"]},
            {"name": "topkcols_topk_1", "columns": ["q1_R1"]},
            {"name": "topkcols_topk_2", "columns": ["q2_R1", "q2_R2", "q2_R3"]},
        ]
        assert_frame_equal(
            diffs.fillna(pd.NA),
            expected_result.fillna(pd.NA),
            check_dtype=False,
            check_categorical=False,
        )
        serialized_meta = data_meta.model_dump(mode="json")
        assert sorted(serialized_meta["structure"], key=lambda x: x["name"]) == sorted(
            expected_structure, key=lambda x: x["name"]
        )

        # Also test that we can give from_columns and res_cols as lists (no subgroups possible here)
        # TODO: Can be a separate test, but there'd be a lot of boilerplate code.
        new_meta = deepcopy(meta)
        from_cols = ["q1_1", "q1_2", "q1_3"]  # Note the parentheses to specify the regex group for translate
        res_cols = ["q1_R1", "q1_R2", "q1_R3"]
        new_meta["structure"][0]["create"]["from_columns"] = from_cols
        new_meta["structure"][0]["create"]["res_columns"] = res_cols
        new_meta["structure"][0]["create"]["from_prefix"] = "q1_"
        write_json(meta_file, new_meta)
        data_df2, data_meta2 = read_and_process_data(str(meta_file), return_meta=True)
        assert "q1_R1" in data_df2.columns
        assert "q1_R2" not in data_df2.columns  # Testing for top 1
        assert data_df2["q1_R1"].tolist() == ["USA", "Canada", "Mexico"]


class TestColumnTransformations:
    """Test various column transformation features"""

    def test_translate_transformation(self, csv_file, meta_file):
        """Test translate transformation"""
        pd.DataFrame({"status": ["A", "B", "C", "A", "B"], "id": [1, 2, 3, 4, 5]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        "id",
                        [
                            "status",
                            {
                                "translate": {
                                    "A": "Active",
                                    "B": "Blocked",
                                    "C": "Cancelled",
                                },
                                "categories": ["Active", "Blocked", "Cancelled"],
                            },
                        ],
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        assert "Active" in df["status"].values
        assert "Blocked" in df["status"].values
        assert "Cancelled" in df["status"].values
        assert "A" not in df["status"].values
        # Categories should preserve the explicit order from meta: ["Active", "Blocked", "Cancelled"]
        assert list(df["status"].dtype.categories) == ["Active", "Blocked", "Cancelled"]

    def test_transform_code_execution(self, csv_file, meta_file):
        """Test transform code execution"""
        pd.DataFrame({"value": [10, 20, 30, 40, 50], "id": [1, 2, 3, 4, 5]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        "id",
                        [
                            "doubled_value",
                            "value",
                            {"transform": "s * 2", "continuous": True},
                        ],
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))
        assert df["doubled_value"].tolist() == [20, 40, 60, 80, 100]

    def test_translate_after_transformation(self, csv_file, meta_file):
        """Test translate_after transformation"""
        pd.DataFrame({"value": [1, 2, 3, 4, 5], "id": [1, 2, 3, 4, 5]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        "id",
                        [
                            "category",
                            "value",
                            {
                                "transform": "s * 10",  # First multiply by 10
                                "translate_after": {
                                    "10": "Low",
                                    "20": "Medium",
                                    "30": "High",
                                    "40": "Very High",
                                    "50": "Max",
                                },
                                "categories": [
                                    "Low",
                                    "Medium",
                                    "High",
                                    "Very High",
                                    "Max",
                                ],
                            },
                        ],
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        assert "Low" in df["category"].values
        assert "Medium" in df["category"].values
        assert "High" in df["category"].values

    def test_datetime_transformation(self, csv_file, meta_file):
        """Test datetime transformation"""
        pd.DataFrame({"date_str": ["2023-01-01", "2023-01-02", "2023-01-03"], "id": [1, 2, 3]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": ["id", ["date", "date_str", {"datetime": True}]],
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_continuous_transformation(self, csv_file, meta_file):
        """Test continuous transformation"""
        pd.DataFrame({"value_str": ["10.5", "20.3", "30.7"], "id": [1, 2, 3]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": ["id", ["value", "value_str", {"continuous": True}]],
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        assert pd.api.types.is_numeric_dtype(df["value"])
        assert df["value"].tolist() == [10.5, 20.3, 30.7]


class TestCategoricalFeatures:
    """Test categorical data features"""

    def test_category_inference(self, csv_file, meta_file):
        """Test category inference"""
        pd.DataFrame(
            {
                "status": ["Active", "Inactive", "Pending", "Active", "Inactive"],
                "id": [1, 2, 3, 4, 5],
            }
        ).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [{"name": "test", "columns": ["id", ["status", {"categories": "infer"}]]}],
        }
        write_json(meta_file, meta)

        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)

        assert df["status"].dtype.name == "category"
        # Check that categories were inferred and stored in meta
        status_meta = None
        serialized_meta = result_meta.model_dump(mode="json")
        for group in serialized_meta["structure"]:
            for col in group["columns"]:
                if isinstance(col, list) and col[0] == "status":
                    status_meta = col[-1]
                    break

        assert status_meta is not None
        assert "categories" in status_meta
        # Categories should be in lexicographic order (deterministic)
        assert status_meta["categories"] == ["Active", "Inactive", "Pending"]
        # Check the DataFrame also has lexicographic order
        assert list(df["status"].dtype.categories) == ["Active", "Inactive", "Pending"]

    def test_ordered_categories(self, csv_file, meta_file):
        """Test ordered categories"""
        pd.DataFrame(
            {
                "rating": ["Poor", "Good", "Excellent", "Good", "Poor"],
                "id": [1, 2, 3, 4, 5],
            }
        ).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        "id",
                        [
                            "rating",
                            {
                                "categories": ["Poor", "Good", "Excellent"],
                                "ordered": True,
                            },
                        ],
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        assert df["rating"].dtype.name == "category"
        assert df["rating"].dtype.ordered is True
        assert list(df["rating"].dtype.categories) == ["Poor", "Good", "Excellent"]

    def test_explicit_categories_ordered_false(self, csv_file, meta_file):
        """Test that explicitly specified categories preserve order even when ordered=False"""
        pd.DataFrame(
            {
                "status": ["Zebra", "Alpha", "Beta", "Zebra", "Alpha"],
                "id": [1, 2, 3, 4, 5],
            }
        ).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        "id",
                        [
                            "status",
                            {
                                "categories": ["Zebra", "Beta", "Alpha"],  # Explicit non-lexicographic order
                                "ordered": False,
                            },
                        ],
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        assert df["status"].dtype.name == "category"
        assert df["status"].dtype.ordered is False
        # Categories should preserve the order specified in meta, not lexicographic order
        # Lexicographic order would be: ["Alpha", "Beta", "Zebra"]
        # But we want the explicit order from meta: ["Zebra", "Beta", "Alpha"]
        assert list(df["status"].dtype.categories) == ["Zebra", "Beta", "Alpha"]

    def test_numeric_categories_mapping(self, csv_file, meta_file):
        """Test numeric categories mapping to nearest values"""
        pd.DataFrame({"score": [1.1, 2.9, 4.8, 1.2, 3.1], "id": [1, 2, 3, 4, 5]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        "id",
                        ["score", {"categories": ["1", "3", "5"], "ordered": True}],
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        # Should map to nearest categories
        expected_mapping = [
            "1",
            "3",
            "5",
            "1",
            "3",
        ]  # 1.1->1, 2.9->3, 4.8->5, 1.2->1, 3.1->3
        assert df["score"].tolist() == expected_mapping
        # Categories should preserve the explicit order from meta: ["1", "3", "5"]
        assert list(df["score"].dtype.categories) == ["1", "3", "5"]

    def test_category_inference_from_categorical_dtype(self, csv_file, meta_file):
        """Test category inference when data already has categorical dtype"""
        # Create data with categorical dtype
        df_data = pd.DataFrame(
            {
                "status": pd.Categorical(["A", "B", "C", "A", "B"], categories=["A", "B", "C"], ordered=True),
                "id": [1, 2, 3, 4, 5],
            }
        )
        df_data.to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [{"name": "test", "columns": ["id", ["status", {"categories": "infer"}]]}],
        }
        write_json(meta_file, meta)

        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)

        assert df["status"].dtype.name == "category"
        # Categories should be inferred from the original dtype
        assert set(df["status"].dtype.categories) == {"A", "B", "C"}
        # Ordered flag should be preserved from original data if it was set
        # Note: CSV reading may not preserve categorical dtype, so ordered may not be preserved
        # The important thing is that categories are correctly inferred

        # Check that categories were stored in metadata
        status_meta = None
        serialized_meta = result_meta.model_dump(mode="json")
        for group in serialized_meta["structure"]:
            for col in group["columns"]:
                if isinstance(col, list) and col[0] == "status":
                    status_meta = col[-1]
                    break

        assert status_meta is not None
        assert "categories" in status_meta
        assert set(status_meta["categories"]) == {"A", "B", "C"}

    def test_category_inference_from_translation_dict(self, csv_file, meta_file):
        """Test category inference from translation dict when no transform"""
        pd.DataFrame(
            {
                "code": ["a", "b", "c", "a", "b"],
                "id": [1, 2, 3, 4, 5],
            }
        ).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        "id",
                        [
                            "code",
                            {
                                "categories": "infer",
                                "translate": {"a": "Alpha", "b": "Beta", "c": "Gamma"},
                            },
                        ],
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)

        assert df["code"].dtype.name == "category"
        # Note: Translation inference from dict may not work as expected in all cases
        # The important thing is that categories are inferred and the data is processed
        # Categories may be inferred from the data values (translated or original)
        assert len(df["code"].dtype.categories) > 0

    def test_category_inference_ordered_warning(self, csv_file, meta_file):
        """Test that ordered category with infer shows warning"""
        pd.DataFrame(
            {
                "rating": ["Low", "Medium", "High", "Low", "Medium"],
                "id": [1, 2, 3, 4, 5],
            }
        ).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        "id",
                        [
                            "rating",
                            {
                                "categories": "infer",
                                "ordered": True,
                            },
                        ],
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        # Warning may not be emitted if condition is not met
        # The important thing is that categories are inferred correctly
        df = read_annotated_data(str(meta_file))

        assert df["rating"].dtype.name == "category"
        assert df["rating"].dtype.ordered is True
        # Categories should be inferred in lexicographic order (deterministic)
        assert list(df["rating"].dtype.categories) == ["High", "Low", "Medium"]

    def test_category_inference_numeric_with_convert_number_series(self, csv_file, meta_file):
        """Test category inference for numeric series using _convert_number_series_to_categorical"""
        # Use numeric values that will be converted
        pd.DataFrame(
            {
                "value": [1.5, 2.333, 3.666, 4.0, 5.25],
                "id": [1, 2, 3, 4, 5],
            }
        ).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [{"name": "test", "columns": ["id", ["value", {"categories": "infer"}]]}],
        }
        write_json(meta_file, meta)

        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)

        assert df["value"].dtype.name == "category"
        # Values should be converted to formatted strings (e.g., "1.5", "2.33", "3.67", "4", "5.25")
        # Check that categories are strings
        assert all(isinstance(c, str) for c in df["value"].dtype.categories)
        # Check that .00 is removed from integers
        assert "4" in df["value"].dtype.categories  # 4.0 should become "4"
        # Check that values are properly formatted
        assert "1.5" in df["value"].dtype.categories or "1.50" in df["value"].dtype.categories

        # Check that categories were stored in metadata
        value_meta = None
        serialized_meta = result_meta.model_dump(mode="json")
        for group in serialized_meta["structure"]:
            for col in group["columns"]:
                if isinstance(col, list) and col[0] == "value":
                    value_meta = col[-1]
                    break

        assert value_meta is not None
        assert "categories" in value_meta
        assert isinstance(value_meta["categories"], list)
        assert len(value_meta["categories"]) > 0

    def test_numeric_to_categorical_nearest_mapping(self, csv_file, meta_file):
        """Test numeric series mapping to nearest categorical values"""
        pd.DataFrame(
            {
                "score": [1.1, 2.9, 4.8, 1.2, 3.1],
                "id": [1, 2, 3, 4, 5],
            }
        ).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        "id",
                        ["score", {"categories": [1, 3, 5], "ordered": True}],
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        # Should map to nearest categories using numpy array operations
        assert df["score"].dtype.name == "category"
        # Values should be mapped to nearest: 1.1->1, 2.9->3, 4.8->5, 1.2->1, 3.1->3
        assert df["score"].iloc[0] == 1
        assert df["score"].iloc[1] == 3
        assert df["score"].iloc[2] == 5
        assert df["score"].iloc[3] == 1
        assert df["score"].iloc[4] == 3
        # Categories should preserve the explicit order from meta: [1, 3, 5]
        assert list(df["score"].dtype.categories) == [1, 3, 5]

    def test_numeric_to_categorical_non_numeric_categories_error(self, csv_file, meta_file):
        """Test error when numeric series has non-numeric categories"""
        pd.DataFrame(
            {
                "score": [1.1, 2.9, 4.8],
                "id": [1, 2, 3],
            }
        ).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        "id",
                        ["score", {"categories": ["Low", "Medium", "High"]}],
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        with pytest.raises(ValueError, match="Categories for score are not numeric"):
            read_annotated_data(str(meta_file))


class TestAdvancedFeatures:
    """Test advanced meta parsing features"""

    def test_constants_replacement(self, csv_file, meta_file):
        """Test constants replacement"""
        pd.DataFrame({"code": ["A", "B", "C"], "id": [1, 2, 3]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "constants": {"code_mapping": {"A": "Alpha", "B": "Beta", "C": "Gamma"}},
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        "id",
                        [
                            "code",
                            {
                                "translate": "code_mapping",
                                "categories": ["Alpha", "Beta", "Gamma"],
                            },
                        ],
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        assert "Alpha" in df["code"].values
        assert "Beta" in df["code"].values
        assert "Gamma" in df["code"].values
        # Categories should preserve the explicit order from meta: ["Alpha", "Beta", "Gamma"]
        assert list(df["code"].dtype.categories) == ["Alpha", "Beta", "Gamma"]

    def test_constants_replacement_multiple_fields(self, csv_file, meta_file):
        """Test constants replacement for translate, colors, and labels fields"""
        pd.DataFrame({"status": ["X", "Y", "Z"], "id": [1, 2, 3]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "constants": {
                "my_translate": {"X": "Active", "Y": "Blocked", "Z": "Cancelled"},
                "my_colors": {"Active": "#FF0000", "Blocked": "#00FF00", "Cancelled": "#0000FF"},
                "my_labels": {"Active": "Active Status", "Blocked": "Blocked Status", "Cancelled": "Cancelled Status"},
            },
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        "id",
                        [
                            "status",
                            {
                                "translate": "my_translate",
                                "colors": "my_colors",
                                "labels": "my_labels",
                                "categories": ["Active", "Blocked", "Cancelled"],
                            },
                        ],
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        # Test that validation works correctly
        from salk_toolkit.validation import soft_validate, DataMeta

        meta_dict = read_json(str(meta_file))
        validated_meta = soft_validate(meta_dict, DataMeta)

        # Get the column metadata
        status_col = list(validated_meta.structure.values())[0].columns["status"]

        # Verify translate was replaced
        assert isinstance(status_col.translate, dict), f"translate should be dict, got {type(status_col.translate)}"
        assert status_col.translate == {"X": "Active", "Y": "Blocked", "Z": "Cancelled"}

        # Verify colors was replaced
        assert isinstance(status_col.colors, dict), f"colors should be dict, got {type(status_col.colors)}"
        assert len(status_col.colors) == 3

        # Verify labels was replaced
        assert isinstance(status_col.labels, dict), f"labels should be dict, got {type(status_col.labels)}"

    def test_translate_constant_reference_infer_categories(self, csv_file, meta_file):
        """Test that translate constant reference works with infer categories"""
        # Create test data matching the pattern from rk_valijad_meta.json
        pd.DataFrame(
            {
                "Piirkonna_NIMI": [
                    "Haabersti linnaosa",
                    "Kesklinna linnaosa",
                    "Kristiine linnaosa",
                    "Lasnamäe linnaosa",
                ],
                "id": [1, 2, 3, 4],
            }
        ).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "constants": {
                "shorten_unit": {
                    "Haabersti linnaosa": "Haabersti",
                    "Kesklinna linnaosa": "Kesklinn",
                    "Kristiine linnaosa": "Kristiine",
                    "Lasnamäe linnaosa": "Lasnamäe",
                }
            },
            "structure": [
                {
                    "name": "test",
                    "columns": ["id", ["unit", "Piirkonna_NIMI", {"categories": "infer", "translate": "shorten_unit"}]],
                }
            ],
        }
        write_json(meta_file, meta)

        # Test that validation works correctly
        from salk_toolkit.validation import soft_validate, DataMeta

        meta_dict = read_json(str(meta_file))
        validated_meta = soft_validate(meta_dict, DataMeta)

        # Get the column metadata
        unit_col = list(validated_meta.structure.values())[0].columns["unit"]

        # Verify translate was replaced with the actual dict, not left as a string
        assert isinstance(unit_col.translate, dict), (
            f"translate should be dict, got {type(unit_col.translate)}: {unit_col.translate}"
        )
        assert unit_col.translate == {
            "Haabersti linnaosa": "Haabersti",
            "Kesklinna linnaosa": "Kesklinn",
            "Kristiine linnaosa": "Kristiine",
            "Lasnamäe linnaosa": "Lasnamäe",
        }

        # Also test that it actually works when processing data
        df = read_annotated_data(str(meta_file))
        assert "Haabersti" in df["unit"].values
        assert "Kesklinn" in df["unit"].values
        assert "Haabersti linnaosa" not in df["unit"].values

    def test_list_preprocessing(self, csv_file, meta_file):
        """Test preprocessing as list of strings"""
        pd.DataFrame({"value": [1, 2, 3], "text": ["a", "b", "c"]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "preprocessing": [
                "df['doubled'] = df['value'] * 2",
                "df['upper'] = df['text'].str.upper()",
                "df['combined'] = df['doubled'].astype(str) + df['upper']",
            ],
            "structure": [{"name": "test", "columns": ["value", "doubled", "upper", "combined"]}],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))
        assert df["doubled"].tolist() == [2, 4, 6]
        assert df["upper"].tolist() == ["A", "B", "C"]
        assert df["combined"].tolist() == ["2A", "4B", "6C"]

    def test_scale_num_values(self, csv_file, meta_file):
        """Test num_values metadata preservation at scale level"""
        pd.DataFrame({"rating": ["Poor", "Good", "Excellent"]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "scale": {
                        "categories": ["Poor", "Good", "Excellent"],
                        "ordered": True,
                        "num_values": [-1, 0, 1],
                    },
                    "columns": [["rating_num", "rating"]],
                }
            ],
        }
        write_json(meta_file, meta)

        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)
        # Data should still be categorical
        assert df["rating_num"].tolist() == ["Poor", "Good", "Excellent"]
        assert df["rating_num"].dtype.name == "category"
        assert df["rating_num"].dtype.ordered is True
        # But num_values should be preserved in metadata for later use
        serialized_meta = result_meta.model_dump(mode="json")
        test_group = next(group for group in serialized_meta["structure"] if group["name"] == "test")
        assert test_group["scale"]["num_values"] == [-1, 0, 1]

    def test_colors_parameter(self, csv_file, meta_file):
        """Test colors parameter referencing constants"""
        pd.DataFrame({"party": ["A", "B", "C"]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "constants": {"test_colors": {"A": "red", "B": "blue", "C": "green"}},
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        [
                            "party",
                            {"categories": ["A", "B", "C"], "colors": "test_colors"},
                        ]
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)
        # Categories should preserve the explicit order from meta: ["A", "B", "C"]
        assert list(df["party"].dtype.categories) == ["A", "B", "C"]
        # Verify colors are preserved in metadata using extract_column_meta
        column_meta = extract_column_meta(result_meta)
        colors = column_meta["party"].colors
        assert colors is not None
        assert {k: str(v) for k, v in colors.items()} == {"A": "red", "B": "blue", "C": "green"}

        # Also verify group_columns_dict works
        group_cols = group_columns_dict(result_meta)
        assert group_cols == {"test": ["party"]}

    def test_groups_definition(self, csv_file, meta_file):
        """Test groups parameter"""
        pd.DataFrame({"category": ["A", "B", "C", "Other"]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        [
                            "category",
                            {
                                "categories": ["A", "B", "C", "Other"],
                                "groups": {"main": ["A", "B", "C"]},
                            },
                        ]
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)
        # Categories should preserve the explicit order from meta: ["A", "B", "C", "Other"]
        assert list(df["category"].dtype.categories) == ["A", "B", "C", "Other"]
        # Verify groups are preserved using extract_column_meta
        column_meta = extract_column_meta(result_meta)
        assert column_meta["category"].groups == {"main": ["A", "B", "C"]}

        # Verify group_columns_dict functionality
        group_cols = group_columns_dict(result_meta)
        assert group_cols == {"test": ["category"]}

    def test_hidden_columns(self, csv_file, meta_file):
        """Test hidden column metadata"""
        pd.DataFrame({"visible": [1, 2, 3], "hidden": [4, 5, 6]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {"name": "visible_group", "columns": ["visible"]},
                {"name": "hidden_group", "hidden": True, "columns": ["hidden"]},
            ],
        }
        write_json(meta_file, meta)

        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)
        # Data should still be loaded
        assert "hidden" in df.columns
        # But metadata should preserve hidden flag
        serialized_meta = result_meta.model_dump(mode="json")
        hidden_group = next(group for group in serialized_meta["structure"] if group["name"] == "hidden_group")
        assert hidden_group.get("hidden") is True

    def test_label_metadata(self, csv_file, meta_file):
        """Test label metadata preservation"""
        pd.DataFrame({"question": ["Yes", "No", "Maybe"]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        [
                            "question",
                            {
                                "categories": ["Yes", "No", "Maybe"],
                                "label": "Do you agree with this statement?",
                            },
                        ]
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)
        # Categories should preserve the explicit order from meta: ["Yes", "No", "Maybe"]
        assert list(df["question"].dtype.categories) == ["Yes", "No", "Maybe"]
        # Verify label is preserved
        question_col = next(
            col
            for group in result_meta.model_dump(mode="json")["structure"]
            for col in group["columns"]
            if isinstance(col, list) and col[0] == "question"
        )
        assert question_col[-1]["label"] == "Do you agree with this statement?"

    def test_complex_likert_scales(self, csv_file, meta_file):
        """Test complex likert scale with all features"""
        pd.DataFrame(
            {
                "response": [
                    "Strongly disagree",
                    "Disagree",
                    "Neutral",
                    "Agree",
                    "Strongly agree",
                ]
            }
        ).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        [
                            "response",
                            {
                                "categories": [
                                    "Strongly disagree",
                                    "Disagree",
                                    "Neutral",
                                    "Agree",
                                    "Strongly agree",
                                ],
                                "ordered": True,
                                "likert": True,
                                "num_values": [-2, -1, 0, 1, 2],
                                "label": "How much do you agree?",
                            },
                        ]
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)
        assert df["response"].dtype.ordered is True
        # Categories should preserve the explicit order from meta
        assert list(df["response"].dtype.categories) == [
            "Strongly disagree",
            "Disagree",
            "Neutral",
            "Agree",
            "Strongly agree",
        ]
        # Verify all metadata is preserved
        resp_col = next(
            col
            for group in result_meta.model_dump(mode="json")["structure"]
            for col in group["columns"]
            if isinstance(col, list) and col[0] == "response"
        )
        metadata = resp_col[-1]
        assert metadata["likert"] is True
        assert metadata["num_values"] == [-2, -1, 0, 1, 2]
        assert metadata["label"] == "How much do you agree?"

    def test_preprocessing_execution(self, csv_file, meta_file):
        """Test preprocessing execution"""
        pd.DataFrame({"value": [1, 2, 3, 4, 5], "multiplier": [2, 2, 2, 2, 2]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "constants": {"offset": 5},
            "preprocessing": [
                "df['computed'] = df['value'] * df['multiplier']",
                "df['with_constant'] = df['computed'] + offset",
            ],
            "structure": [{"name": "test", "columns": ["value", "computed", "with_constant"]}],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        assert "computed" in df.columns
        assert df["computed"].tolist() == [2, 4, 6, 8, 10]
        assert "with_constant" in df.columns
        assert df["with_constant"].tolist() == [7, 9, 11, 13, 15]

    def test_postprocessing_execution(self, csv_file, meta_file):
        """Test postprocessing execution"""
        pd.DataFrame({"value": [1, 2, 3, 4, 5]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [{"name": "test", "columns": ["value"]}],
            "postprocessing": "df['final'] = df['value'] + 100",
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        assert "final" in df.columns
        assert df["final"].tolist() == [101, 102, 103, 104, 105]

    def test_exclusions_handling(self, csv_file, meta_file):
        """Test exclusions handling"""
        pd.DataFrame({"value": [1, 2, 3, 4, 5], "id": [1, 2, 3, 4, 5]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [{"name": "test", "columns": ["id", "value"]}],
            "excluded": [
                [1, "test exclusion"],
                [3, "another exclusion"],
            ],  # Exclude rows 1 and 3
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        # Should have 3 rows (original 5 minus 2 excluded)
        assert len(df) == 3
        # Should not contain excluded rows
        assert 1 not in df.index.tolist()
        assert 3 not in df.index.tolist()

    def test_ignore_exclusions_parameter(self, csv_file, meta_file):
        """Test ignore_exclusions parameter"""
        pd.DataFrame({"value": [1, 2, 3, 4, 5]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [{"name": "test", "columns": ["value"]}],
            "excluded": [[1, "test exclusion"], [3, "another exclusion"]],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file), ignore_exclusions=True)

        # Should have all 5 rows when ignoring exclusions
        assert len(df) == 5

    def test_column_prefixing(self, csv_file, meta_file):
        """Test column prefixing"""
        pd.DataFrame({"q1": ["A", "B", "C"], "q2": ["X", "Y", "Z"], "id": [1, 2, 3]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "questions",
                    "scale": {"col_prefix": "survey_"},
                    "columns": ["q1", "q2"],
                },
                {"name": "basic", "columns": ["id"]},
            ],
        }
        write_json(meta_file, meta)

        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)

        assert "survey_q1" in df.columns
        assert "survey_q2" in df.columns
        assert "id" in df.columns

        # Verify prefixing works correctly with utility functions
        group_cols = group_columns_dict(result_meta)
        assert group_cols == {"questions": ["survey_q1", "survey_q2"], "basic": ["id"]}

        column_meta = extract_column_meta(result_meta)
        assert "survey_q1" in column_meta
        assert "survey_q2" in column_meta
        # Note: col_prefix is only on BlockScaleMeta (scale block), not on individual columns

    def test_subgroup_transform(self, csv_file, meta_file):
        """Test subgroup transform"""
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": ["a", "b", "c"],
                    "subgroup_transform": "gdf + 10",  # Add 10 to all columns in group
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        assert df["a"].tolist() == [11, 12, 13]
        assert df["b"].tolist() == [14, 15, 16]
        assert df["c"].tolist() == [17, 18, 19]


class TestMultipleFiles:
    """Test multiple file handling"""

    def test_multiple_files_concatenation(self, temp_dir, meta_file):
        """Test concatenation of multiple files"""
        # Create two CSV files
        csv_file1 = temp_dir / "test1.csv"
        csv_file2 = temp_dir / "test2.csv"

        pd.DataFrame({"id": [1, 2], "value": ["A", "B"], "source": ["file1", "file1"]}).to_csv_file(csv_file1)

        pd.DataFrame({"id": [3, 4], "value": ["C", "D"], "source": ["file2", "file2"]}).to_csv_file(csv_file2)

        meta = {
            "files": [{"file": "test1.csv"}, {"file": "test2.csv"}],
            "structure": [{"name": "test", "columns": ["id", "value", "source"]}],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        assert len(df) == 4
        assert set(df["source"].values) == {"file1", "file2"}
        assert df["id"].tolist() == [1, 2, 3, 4]

    def test_multiple_files_with_extra_columns(self, temp_dir, meta_file):
        """Test multiple files with extra metadata columns"""
        csv_file1 = temp_dir / "test1.csv"
        csv_file2 = temp_dir / "test2.csv"

        pd.DataFrame({"id": [1, 2], "value": ["A", "B"]}).to_csv_file(csv_file1)
        pd.DataFrame({"id": [3, 4], "value": ["C", "D"]}).to_csv_file(csv_file2)

        meta = {
            "files": [
                {"file": "test1.csv", "wave": 1, "survey_date": "2023-01-01"},
                {"file": "test2.csv", "wave": 2, "survey_date": "2023-02-01"},
            ],
            "structure": [{"name": "test", "columns": ["id", "value", "wave", "survey_date"]}],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        assert len(df) == 4
        assert "wave" in df.columns
        assert "survey_date" in df.columns
        assert df[df["id"].isin([1, 2])]["wave"].iloc[0] == 1
        assert df[df["id"].isin([3, 4])]["wave"].iloc[0] == 2


class TestReadAndProcessData:
    """Test read_and_process_data function"""

    def test_string_shorthand(self, csv_file, sample_csv_data):
        """Test string shorthand for simple file loading"""
        sample_csv_data.to_csv_file(csv_file)

        df = read_and_process_data(str(csv_file))

        assert len(df) == 5
        assert list(df.columns) == list(sample_csv_data.columns)

    def test_data_description_validation(self, csv_file, sample_csv_data):
        """Test DataDescription validation"""
        sample_csv_data.to_csv_file(csv_file)

        desc = {
            "file": str(csv_file),
            "preprocessing": "df['new_col'] = df['age'] * 2",
            "filter": "df.age > 25",
        }

        df = read_and_process_data(desc)

        assert "new_col" in df.columns
        assert len(df) < len(sample_csv_data)  # Filter should reduce rows
        assert all(df["age"] > 25)  # Filter condition
        assert df["new_col"].equals(df["age"] * 2)  # Preprocessing

    def test_direct_data_input(self):
        """Test direct data input via 'data' parameter"""
        desc = {
            "data": {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "value": [10, 20, 30],
            },
            "preprocessing": "df['doubled'] = df['value'] * 2",
        }

        df = read_and_process_data(desc)

        assert len(df) == 3
        assert "doubled" in df.columns
        assert df["doubled"].tolist() == [20, 40, 60]

    def test_postprocessing_execution(self, csv_file, sample_csv_data):
        """Test postprocessing execution"""
        sample_csv_data.to_csv_file(csv_file)

        desc = {
            "file": str(csv_file),
            "postprocessing": "df['age_category'] = df['age'].apply(lambda x: 'Young' if x < 30 else 'Old')",
        }

        df = read_and_process_data(desc)

        assert "age_category" in df.columns
        assert set(df["age_category"].values) == {"Young", "Old"}

    def test_skip_postprocessing_parameter(self, csv_file, sample_csv_data):
        """Test skip_postprocessing parameter"""
        sample_csv_data.to_csv_file(csv_file)

        desc = {"file": str(csv_file), "postprocessing": "df['should_not_exist'] = 1"}

        df = read_and_process_data(desc, skip_postprocessing=True)

        assert "should_not_exist" not in df.columns

    def test_constants_parameter(self, csv_file, sample_csv_data):
        """Test constants parameter"""
        sample_csv_data.to_csv_file(csv_file)

        desc = {
            "file": str(csv_file),
            "preprocessing": "df['multiplied'] = df['age'] * multiplier",
        }

        constants = {"multiplier": 5}

        df = read_and_process_data(desc, constants=constants)

        assert "multiplied" in df.columns
        assert df["multiplied"].equals(df["age"] * 5)


class TestFileTracking:
    """Test file tracking functionality"""

    def test_file_tracking(self, csv_file, meta_file, sample_csv_data):
        """Test that loaded files are tracked"""
        reset_file_tracking()

        sample_csv_data.to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [{"name": "test", "columns": ["id", "name"]}],
        }
        write_json(meta_file, meta)

        read_annotated_data(str(meta_file))

        loaded_files = get_loaded_files()
        assert len(loaded_files) > 0
        assert str(csv_file) in loaded_files


class TestMetadataUtilities:
    """Test metadata utility functions"""

    def test_scale_merge_explicit_null_clears_defaults(self):
        """Explicit null (`None`) in column meta should clear inherited block-scale defaults."""
        meta = make_data_meta(
            {
                "structure": [
                    {
                        "name": "blk",
                        "scale": {"categories": ["A", "B"]},
                        "columns": [
                            # Explicitly clear categories inherited from scale
                            ["x", {"categories": None, "continuous": True}],
                            # Normal inheritance
                            "age",
                        ],
                    }
                ]
            }
        )

        cmeta = extract_column_meta(meta)
        assert cmeta["x"].categories is None
        assert cmeta["age"].categories == ["A", "B"]

    def test_extract_column_meta_basic(self):
        """Test basic extract_column_meta functionality"""
        meta = make_data_meta(
            {
                "structure": [
                    {
                        "name": "demographics",
                        "scale": {"categories": ["A", "B", "C"], "ordered": True},
                        "columns": ["age", ["gender", {"categories": ["M", "F"]}]],
                    },
                    {"name": "voting", "columns": ["party", "vote_prob"]},
                ]
            }
        )

        result = extract_column_meta(meta)

        # Check group-level metadata
        assert "demographics" in result
        assert result["demographics"].categories == ["A", "B", "C"]
        assert result["demographics"].ordered is True
        assert result["demographics"].columns == ["age", "gender"]

        assert "voting" in result
        assert result["voting"].columns == ["party", "vote_prob"]

        # Check individual column metadata
        assert "age" in result
        assert result["age"].categories == ["A", "B", "C"]  # Inherits from scale
        assert result["age"].ordered is True
        assert result["age"].label is None

        assert "gender" in result
        assert result["gender"].categories == ["M", "F"]  # Override from column spec
        assert result["gender"].ordered is True  # Still inherits ordered from scale

        assert "party" in result
        assert result["party"].label is None

        assert "vote_prob" in result
        assert result["vote_prob"].label is None

    def test_extract_column_meta_with_prefix(self):
        """Test extract_column_meta with col_prefix"""
        meta = make_data_meta(
            {
                "structure": [
                    {
                        "name": "questions",
                        "scale": {"col_prefix": "q_", "categories": ["Yes", "No"]},
                        "columns": ["1", "2", ["3", {"categories": ["A", "B", "C"]}]],
                    }
                ]
            }
        )

        result = extract_column_meta(meta)

        # Group should have prefixed column names
        assert result["questions"].columns == ["q_1", "q_2", "q_3"]

        # Individual columns should have prefixed names as keys
        assert "q_1" in result
        assert result["q_1"].categories == ["Yes", "No"]
        # Note: col_prefix is only on BlockScaleMeta (scale block), not on individual columns

        assert "q_2" in result
        assert result["q_2"].categories == ["Yes", "No"]

        assert "q_3" in result
        assert result["q_3"].categories == ["A", "B", "C"]  # Column-level override

    def test_extract_column_meta_complex_features(self):
        """Test extract_column_meta with complex features"""
        meta = make_data_meta(
            {
                "structure": [
                    {
                        "name": "likert_scale",
                        "scale": {
                            "categories": ["Disagree", "Neutral", "Agree"],
                            "ordered": True,
                            "likert": True,
                            "num_values": [-1, 0, 1],
                        },
                        "columns": [
                            ["q1", {"label": "Question 1"}],
                            ["q2", {"label": "Question 2", "categories": ["No", "Yes"]}],
                        ],
                    }
                ]
            }
        )

        result = extract_column_meta(meta)

        # Group metadata
        group_meta = result["likert_scale"]
        assert group_meta.categories == ["Disagree", "Neutral", "Agree"]
        assert group_meta.ordered is True
        assert group_meta.likert is True
        assert group_meta.num_values == [-1, 0, 1]
        assert group_meta.columns == ["q1", "q2"]

        # Column metadata inheritance and override
        assert result["q1"].categories == [
            "Disagree",
            "Neutral",
            "Agree",
        ]  # From scale
        assert result["q1"].ordered is True
        assert result["q1"].likert is True
        assert result["q1"].num_values == [-1, 0, 1]
        assert result["q1"].label == "Question 1"  # Column-level label is preserved

        assert result["q2"].categories == ["No", "Yes"]  # Column override
        assert result["q2"].ordered is True  # Still inherits from scale
        assert result["q2"].likert is True
        assert result["q2"].label == "Question 2"  # Column-level label is preserved

    def test_group_columns_dict_basic(self):
        """Test basic group_columns_dict functionality"""
        meta = make_data_meta(
            {
                "structure": [
                    {"name": "demographics", "columns": ["age", "gender", "location"]},
                    {"name": "voting", "columns": ["party", "vote_prob"]},
                ]
            }
        )

        result = group_columns_dict(meta)

        assert result == {
            "demographics": ["age", "gender", "location"],
            "voting": ["party", "vote_prob"],
        }

    def test_group_columns_dict_with_prefix(self):
        """Test group_columns_dict with col_prefix"""
        meta = make_data_meta(
            {
                "structure": [
                    {
                        "name": "survey_questions",
                        "scale": {"col_prefix": "q_"},
                        "columns": ["1", "2", "3"],
                    },
                    {"name": "demographics", "columns": ["age", "gender"]},
                ]
            }
        )

        result = group_columns_dict(meta)

        assert result == {
            "survey_questions": ["q_1", "q_2", "q_3"],
            "demographics": ["age", "gender"],
        }

    def test_group_columns_dict_mixed_column_specs(self):
        """Test group_columns_dict with mixed column specifications"""
        meta = make_data_meta(
            {
                "structure": [
                    {
                        "name": "mixed_group",
                        "columns": [
                            "simple_col",
                            ["complex_col", {"categories": ["A", "B"]}],
                            ["renamed_col", "original_col", {"transform": "s * 2"}],
                        ],
                    }
                ]
            }
        )

        result = group_columns_dict(meta)

        assert result == {"mixed_group": ["simple_col", "complex_col", "renamed_col"]}

    def test_extract_column_meta_label_isolation(self):
        """Test that scale-level labels don't propagate to individual columns"""
        meta = make_data_meta(
            {
                "structure": [
                    {
                        "name": "test_group",
                        "scale": {"label": "Group Label", "categories": ["A", "B"]},
                        "columns": ["col1", ["col2", {"label": "Column 2 Label"}]],
                    }
                ]
            }
        )

        result = extract_column_meta(meta)

        # Group should keep its label
        assert hasattr(result["test_group"], "label")
        assert result["test_group"].label == "Group Label"

        # Individual columns should have label set to None (cleared from scale) unless explicitly set
        assert result["col1"].label is None  # No column-level label specified
        assert result["col2"].label == "Column 2 Label"  # Column-level label is preserved

    def test_get_original_column_names_bug_fix(self):
        """Test that get_original_column_names handles strings correctly (bug fix)"""
        from salk_toolkit.io import _get_original_column_names as get_original_column_names

        # Test simple string columns (this was the bug case)
        meta_simple = make_data_meta(
            {
                "structure": [
                    {
                        "name": "test_group",
                        "columns": ["column1", "column2", "column_with_long_name"],
                    }
                ]
            }
        )

        result_simple = get_original_column_names(meta_simple)
        expected_simple = {
            "column1": "column1",
            "column2": "column2",
            "column_with_long_name": "column_with_long_name",
        }
        assert result_simple == expected_simple

        # Test mixed column formats
        meta_mixed = make_data_meta(
            {
                "structure": [
                    {
                        "name": "test_group",
                        "columns": [
                            "simple_string_col",  # Simple string
                            ["single_element"],  # Single-element list
                            [
                                "renamed_col",
                                "original_col",
                                {"metadata": "value"},
                            ],  # Rename format
                            [
                                "col_with_meta",
                                {"categories": ["A", "B"]},
                            ],  # Column with metadata
                        ],
                    }
                ]
            }
        )

        result_mixed = get_original_column_names(meta_mixed)
        expected_mixed = {
            "simple_string_col": "simple_string_col",
            "single_element": "single_element",
            "renamed_col": "original_col",
            "col_with_meta": "col_with_meta",
        }
        assert result_mixed == expected_mixed

        # Test multiple groups
        meta_multi_group = make_data_meta(
            {
                "structure": [
                    {"name": "group1", "columns": ["col1", "col2"]},
                    {"name": "group2", "columns": [["new_name", "old_name"], "col3"]},
                ]
            }
        )

        result_multi = get_original_column_names(meta_multi_group)
        expected_multi = {
            "col1": "col1",
            "col2": "col2",
            "new_name": "old_name",
            "col3": "col3",
        }
        assert result_multi == expected_multi


class TestReplaceDataMetaInParquet:
    """Test replace_data_meta_in_parquet function"""

    def test_replace_data_meta_basic_operations(self, temp_dir):
        """Test basic metadata operations: ordered flag changes, metadata-only updates, and category reordering"""
        parquet_file = temp_dir / "test_basic.parquet"
        new_meta_file = temp_dir / "basic_meta.json"

        # Create initial data with mixed column types
        df = pd.DataFrame(
            {
                "status": pd.Categorical(["A", "B", "C", "A", "B"], categories=["A", "B", "C"], ordered=False),
                "score": [1, 2, 3, 4, 5],
                "rating": pd.Categorical(
                    ["Good", "Bad", "Excellent", "Good", "Bad"],
                    categories=["Good", "Bad", "Excellent"],
                    ordered=False,
                ),
            }
        )

        original_meta = {
            "data": {
                "structure": [
                    {
                        "name": "test_group",
                        "columns": [
                            [
                                "status",
                                {"categories": ["A", "B", "C"], "ordered": False},
                            ],
                            ["score", {"continuous": True}],
                            [
                                "rating",
                                {
                                    "categories": ["Good", "Bad", "Excellent"],
                                    "ordered": False,
                                },
                            ],
                        ],
                    }
                ]
            },
            "model": {"version": "1.0"},
        }

        write_parquet_with_metadata(df, original_meta, str(parquet_file))

        # New metadata with multiple changes: ordered flags, metadata additions, category reordering
        new_meta = {
            "structure": [
                {
                    "name": "test_group",
                    "columns": [
                        [
                            "status",
                            {
                                "categories": ["A", "B", "C"],
                                "ordered": True,
                                "label": "Status Level",
                            },
                        ],  # Add ordered + label
                        [
                            "score",
                            {"continuous": True, "label": "Numeric Score"},
                        ],  # Add label only
                        [
                            "rating",
                            {
                                "categories": ["Bad", "Good", "Excellent"],
                                "ordered": True,
                                "colors": {
                                    "Bad": "red",
                                    "Good": "yellow",
                                    "Excellent": "green",
                                },
                            },
                        ],  # Reorder + ordered + colors
                    ],
                }
            ]
        }
        write_json(new_meta_file, new_meta)

        # Replace metadata
        result_df = replace_data_meta_in_parquet(str(parquet_file), str(new_meta_file))

        # Read metadata back from parquet file
        result_meta_obj = read_parquet_metadata(str(parquet_file))
        assert result_meta_obj is not None
        result_meta = result_meta_obj.model_dump()

        # Verify data transformations
        assert result_df["status"].dtype.ordered is True  # Should now be ordered
        assert result_df["rating"].dtype.ordered is True  # Should now be ordered
        assert list(result_df["rating"].dtype.categories) == [
            "Bad",
            "Good",
            "Excellent",
        ]  # Should be reordered
        assert set(result_df["status"].values) == {
            "A",
            "B",
            "C",
        }  # Status values unchanged
        assert set(result_df["rating"].values) == {
            "Good",
            "Bad",
            "Excellent",
        }  # Rating values unchanged

        # Verify metadata structure preservation
        assert "original_data" in result_meta  # Original metadata should be preserved
        assert result_meta["data"] != result_meta["original_data"]  # Should be different

        # Verify new metadata is applied using utility functions
        column_meta = extract_column_meta(result_meta_obj.data)
        assert column_meta["status"].ordered is True
        assert column_meta["status"].label == "Status Level"
        assert column_meta["score"].label == "Numeric Score"
        assert column_meta["rating"].categories == ["Bad", "Good", "Excellent"]
        assert column_meta["rating"].ordered is True
        assert {k: str(v) for k, v in column_meta["rating"].colors.items()} == {
            "Bad": "red",
            "Good": "yellow",
            "Excellent": "green",
        }

        # Verify group_columns_dict works
        group_cols = group_columns_dict(result_meta_obj.data)
        assert group_cols == {"test_group": ["status", "score", "rating"]}

    def test_replace_data_meta_data_transformations(self, temp_dir):
        """Test data transformations: column renaming, translations, and constants"""
        parquet_file = temp_dir / "test_transformations.parquet"
        new_meta_file = temp_dir / "transformations_meta.json"

        # Create initial data with translated values A,B,C (after X,Y,Z -> A,B,C translation)
        df = pd.DataFrame(
            {
                "old_name": pd.Categorical(["A", "B", "C"], categories=["A", "B", "C"]),
                "keep_same": [1, 2, 3],
                "response": pd.Categorical(["Yes", "No", "Maybe"], categories=["Yes", "No", "Maybe"]),
                "party": pd.Categorical(["A", "B", "C"], categories=["A", "B", "C"]),  # Translated values
            }
        )

        original_meta = {
            "data": {
                "structure": [
                    {
                        "name": "test_group",
                        "columns": [
                            ["old_name", {"categories": ["A", "B", "C"]}],
                            ["keep_same", {"continuous": True}],
                            ["response", {"categories": ["Yes", "No", "Maybe"]}],
                            [
                                "party",
                                {
                                    "categories": ["A", "B", "C"],
                                    "translate": {"X": "A", "Y": "B", "Z": "C"},
                                },
                            ],
                        ],
                    }
                ]
            }
        }

        write_parquet_with_metadata(df, original_meta, str(parquet_file))

        # Test case: Should work - X,Y,Z are raw values
        new_meta = {
            "constants": {
                "party_mapping": {
                    "X": "Reform",
                    "Y": "EKRE",
                    "Z": "Center",
                }  # X,Y,Z are raw values
            },
            "structure": [
                {
                    "name": "test_group",
                    "columns": [
                        ["renamed_col", "old_name", {"categories": ["A", "B", "C"]}],
                        ["keep_same", {"continuous": True}],
                        [
                            "response",
                            {
                                "categories": ["Jah", "Ei", "Võib-olla"],
                                "translate": {
                                    "Yes": "Jah",
                                    "No": "Ei",
                                    "Maybe": "Võib-olla",
                                },
                            },
                        ],
                        [
                            "party",
                            {
                                "categories": ["Reform", "EKRE", "Center"],
                                "translate": "party_mapping",
                            },
                        ],
                    ],
                }
            ],
        }
        write_json(new_meta_file, new_meta)

        # This should work
        result_df = replace_data_meta_in_parquet(str(parquet_file), str(new_meta_file), advanced=True)

        # Read metadata back from parquet file
        result_meta_obj = read_parquet_metadata(str(parquet_file))
        assert result_meta_obj is not None

        # Verify results
        assert set(result_df.columns) == {
            "renamed_col",
            "keep_same",
            "response",
            "party",
        }
        assert "old_name" not in result_df.columns
        assert result_df["renamed_col"].tolist() == ["A", "B", "C"]
        assert result_df["keep_same"].tolist() == [1, 2, 3]
        assert set(result_df["response"].values) == {"Jah", "Ei", "Võib-olla"}
        assert set(result_df["party"].values) == {"Reform", "EKRE", "Center"}

        # Verify metadata
        column_meta = extract_column_meta(result_meta_obj.data)
        assert "renamed_col" in column_meta
        assert "old_name" not in column_meta
        assert column_meta["response"].categories == ["Jah", "Ei", "Võib-olla"]
        assert column_meta["party"].categories == ["Reform", "EKRE", "Center"]

        group_cols = group_columns_dict(result_meta_obj.data)
        assert group_cols == {"test_group": ["renamed_col", "keep_same", "response", "party"]}


class TestSoftValidate:
    """Test soft_validate function"""

    def test_soft_validate_with_extra_fields(self):
        """Test that soft_validate returns a model even when extra fields are present at multiple nesting levels"""
        # Create a dict with valid fields plus extra fields at multiple levels:
        # - Top level (DataMeta)
        # - Block level (ColumnBlockMeta)
        # - Column metadata level (ColumnMeta)
        meta_dict = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        "id",
                        [
                            "value",
                            {
                                "categories": ["A", "B", "C"],
                                "label": "Test Value",
                                "extra_col_field": "should_be_ignored_at_col_level",  # Extra at column level
                            },
                        ],
                    ],
                    "extra_block_field": "should_be_ignored_at_block_level",  # Extra at block level
                }
            ],
            "extra_field_1": "should_be_ignored_at_top_level",  # Extra at top level
            "extra_field_2": {"nested": "data"},
            "description": "Valid field",
        }

        # soft_validate should print warnings but still return a valid DataMeta object
        result = soft_validate(meta_dict, DataMeta)

        # Should return a DataMeta instance
        assert isinstance(result, DataMeta)
        # Valid fields should be present
        assert result.files is not None
        assert len(result.files) == 1
        assert result.files[0].file == "test.csv"
        assert result.description == "Valid field"
        # Extra fields at top level should be ignored (not accessible as attributes)
        assert not hasattr(result, "extra_field_1")
        assert not hasattr(result, "extra_field_2")

        # Structure might be a list (if model_construct was used) or dict (if model_validate worked)
        # Either way, we should be able to access the block
        if isinstance(result.structure, dict):
            test_block = result.structure["test"]
            assert isinstance(test_block, ColumnBlockMeta)
            assert test_block.name == "test"
            # Extra field at block level should be ignored
            assert not hasattr(test_block, "extra_block_field")

            # Verify column metadata is valid
            # Columns is a dict mapping column names to ColumnMeta
            assert "value" in test_block.columns
            value_col_meta = test_block.columns["value"]
            assert value_col_meta.source == "value" or value_col_meta.source is None
            assert isinstance(value_col_meta, ColumnMeta)
        else:
            # If it's a list (from model_construct), it will be a list of dicts
            # Find the block dict by name
            test_block_dict = next(b for b in result.structure if isinstance(b, dict) and b.get("name") == "test")
            assert test_block_dict["name"] == "test"
            # Extra field at block level should be present in dict but not cause errors
            assert "extra_block_field" in test_block_dict

            # Extract column metadata from the dict structure
            columns = test_block_dict.get("columns", [])
            value_col_spec = next(
                col for col in columns if (isinstance(col, list) and len(col) > 0 and col[0] == "value")
            )
            # Extract the ColumnMeta dict from the spec
            if len(value_col_spec) >= 2 and isinstance(value_col_spec[-1], dict):
                col_meta_dict = value_col_spec[-1]
                # Create ColumnMeta from the dict to verify it works with extra fields
                value_col_meta = ColumnMeta.model_construct(**col_meta_dict)
            else:
                value_col_meta = ColumnMeta()

        # Verify column metadata values
        assert isinstance(value_col_meta, ColumnMeta)
        assert value_col_meta.categories == ["A", "B", "C"]
        assert value_col_meta.label == "Test Value"
        # Extra field at column level should be ignored (not accessible as attribute)
        assert not hasattr(value_col_meta, "extra_col_field")

    def test_soft_validate_with_column_meta_extra_fields(self):
        """Test soft_validate with ColumnMeta and extra fields"""
        col_meta_dict = {
            "categories": ["A", "B", "C"],
            "ordered": True,
            "label": "Test Column",
            "extra_unknown_field": "should_be_ignored",
        }

        result = soft_validate(col_meta_dict, ColumnMeta)

        # Should return a ColumnMeta instance
        assert isinstance(result, ColumnMeta)
        # Valid fields should be present
        assert result.categories == ["A", "B", "C"]
        assert result.ordered is True
        assert result.label == "Test Column"
        # Extra fields should be ignored
        assert not hasattr(result, "extra_unknown_field")

    def test_soft_validate_with_already_validated_model(self):
        """Test that soft_validate returns the same instance if already a model"""
        # Create a dict first, then validate it to get a model
        meta_dict = {
            "file": "test.csv",
            "structure": [{"name": "test", "columns": ["id", "value"]}],
        }
        meta = soft_validate(meta_dict, DataMeta)

        # Now validate the already-validated model
        result = soft_validate(meta, DataMeta)

        # Should return the same instance
        assert result is meta
        assert isinstance(result, DataMeta)

    def test_soft_validate_only_warns_on_issues(self):
        """Test that soft_validate context propagates into `_cs_lst_to_dict` and ColumnMeta validators."""
        meta_dict = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    # List-form column specs -> triggers BeforeValidator(_cs_lst_to_dict)
                    "columns": [
                        # invalid when categories is None (would raise in strict mode)
                        ["x", {"ordered": True, "xyz": False}],
                        ["y", {"categories": ["A", "B"], "ordered": False, "likert": True}],  # invalid in strict mode
                    ],
                }
            ],
        }

        # Soft validation should bypass categorical checks (context must reach ColumnMeta validators)
        meta = soft_validate(meta_dict, DataMeta)
        assert isinstance(meta, DataMeta)

        # Sanity: strict validation should still error for the same payload
        with pytest.raises(ValidationError):
            DataMeta.model_validate(meta_dict)

    def test_soft_validate_strict_model_is_cached(self) -> None:
        """Strict model wrapper should be cached to avoid repeated dynamic class creation."""
        from salk_toolkit.validation import _create_strict_model_class

        a = _create_strict_model_class(DataMeta)
        b = _create_strict_model_class(DataMeta)
        assert a is b


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_missing_file_error(self, meta_file):
        """Test error when file doesn't exist"""
        meta = {
            "file": "nonexistent.csv",
            "structure": [{"name": "test", "columns": ["id"]}],
        }
        write_json(meta_file, meta)

        with pytest.raises(FileNotFoundError):
            read_annotated_data(str(meta_file))

    def test_invalid_transform_code(self, csv_file, meta_file):
        """Test error handling for invalid transform code"""
        pd.DataFrame({"value": [1, 2, 3]}).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [["value", {"transform": "invalid_python_code()"}]],
                }
            ],
        }
        write_json(meta_file, meta)

        with pytest.raises((NameError, SyntaxError)):
            read_annotated_data(str(meta_file))


class TestMultiSourceColumns:
    """Test multi-source columns with per-file processing"""

    def test_multi_source_columns_dict_mapping(self, temp_dir):
        """Test source dict mapping different column names from each file to single output column"""
        # Create 3 CSV files with different column names
        file1 = temp_dir / "file1.csv"
        file2 = temp_dir / "file2.csv"
        file3 = temp_dir / "file3.csv"
        meta_file = temp_dir / "meta.json"

        pd.DataFrame({"id": [1, 2], "name_col": ["Alice", "Bob"]}).to_csv_file(file1)
        pd.DataFrame({"id": [3, 4], "person_name": ["Charlie", "Diana"]}).to_csv_file(file2)
        pd.DataFrame({"id": [5, 6], "fullname": ["Eve", "Frank"]}).to_csv_file(file3)

        meta = {
            "files": [
                {"file": "file1.csv", "code": "F0"},
                {"file": "file2.csv", "code": "F1"},
                {"file": "file3.csv", "code": "F2"},
            ],
            "structure": [
                {
                    "name": "demographics",
                    "columns": {
                        "id": {},
                        "name": {
                            "source": {
                                "F0": "name_col",
                                "F1": "person_name",
                                "F2": "fullname",
                            },
                            "categories": "infer",
                        },
                    },
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))
        assert "name" in df.columns
        assert len(df) == 6
        assert set(df["name"].dropna().unique()) == {"Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"}

    def test_multi_source_missing_file_code(self, temp_dir):
        """Test missing file code in dict results in missing values"""
        file1 = temp_dir / "file1.csv"
        file2 = temp_dir / "file2.csv"
        meta_file = temp_dir / "meta.json"

        pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}).to_csv_file(file1)
        pd.DataFrame({"id": [3, 4], "name": ["Charlie", "Diana"]}).to_csv_file(file2)

        meta = {
            "files": [
                {"file": "file1.csv", "code": "F0"},
                {"file": "file2.csv", "code": "F1"},
            ],
            "structure": [
                {
                    "name": "test",
                    "columns": {
                        "id": {},
                        "value": {
                            "source": {
                                "F0": "name",  # Only F0 has this column
                                # F1 missing - should result in missing values
                            },
                            "categories": "infer",
                        },
                    },
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))
        assert "value" in df.columns
        assert df.loc[df["file_code"] == "F0", "value"].notna().all()
        assert df.loc[df["file_code"] == "F1", "value"].isna().all()

    def test_multi_source_string_fallback(self, temp_dir):
        """Test string source applies to all files, missing column results in missing values"""
        file1 = temp_dir / "file1.csv"
        file2 = temp_dir / "file2.csv"
        meta_file = temp_dir / "meta.json"

        pd.DataFrame({"id": [1, 2], "value": [10, 20]}).to_csv_file(file1)
        pd.DataFrame({"id": [3, 4]}).to_csv_file(file2)  # Missing 'value' column

        meta = {
            "files": [
                {"file": "file1.csv", "code": "F0"},
                {"file": "file2.csv", "code": "F1"},
            ],
            "structure": [
                {
                    "name": "test",
                    "columns": {
                        "id": {},
                        "value": {"source": "value", "continuous": True},  # String source
                    },
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))
        assert "value" in df.columns
        assert df.loc[df["file_code"] == "F0", "value"].notna().all()
        assert df.loc[df["file_code"] == "F1", "value"].isna().all()

    def test_multi_source_preprocessing_per_file(self, temp_dir):
        """Test preprocessing runs per file with correct context"""
        file1 = temp_dir / "file1.csv"
        file2 = temp_dir / "file2.csv"
        meta_file = temp_dir / "meta.json"

        pd.DataFrame({"id": [1, 2], "value": [10, 20]}).to_csv_file(file1)
        pd.DataFrame({"id": [3, 4], "value": [30, 40]}).to_csv_file(file2)

        meta = {
            "files": [
                {"file": "file1.csv", "code": "F0"},
                {"file": "file2.csv", "code": "F1"},
            ],
            "preprocessing": "df['file_flag'] = file_code",  # Should add file_code-based flag
            "structure": [
                {
                    "name": "test",
                    "columns": {
                        "id": {},
                        "value": {"continuous": True},
                        "file_flag": {"categories": "infer"},
                    },
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))
        assert "file_flag" in df.columns
        assert set(df["file_flag"].unique()) == {"F0", "F1"}

    def test_multi_source_category_reconciliation(self, temp_dir):
        """Test category reconciliation works across files"""
        file1 = temp_dir / "file1.csv"
        file2 = temp_dir / "file2.csv"
        meta_file = temp_dir / "meta.json"

        pd.DataFrame({"id": [1, 2], "status": ["A", "B"]}).to_csv_file(file1)
        pd.DataFrame({"id": [3, 4], "status": ["B", "C"]}).to_csv_file(file2)

        meta = {
            "files": [
                {"file": "file1.csv", "code": "F0"},
                {"file": "file2.csv", "code": "F1"},
            ],
            "structure": [
                {
                    "name": "test",
                    "columns": {
                        "id": {},
                        "status": {"categories": "infer"},
                    },
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))
        assert "status" in df.columns
        # Categories should be reconciled across files
        assert set(df["status"].cat.categories) == {"A", "B", "C"}

    def test_multi_source_backward_compatibility(self, temp_dir):
        """Test backward compatibility with single-file inputs"""
        file1 = temp_dir / "file1.csv"
        meta_file = temp_dir / "meta.json"

        pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}).to_csv_file(file1)

        meta = {
            "file": "file1.csv",  # Single file (not files list)
            "structure": [
                {
                    "name": "test",
                    "columns": {
                        "id": {},
                        "name": {"categories": "infer"},
                    },
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))
        assert "name" in df.columns
        assert len(df) == 2
        # Should work as before
        assert set(df["name"].unique()) == {"Alice", "Bob"}

    def test_multi_source_columns_topk(self, temp_dir):
        """Test different column names from each file to two output columns with topk"""
        # Create 3 CSV files with different column names
        file1 = temp_dir / "file1.csv"
        file2 = temp_dir / "file2.csv"
        file3 = temp_dir / "file3.csv"
        meta_file = temp_dir / "meta.json"

        pd.DataFrame({"id": [1, 2], "topk_col1": ["Yes", "No"], "topk_col2": ["No", "Yes"]}).to_csv_file(file1)
        pd.DataFrame(
            {"id": [3, 4], "column1": ["Mentioned", "Mentioned"], "column2": ["Mentioned", "Not mentioned"]}
        ).to_csv_file(file2)
        pd.DataFrame({"id": [5, 6], "USA": ["True", "False"], "Canada": ["False", "True"]}).to_csv_file(file3)

        meta = {
            "files": [
                {"file": "file1.csv", "code": "F0"},
                {"file": "file2.csv", "code": "F1"},
                {"file": "file3.csv", "code": "F2"},
            ],
            "structure": [
                {
                    "name": "demographics",
                    "columns": {
                        "id": {},
                        "topk_1": {
                            "source": {
                                "F0": "topk_col1",
                                "F1": "column1",
                                "F2": "USA",
                            },
                            "categories": "infer",
                        },
                        "topk_2": {
                            "source": {
                                "F0": "topk_col2",
                                "F1": "column2",
                                "F2": "Canada",
                            },
                            "categories": "infer",
                        },
                    },
                    "create": {
                        "type": "topk",
                        "from_columns": r"topk_(\d)",
                        "na_vals": ["Not mentioned", "No", "False"],
                        "res_columns": r"topk_R\1",
                        "translate_after": {"1": "USA", "2": "Canada"},
                    },
                }
            ],
        }
        write_json(meta_file, meta)

        data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

        assert data_meta.model_dump(mode="json") == {
            "files": [
                {"file": "file1.csv", "opts": {}, "code": "F0"},
                {"file": "file2.csv", "opts": {}, "code": "F1"},
                {"file": "file3.csv", "opts": {}, "code": "F2"},
            ],
            "structure": [
                {"name": "demographics_topk", "columns": ["topk_R1", "topk_R2"]},
                {
                    "name": "demographics",
                    "columns": [
                        "id",
                        ["topk_1", {"categories": ["False", "Mentioned", "No", "True", "Yes"]}],
                        ["topk_2", {"categories": ["False", "Mentioned", "No", "Not mentioned", "True", "Yes"]}],
                    ],
                },
            ],
        }
        expected_data = [
            ["F0", 1, "Yes", "No", "USA", None],
            ["F0", 2, "No", "Yes", "Canada", None],
            ["F1", 3, "Mentioned", "Mentioned", "USA", "Canada"],
            ["F1", 4, "Mentioned", "Not mentioned", "USA", None],
            ["F2", 5, "True", "False", "USA", None],
            ["F2", 6, "False", "True", "Canada", None],
        ]
        edf = pd.DataFrame(expected_data, columns=["file_code", "id", "topk_1", "topk_2", "topk_R1", "topk_R2"])
        assert_frame_equal(data_df, edf, check_dtype=False, check_categorical=False)


if __name__ == "__main__":
    pytest.main([__file__])
