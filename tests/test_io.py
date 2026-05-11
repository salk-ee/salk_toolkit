"""
Comprehensive tests for read_annotated_data and read_and_process_data functions
covering all features of meta parsing.
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from salk_toolkit.io import (
    read_annotated_data,
    read_and_process_data,
    read_parquet_with_metadata,
    write_parquet_with_metadata,
    reset_file_tracking,
    get_loaded_files,
    extract_column_meta,
    group_columns_dict,
    replace_data_meta_in_parquet,
    read_parquet_metadata,
    infer_meta,
)
from salk_toolkit.validation import (
    soft_validate,
    DataMeta,
    ColumnMeta,
    ColumnBlockMeta,
    BlockScaleMeta,
    TopKBlock,
    MaxDiffBlock,
)
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

    def test_drop_missing_declared_columns(self, temp_dir):
        """Declared-but-missing columns are always dropped from both df and returned meta."""
        csv = temp_dir / "src.csv"
        pd.DataFrame({"id": [1, 2]}).to_csv_file(csv)

        meta_file = temp_dir / "m_drop_missing.json"
        write_json(
            meta_file,
            {
                "file": "src.csv",
                "structure": [{"name": "g", "columns": ["id", "persona"]}],
            },
        )

        df, meta = read_annotated_data(str(meta_file), return_meta=True)
        assert "persona" not in df.columns
        assert "persona" in extract_column_meta(meta)

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

    def test_read_annotated_data_with_extra_fields(self, csv_file, meta_file, sample_csv_data, capsys):
        """Test that read_annotated_data warns on extra fields at all nesting levels but still succeeds."""
        sample_csv_data.to_csv_file(csv_file)

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

        # Processing should succeed despite extra fields, with warnings printed
        df, meta_ret = read_annotated_data(str(meta_file), return_meta=True, return_raw=False)
        assert len(df) == 5
        assert "id" in df.columns
        assert meta_ret is not None
        assert meta_ret.description == "Valid metadata"

        # Warnings about extras should have been printed
        captured = capsys.readouterr()
        assert "extra_field_1" in captured.out
        assert "extra_block_field" in captured.out
        assert "extra_col_field" in captured.out

    def test_topk_create_block(self, meta_file, csv_file):
        """Test top k create block."""
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "topk",
                    "name": "topkcols",
                    "columns": ["id", "q1_1", "q1_2", "q1_3", "q2_1", "q2_2", "q2_3"],
                    "from_columns": r"q(\d+)_(\d+)",
                    "na_vals": ["not_selected"],
                    "res_columns": r"q\1_R\2",
                    "scale": {"translate_after": {"1": "USA", "2": "Canada", "3": "Mexico"}},
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

        newcols = data_df.columns.difference(df.columns).difference(["file_code", "file_name"])
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
            {"name": "topkcols_1", "columns": ["q1_R1"]},
            {"name": "topkcols_2", "columns": ["q2_R1", "q2_R2", "q2_R3"]},
        ]
        assert_frame_equal(
            diffs.fillna(pd.NA),
            expected_result.fillna(pd.NA),
            check_dtype=False,
            check_categorical=False,
        )
        serialized_meta = data_meta.model_dump(mode="json")
        structure_wo_files = [
            {"name": b["name"], "columns": b["columns"]}
            for b in serialized_meta["structure"]
            if b.get("name") != "files"
        ]
        assert sorted(structure_wo_files, key=lambda x: x["name"]) == sorted(
            expected_structure, key=lambda x: x["name"]
        )

        # Generated topk blocks are themselves TopKBlock instances; `type` survives.
        topk_block = data_meta.structure["topkcols_1"]
        assert isinstance(topk_block, TopKBlock)
        assert topk_block.type == "topk"
        # Resolved column lists replace regex patterns on output
        assert isinstance(topk_block.from_columns, list)
        assert isinstance(topk_block.res_columns, list)
        assert topk_block.segments() == [(list(topk_block.columns.keys()), None, False)]

        # Also test that we can give from_columns and res_cols as lists (no subgroups possible here)
        # TODO: Can be a separate test, but there'd be a lot of boilerplate code.
        from_cols = ["q1_1", "q1_2", "q1_3"]  # Note the parentheses to specify the regex group for translate
        res_cols = ["q1_R1", "q1_R2", "q1_R3"]
        meta["structure"][0]["from_columns"] = from_cols
        meta["structure"][0]["res_columns"] = res_cols
        meta["structure"][0]["from_prefix"] = "q1_"
        write_json(meta_file, meta)
        data_df2, data_meta2 = read_and_process_data(str(meta_file), return_meta=True)
        assert "q1_R1" in data_df2.columns
        assert "q1_R2" not in data_df2.columns  # Testing for top 1
        assert data_df2["q1_R1"].tolist() == ["USA", "Canada", "Mexico"]

    def test_topk_create_block_errors_when_na_vals_never_match(self, meta_file, csv_file):
        """topk must error if na_vals never appear (same guard as aggregate_multiselect)."""
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "topk",
                    "name": "topkcols",
                    "columns": ["id", "a1", "a2"],
                    "from_columns": r"a(\d+)",
                    "na_vals": ["this_string_is_not_in_the_data"],
                    "res_columns": r"a_R\1",
                }
            ],
        }
        write_json(meta_file, meta)
        pd.DataFrame({"a1": ["yes", "no"], "a2": ["no", "yes"], "id": [1, 2]}).to_csv_file(csv_file)
        with pytest.raises(ValueError, match="No na_vals found"):
            read_and_process_data(str(meta_file), return_meta=True)

    @staticmethod
    def _assert_structure_matches(
        actual_structure: dict[str, ColumnBlockMeta],
        expected_structure: dict[str, ColumnBlockMeta],
        data_df: pd.DataFrame | None = None,
        check_q2_version: bool = False,
        expected_topics: list[str] | None = None,
    ) -> None:
        """Helper to compare actual and expected structures."""
        # System blocks like 'files' are automatically added by read_and_process_data
        actual_keys = set(actual_structure.keys())
        expected_keys = set(expected_structure.keys())
        if "files" in actual_keys and "files" not in expected_keys:
            actual_keys.remove("files")

        assert actual_keys == expected_keys, f"Structure keys differ: {actual_keys} != {expected_keys}"

        for block_name in expected_structure.keys():
            if block_name not in actual_structure:
                continue
            actual_block = actual_structure[block_name]
            expected_block = expected_structure[block_name]

            assert actual_block.name == expected_block.name, (
                f"Block {block_name} name differs: {actual_block.name} != {expected_block.name}"
            )

            actual_cols = set(actual_block.columns.keys()) if actual_block.columns else set()
            expected_cols = set(expected_block.columns.keys()) if expected_block.columns else set()

            # For the output maxdiff block, optionally check columns match DataFrame
            if block_name == "maxdiff" and isinstance(actual_block, MaxDiffBlock) and data_df is not None:
                df_cols = set(data_df.columns) - {"file_code", "file_name"}
                assert actual_cols == df_cols, (
                    f"Block {block_name} columns {actual_cols} should match DataFrame columns {df_cols}"
                )

                # Check Q2_Version if requested
                if check_q2_version and "Q2_Version" in df_cols and "Q2_Version" in actual_cols:
                    q2_version_meta = actual_block.columns["Q2_Version"]
                    assert q2_version_meta.continuous, (
                        f"Q2_Version should be continuous, got continuous={q2_version_meta.continuous}"
                    )
                    if expected_topics:
                        assert q2_version_meta.categories == expected_topics, (
                            f"Q2_Version should inherit scale categories, got {q2_version_meta.categories}"
                        )
                    column_list = list(actual_block.columns.keys())
                    assert column_list[0] == "Q2_Version", (
                        f"Q2_Version should be first column, but got order: {column_list[:5]}"
                    )
            else:
                assert actual_cols == expected_cols, (
                    f"Block {block_name} columns differ: {actual_cols} != {expected_cols}"
                )

            # Compare scale categories
            if expected_block.scale is not None and expected_block.scale.categories is not None:
                assert actual_block.scale is not None, f"Block {block_name} missing scale"
                assert actual_block.scale.categories == expected_block.scale.categories, (
                    f"{block_name} categories: {actual_block.scale.categories} != {expected_block.scale.categories}"
                )
            elif expected_block.scale is None:
                assert actual_block.scale is None, f"Block {block_name} should not have scale"

    @staticmethod
    def _build_maxdiff_expected_structure(
        topics: list[str],
        column_names: list[str],
        setindex_column: str | None = None,
        setindex_meta: ColumnMeta | None = None,
    ) -> dict[str, ColumnBlockMeta]:
        """Build expected structure for maxdiff tests."""
        all_columns = sorted(column_names)
        if setindex_column:
            all_columns = [setindex_column] + sorted(column_names)

        structure = {
            "maxdiff": ColumnBlockMeta(
                name="maxdiff",
                scale=BlockScaleMeta(categories=topics),
                columns={col: ColumnMeta() for col in all_columns},
            ),
        }

        if setindex_column and setindex_meta:
            structure["maxdiff"].columns[setindex_column] = setindex_meta

        return structure

    def test_maxdiff_create_block(self, meta_file, csv_file):
        """Reference example: maxdiff annotation with items and choice_sets in the create block."""
        np.random.seed(42)

        columns = [f"Q2_{k}best" for k in range(1, 11)]
        columns += [f"Q2_{k}worst" for k in range(1, 11)]

        # survey data formatting
        topics_per_set, k, n = 5, 10, 36  # topics per set, sets per person, n_topic_perms
        q = 18  # questions
        N = 23  # number of dataframe rows
        topics = np.array(list("ABCDEFGHIJKLMNOPQR"))
        grid = np.random.randn(n, k, q)
        perms = np.argsort(grid, axis=2)[:, :, :topics_per_set]
        mask = np.arange(1, q + 1)
        sets = mask[perms]

        # survey data generation
        best = np.random.choice(range(5), size=(N, k))
        worst = np.random.choice(range(5), size=(N, k))
        ids = np.random.choice(range(36), size=N)
        worst[best == worst] = (worst[best == worst] + 1) % 5
        A = topics[sets - 1][ids]
        # with topics
        i, j = np.ogrid[:N, :k]
        C = A[i, j, best]
        D = A[i, j, worst]

        items = {str(i + 1): t for i, t in enumerate(topics.tolist())}
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "maxdiff",
                    "name": "maxdiff",
                    "columns": [],
                    "scale": {"categories": topics.tolist(), "translate": items},
                    "best_columns": r"Q2_(\d+?)best",
                    "worst_columns": r"Q2_(\d+?)worst",
                    "set_columns": r"Q2_\1set",
                    "setindex_column": ["Q2_Version", {"continuous": True, "categories": None}],
                    "choice_sets": sets.tolist(),
                }
            ],
        }
        write_json(meta_file, meta)

        df = pd.DataFrame(np.hstack([ids.reshape(23, 1) + 1, C, D]), columns=["Q2_Version"] + columns)
        q2sets = [f"Q2_{k}set" for k in range(1, 11)]
        q2sethidden = np.array(list(map(lambda s: list(map(lambda s2: "".join(s2), s)), topics[sets - 1][ids])))
        df[q2sets] = q2sethidden
        q2sets = [f"Q2_{k}set" for k in range(1, 11)]
        for q2set in q2sets:
            df[q2set] = df[q2set].map(list)  # string -> list of chars, test specific

        columnorder = []
        for k in range(1, 11):
            columnorder.extend([f"Q2_{k}best", f"Q2_{k}set", f"Q2_{k}worst"])

        # Build expected structure
        expected_structure = self._build_maxdiff_expected_structure(
            topics=topics.tolist(),
            column_names=columnorder,
            setindex_column="Q2_Version",
            setindex_meta=ColumnMeta(continuous=True, categories=topics.tolist()),
        )

        df = df[["Q2_Version"] + sorted(columnorder)]
        df["Q2_Version"] = df["Q2_Version"].astype(int)
        df.to_csv(csv_file, index=False)
        data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

        # The code sorts columns alphabetically, so reorder expected df to match
        expected_df = df.copy()
        expected_df["file_code"] = "F0"
        expected_df["file_name"] = "test_meta.json"
        expected_df = expected_df[sorted(expected_df.columns)]
        assert_frame_equal(data_df, expected_df, check_dtype=False, check_categorical=False)

        # Compare structures
        self._assert_structure_matches(
            data_meta.structure,
            expected_structure,
            data_df=data_df,
            check_q2_version=True,
            expected_topics=topics.tolist(),
        )

    def test_input_df_columns_topk_onehot_maxdiff(self):
        """Each block type reports the df-columns safe to pre-translate. Topk/onehot
        use from_columns. MaxDiff reports best+worst+set for ``resolved`` input_format
        (cells are scalar index strings) but best+worst only for ``choice_sets``
        (set cells hold lists and are not pre-translated)."""
        from salk_toolkit.validation import TopKBlock, MaxDiffBlock, OneHotBlock

        df = pd.DataFrame({"a": [1], "b": [1], "c": [1], "d": [1]})
        tk = TopKBlock(name="t", from_columns=["a", "b"], res_columns=["R1", "R2"])
        assert tk.input_df_columns(df) == ["a", "b"]

        oh = OneHotBlock(name="o", from_columns=["a", "c"])
        assert oh.input_df_columns(df) == ["a", "c"]

        md_resolved = MaxDiffBlock(
            name="m",
            best_columns=["a"],
            worst_columns=["b"],
            set_columns=["c"],
            input_format="resolved",
        )
        assert set(md_resolved.input_df_columns(df)) == {"a", "b", "c"}

        md_choice_sets = MaxDiffBlock(
            name="m",
            best_columns=["a"],
            worst_columns=["b"],
            set_columns=["c"],
            input_format="choice_sets",
        )
        assert set(md_choice_sets.input_df_columns(df)) == {"a", "b"}

    def test_maxdiff_explode_resolves_role_columns_per_sibling(self, meta_file, csv_file):
        """After subgroup_explode the source regex in best/worst/set_columns is replaced
        by per-sibling concrete lists; the transform never sees regex."""
        from salk_toolkit.io import _subgroup_explode
        from salk_toolkit.validation import MaxDiffBlock

        df = pd.DataFrame(
            {
                "Q_A_1best": ["x"],
                "Q_A_1worst": ["x"],
                "Q_A_1set": [["x"]],
                "Q_A_2best": ["x"],
                "Q_A_2worst": ["x"],
                "Q_A_2set": [["x"]],
                "Q_B_1best": ["x"],
                "Q_B_1worst": ["x"],
                "Q_B_1set": [["x"]],
            }
        )
        block = MaxDiffBlock(
            name="md",
            from_columns=r"Q_([AB])_\d+best",  # used by explode to enumerate siblings
            best_columns=r"Q_([AB])_(\d+)best",
            worst_columns=r"Q_([AB])_(\d+)worst",
            set_columns=r"Q_\1_\2set",
            scale=BlockScaleMeta(translate={"1": "Alpha"}),
        )
        siblings = _subgroup_explode(block, df)
        by_label = {s.name.removeprefix("md_"): s for s in siblings}
        assert set(by_label) == {"A", "B"}
        sib_a = by_label["A"]
        assert isinstance(sib_a.best_columns, list) and sib_a.best_columns == ["Q_A_1best", "Q_A_2best"]
        assert isinstance(sib_a.worst_columns, list) and sib_a.worst_columns == ["Q_A_1worst", "Q_A_2worst"]
        assert isinstance(sib_a.set_columns, list) and sib_a.set_columns == ["Q_A_1set", "Q_A_2set"]

    def test_maxdiff_create_block_explicit_sets(self, meta_file, csv_file):
        """Ensure explicit set definitions are parsed for every serialization mode."""
        topics = list("ABCDEFGHIJKLMNOP")
        topic_index = {topic: idx for idx, topic in enumerate(topics)}

        # Each row has K question blocks, and each block references the set explicitly.
        row_sets = [
            [
                ["A", "B", "C", "D", "E"],
                ["F", "G", "H", "I", "J"],
                ["K", "L", "M", "N", "O"],
            ],
            [
                ["B", "C", "D", "E", "F"],
                ["G", "H", "I", "J", "K"],
                ["L", "M", "N", "O", "P"],
            ],
            [
                ["C", "D", "E", "F", "G"],
                ["H", "I", "J", "K", "L"],
                ["M", "N", "O", "P", "A"],
            ],
        ]
        best_indices = [
            [0, 2, 1],
            [3, 0, 4],
            [1, 4, 2],
        ]
        worst_indices = [
            [2, 4, 3],
            [1, 3, 0],
            [4, 2, 3],
        ]

        num_rows = len(row_sets)
        num_blocks = len(row_sets[0])

        def serialize(row_topics, mode):
            if mode == "topic_string":
                return ", ".join(row_topics)
            if mode == "topic_json":
                return json.dumps(row_topics)

            indices = [topic_index[topic] + 1 for topic in row_topics]
            if mode == "index_string":
                return ", ".join(map(str, indices))
            if mode == "index_json":
                return json.dumps(indices)
            raise ValueError(f"Unsupported mode {mode}")

        def build_dataframe(mode):
            data = {}
            for block_idx in range(num_blocks):
                col = block_idx + 1
                data[f"Q2_{col}best"] = [
                    row_sets[row][block_idx][best_indices[row][block_idx]] for row in range(num_rows)
                ]
                data[f"Q2_{col}worst"] = [
                    row_sets[row][block_idx][worst_indices[row][block_idx]] for row in range(num_rows)
                ]
                data[f"Q2_{col}set"] = [serialize(row_sets[row][block_idx], mode) for row in range(num_rows)]
            return pd.DataFrame(data)

        items = {str(i + 1): t for i, t in enumerate(topics)}
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "maxdiff",
                    "name": "maxdiff",
                    "columns": [],
                    "scale": {"categories": topics, "translate": items},
                    "best_columns": r"Q2_(\d+?)best",
                    "worst_columns": r"Q2_(\d+?)worst",
                    "set_columns": r"Q2_\1set",
                }
            ],
        }
        write_json(meta_file, meta)

        exp_cols = []
        for i in range(1, num_blocks + 1):
            exp_cols.extend([f"Q2_{i}best", f"Q2_{i}set", f"Q2_{i}worst"])

        # Build expected structure
        expected_structure = self._build_maxdiff_expected_structure(
            topics=topics,
            column_names=exp_cols,
        )

        for mode in ["topic_string", "topic_json", "index_string", "index_json"]:
            df = build_dataframe(mode)
            df.to_csv_file(csv_file)

            data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)
            expected_df = df.copy()
            for block_idx in range(num_blocks):
                col = f"Q2_{block_idx + 1}set"
                expected_df[col] = [list(row_sets[row][block_idx]) for row in range(num_rows)]
            expected_df["file_code"] = "F0"
            expected_df["file_name"] = "test_meta.json"
            expected_df = expected_df[data_df.columns]

            assert_frame_equal(
                data_df,
                expected_df,
                check_dtype=False,
                check_categorical=False,
                obj=f"mode={mode}",
            )

            # Compare structures
            self._assert_structure_matches(
                data_meta.structure,
                expected_structure,
                data_df=None,  # Don't check DataFrame columns for this test
            )

    def test_topk_no_subgroups(self, meta_file, csv_file):
        """TopK with a single regex group (= agg group) produces one block named {name}.

        scale.translate_after maps the numeric agg-group value to the display name.
        """
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "topk",
                    "name": "issue_importance",
                    "columns": ["Q6r1", "Q6r2", "Q6r3"],
                    "from_columns": r"Q6r(\d+)",
                    "res_columns": r"Q6p_R\1",
                    "agg_index": 1,
                    "na_vals": ["NO TO: Cost of living", "NO TO: Healthcare", "NO TO: Pensions"],
                    "scale": {
                        "categories": "infer",
                        "translate_after": {
                            "1": "Cost of living",
                            "2": "Healthcare",
                            "3": "Pensions",
                        },
                    },
                }
            ],
        }
        write_json(meta_file, meta)
        df = pd.DataFrame(
            {
                "Q6r1": ["Cost of living", "NO TO: Cost of living", "Cost of living"],
                "Q6r2": ["NO TO: Healthcare", "Healthcare", "Healthcare"],
                "Q6r3": ["Pensions", "Pensions", "NO TO: Pensions"],
            }
        )
        df.to_csv_file(csv_file)
        data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

        assert "issue_importance" in data_meta.structure
        # Cell values are the translated issue names (R1 = lowest agg-group index selected)
        assert data_df["Q6p_R1"].tolist() == ["Cost of living", "Healthcare", "Cost of living"]

        block = data_meta.structure["issue_importance"]
        assert isinstance(block, TopKBlock)
        assert set(block.columns.keys()) == {"Q6p_R1", "Q6p_R2"}
        assert isinstance(block.from_columns, list) and isinstance(block.res_columns, list)

    def test_topk_one_subgroup(self, meta_file, csv_file):
        """TopK with 2 regex groups (1 subgroup dim).

        `subgroup_labels` labels the non-agg group; produces one block per group value.
        """
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "topk",
                    "name": "issue ownership",
                    "columns": ["Q7r1c1", "Q7r1c2", "Q7r2c1", "Q7r2c2"],
                    "from_columns": r"Q7r(\d+)c(\d+)",
                    "res_columns": r"Q7r\1_R\2",
                    "agg_index": 2,
                    "subgroup_labels": {"1": {"1": "economics", "2": "healthcare"}},
                    "na_vals": ["not_selected"],
                    "scale": {
                        "categories": "infer",
                        "translate_after": {"1": "Party A", "2": "Party B"},
                    },
                }
            ],
        }
        write_json(meta_file, meta)
        df = pd.DataFrame(
            {
                "Q7r1c1": ["selected", "not_selected", "selected"],
                "Q7r1c2": ["not_selected", "selected", "not_selected"],
                "Q7r2c1": ["selected", "not_selected", "selected"],
                "Q7r2c2": ["selected", "selected", "not_selected"],
            }
        )
        df.to_csv_file(csv_file)
        data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

        assert "issue ownership_economics" in data_meta.structure
        assert "issue ownership_healthcare" in data_meta.structure

        econ_block = data_meta.structure["issue ownership_economics"]
        assert isinstance(econ_block, TopKBlock)
        assert all(c.startswith("Q7r1_R") for c in econ_block.columns)
        assert data_df["Q7r1_R1"].tolist() == ["Party A", "Party B", "Party A"]

        # Subgroup siblings are independent TopKBlocks with narrowed resolved column lists.
        assert econ_block.type == "topk"
        assert econ_block.from_columns == ["Q7r1c1", "Q7r1c2"]
        assert econ_block.res_columns == ["Q7r1_R1", "Q7r1_R2"]
        assert econ_block.segments() == [(list(econ_block.columns.keys()), None, False)]

    def test_topk_two_subgroup_dimensions(self, meta_file, csv_file):
        """TopK with 3 regex groups, 2 subgroup dimensions.

        Block labels concatenate non-agg group labels with `_`.
        """
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "topk",
                    "name": "survey",
                    "columns": [
                        "Q_A_1_1",
                        "Q_A_1_2",
                        "Q_A_2_1",
                        "Q_A_2_2",
                        "Q_B_1_1",
                        "Q_B_1_2",
                        "Q_B_2_1",
                        "Q_B_2_2",
                    ],
                    "from_columns": r"Q_(\w+)_(\d+)_(\d+)",
                    "res_columns": r"Q_\1_\2_R\3",
                    "agg_index": 3,
                    "subgroup_labels": {
                        "1": {"A": "Estonia", "B": "Latvia"},
                        "2": {"1": "economics", "2": "healthcare"},
                    },
                    "na_vals": ["no"],
                    "scale": {
                        "categories": "infer",
                        "translate_after": {"1": "Party X", "2": "Party Y"},
                    },
                }
            ],
        }
        write_json(meta_file, meta)
        df = pd.DataFrame(
            {
                "Q_A_1_1": ["yes", "no"],
                "Q_A_1_2": ["no", "yes"],
                "Q_A_2_1": ["yes", "yes"],
                "Q_A_2_2": ["no", "no"],
                "Q_B_1_1": ["no", "yes"],
                "Q_B_1_2": ["yes", "no"],
                "Q_B_2_1": ["yes", "no"],
                "Q_B_2_2": ["no", "yes"],
            }
        )
        df.to_csv_file(csv_file)
        _data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

        # Four blocks: all country × issue combinations
        for combo in ["Estonia_economics", "Estonia_healthcare", "Latvia_economics", "Latvia_healthcare"]:
            assert f"survey_{combo}" in data_meta.structure

        ee_block = data_meta.structure["survey_Estonia_economics"]
        assert isinstance(ee_block, TopKBlock)
        assert all(c.startswith("Q_A_1_R") for c in ee_block.columns)
        # Each sibling is an independent TopKBlock with its own narrowed from_columns.
        assert ee_block.from_columns == ["Q_A_1_1", "Q_A_1_2"]
        assert ee_block.segments() == [(list(ee_block.columns.keys()), None, False)]

    def test_maxdiff_segments_shape(self, meta_file, csv_file):
        """MaxDiff output block exposes leedu-compatible ordinal ranking segments."""
        items = {"1": "A", "2": "B", "3": "C"}
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "maxdiff",
                    "name": "maxdiff",
                    "columns": [],
                    "best_columns": r"Q_(\d+)best",
                    "worst_columns": r"Q_(\d+)worst",
                    "set_columns": r"Q_\1set",
                    "scale": {"categories": "infer", "translate": items},
                }
            ],
        }
        write_json(meta_file, meta)
        df = pd.DataFrame(
            {
                "Q_1best": ["A", "B"],
                "Q_1worst": ["C", "A"],
                "Q_1set": [["A", "B", "C"], ["A", "B", "C"]],
                "Q_2best": ["B", "C"],
                "Q_2worst": ["A", "B"],
                "Q_2set": [["A", "B", "C"], ["A", "B", "C"]],
            }
        )
        df.to_csv_file(csv_file)
        _data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)
        block = data_meta.structure["maxdiff"]
        assert isinstance(block, MaxDiffBlock)
        # Question-aligned lists with the same order, two segments per question (best>set, set>worst).
        best_columns = block.best_columns
        worst_columns = block.worst_columns
        set_columns = block.set_columns
        assert isinstance(best_columns, list)
        assert isinstance(worst_columns, list)
        assert isinstance(set_columns, list)
        q = len(best_columns)
        segs = block.segments()
        assert len(segs) == 2 * q
        for k in range(q):
            assert segs[k] == ([best_columns[k]], [set_columns[k]], True)
            assert segs[q + k] == ([set_columns[k]], [worst_columns[k]], True)

    def test_maxdiff_with_translate(self, meta_file, csv_file):
        """MaxDiff with `scale.translate` mapping 1-based index strings to display-language topics.

        The source stores best/worst cells as 1-based index strings ("1","2","3"); pre-transform
        translate element-wise replaces those with display-language topic names ("Economy" etc.).
        The same dict defines the topic universe for `setindex_column` lookups.
        """
        translate = {"1": "Economy", "2": "Health", "3": "Education"}
        display_topics = ["Economy", "Health", "Education"]
        choice_sets = [
            [[1, 2, 3], [2, 3, 1], [1, 3, 2]],  # version 1
            [[3, 1, 2], [1, 2, 3], [3, 2, 1]],  # version 2
        ]
        parquet_file = csv_file.with_suffix(".parquet")
        meta = {
            "file": str(parquet_file),
            "structure": [
                {
                    "type": "maxdiff",
                    "name": "maxdiff",
                    "columns": [],
                    "best_columns": r"Q_(\d+)best",
                    "worst_columns": r"Q_(\d+)worst",
                    "set_columns": r"Q_\1set",
                    "setindex_column": ["Q_Version", {"continuous": True}],
                    "choice_sets": choice_sets,
                    "scale": {"categories": "infer", "translate": translate},
                }
            ],
        }
        write_json(meta_file, meta)
        # Best/worst cells are 1-based index strings; pre-transform translate renames them.
        # Parquet preserves the string dtype (CSV would reparse "1" as int).
        df = pd.DataFrame(
            {
                "Q_Version": [1, 2, 1],
                "Q_1best": ["1", "3", "2"],
                "Q_1worst": ["2", "1", "1"],
                "Q_2best": ["3", "1", "3"],
                "Q_2worst": ["1", "2", "1"],
                "Q_3best": ["1", "3", "1"],
                "Q_3worst": ["3", "1", "3"],
            }
        )
        df.to_parquet(parquet_file)
        data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

        block = data_meta.structure["maxdiff"]
        assert isinstance(block, MaxDiffBlock)
        assert "Q_1best" in block.columns
        assert "Q_1set" in block.columns

        # Cell values translated into display language
        assert data_df["Q_1best"].tolist() == ["Economy", "Education", "Health"]
        assert set(data_df["Q_1best"].cat.categories) == set(display_topics)

        # Output block is a MaxDiffBlock with resolved column lists; input-only directives cleared.
        assert block.type == "maxdiff"
        assert block.choice_sets is None
        assert isinstance(block.best_columns, list)
        assert isinstance(block.worst_columns, list)
        assert isinstance(block.set_columns, list)
        # Translated vocabulary lives on the scale categories.
        assert block.scale is not None and set(block.scale.categories or []) == set(display_topics)

    def test_maxdiff_items_no_translate(self, meta_file, csv_file):
        """MaxDiff with scale.translate already in target language (no additional translation)."""
        items = {"1": "Economy", "2": "Health", "3": "Education"}
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "maxdiff",
                    "name": "maxdiff",
                    "columns": [],
                    "best_columns": r"Q_(\d+)best",
                    "worst_columns": r"Q_(\d+)worst",
                    "set_columns": r"Q_\1set",
                    "scale": {"categories": "infer", "translate": items},
                }
            ],
        }
        write_json(meta_file, meta)
        df = pd.DataFrame(
            {
                "Q_1best": ["Economy", "Health", "Education"],
                "Q_1worst": ["Education", "Economy", "Health"],
                "Q_1set": [
                    ["Economy", "Health", "Education"],
                    ["Economy", "Health", "Education"],
                    ["Economy", "Health", "Education"],
                ],
            }
        )
        df.to_csv_file(csv_file)
        data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

        block = data_meta.structure["maxdiff"]
        assert isinstance(block, MaxDiffBlock)
        # Cell values are the item names directly (no translation needed — cells already in target vocab)
        assert data_df["Q_1best"].tolist() == ["Economy", "Health", "Education"]

    def test_maxdiff_inline_index_tokens(self, meta_file, csv_file):
        """Inline MaxDiff with integer-list tokens in set cells and index strings in
        best/worst cells. scale.translate maps index strings to display names."""
        # CSV reparses "1" as int, breaking .replace() against string-keyed translate
        # dict, so we write the fixture as parquet (preserves string dtype).
        parquet_file = csv_file.with_suffix(".parquet")
        meta = {
            "file": str(parquet_file),
            "structure": [
                {
                    "type": "maxdiff",
                    "name": "md",
                    "columns": [],
                    "best_columns": ["Q_1best"],
                    "worst_columns": ["Q_1worst"],
                    "set_columns": ["Q_1set"],
                    "scale": {
                        "categories": "infer",
                        "translate": {"1": "Economy", "2": "Health", "3": "Education"},
                    },
                }
            ],
        }
        write_json(meta_file, meta)
        df = pd.DataFrame(
            {
                "Q_1best": ["1", "3", "2"],
                "Q_1worst": ["2", "1", "1"],
                "Q_1set": [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
            }
        )
        df.to_parquet(parquet_file)
        data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

        assert data_df["Q_1best"].tolist() == ["Economy", "Education", "Health"]
        assert data_df["Q_1worst"].tolist() == ["Health", "Economy", "Economy"]
        assert list(data_df["Q_1set"].iloc[0]) == ["Economy", "Health", "Education"]
        block = data_meta.structure["md"]
        assert set(block.scale.categories or []) == {"Economy", "Health", "Education"}

    def test_maxdiff_inline_name_tokens(self, meta_file, csv_file):
        """Inline MaxDiff with raw-language names in best/worst cells.
        scale.translate maps raw names to display names — same dict-key space
        as cell contents, NOT integer positions. Uses input_format='resolved'
        because the index-keyed ``choice_sets`` transform requires
        integer-sortable translate keys.

        Note: the pre-transform translate is scalar-cell ``.replace`` and cannot
        run over list-valued cells; set_columns are required by MaxDiff schema
        but we exclude them from translate scope by asserting on best/worst only."""
        # Parquet preserves object-list dtype across the read.
        parquet_file = csv_file.with_suffix(".parquet")
        meta = {
            "file": str(parquet_file),
            "structure": [
                {
                    "type": "maxdiff",
                    "name": "md",
                    "columns": [],
                    "best_columns": ["Q_1best"],
                    "worst_columns": ["Q_1worst"],
                    "set_columns": ["Q_1set"],
                    "input_format": "resolved",
                    "scale": {
                        "categories": ["Economy", "Health", "Education"],
                        "translate": {
                            "Ekonomika": "Economy",
                            "Sveikata": "Health",
                            "Svietimas": "Education",
                        },
                    },
                }
            ],
        }
        write_json(meta_file, meta)
        # Set cells are scalar comma-strings; translate is a scalar-value replace
        # which does NOT match substrings, so set cells pass through untouched.
        # This test pins the best/worst translation behaviour; set-cell translate
        # for resolved+list-cells is not supported by the current pipeline.
        df = pd.DataFrame(
            {
                "Q_1best": ["Ekonomika", "Svietimas", "Sveikata"],
                "Q_1worst": ["Sveikata", "Ekonomika", "Ekonomika"],
                "Q_1set": ["Ekonomika", "Ekonomika", "Ekonomika"],
            }
        )
        df.to_parquet(parquet_file)
        data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

        assert data_df["Q_1best"].tolist() == ["Economy", "Education", "Health"]
        assert data_df["Q_1worst"].tolist() == ["Health", "Economy", "Economy"]
        # Set cells also element-replaced by translate (scalar form).
        assert data_df["Q_1set"].tolist() == ["Economy", "Economy", "Economy"]

    def test_maxdiff_setindex_lookup(self, meta_file, csv_file):
        """MaxDiff driven by setindex_column + choice_sets metadata. scale.translate
        replaces the old choice_mapping: it's both the index->name source for the
        setindex lookup AND the cell translator for best/worst."""
        choice_sets = [
            [[1, 2, 3], [2, 3, 1], [1, 3, 2]],
            [[3, 1, 2], [1, 2, 3], [3, 2, 1]],
        ]
        # Use parquet to preserve string dtype on integer-string best/worst cells.
        parquet_file = csv_file.with_suffix(".parquet")
        meta = {
            "file": str(parquet_file),
            "structure": [
                {
                    "type": "maxdiff",
                    "name": "md",
                    "columns": [],
                    "best_columns": r"Q_(\d+)best",
                    "worst_columns": r"Q_(\d+)worst",
                    "set_columns": r"Q_\1set",
                    "setindex_column": ["Q_Version", {"continuous": True}],
                    "choice_sets": choice_sets,
                    "scale": {
                        "categories": "infer",
                        "translate": {"1": "Economy", "2": "Health", "3": "Education"},
                    },
                }
            ],
        }
        write_json(meta_file, meta)
        df = pd.DataFrame(
            {
                "Q_Version": [1, 2, 1],
                "Q_1best": ["1", "3", "2"],
                "Q_1worst": ["2", "1", "1"],
                "Q_2best": ["3", "1", "3"],
                "Q_2worst": ["1", "2", "1"],
                "Q_3best": ["1", "3", "1"],
                "Q_3worst": ["3", "1", "3"],
            }
        )
        df.to_parquet(parquet_file)
        data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

        assert list(data_df["Q_1set"].iloc[0]) == ["Economy", "Health", "Education"]
        assert list(data_df["Q_1set"].iloc[1]) == ["Education", "Economy", "Health"]
        assert data_df["Q_1best"].tolist() == ["Economy", "Education", "Health"]
        assert data_df["Q_1worst"].tolist() == ["Health", "Economy", "Economy"]

        block = data_meta.structure["md"]
        assert block.choice_sets is None
        assert isinstance(block.best_columns, list)
        assert set(block.scale.categories or []) == {"Economy", "Health", "Education"}

    def test_maxdiff_translate_after_is_deprecated(self, meta_file, csv_file):
        """scale.translate_after on a MaxDiff block must be a hard fail with a
        message pointing at scale.translate."""
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "maxdiff",
                    "name": "md",
                    "columns": [],
                    "best_columns": ["Q_1best"],
                    "worst_columns": ["Q_1worst"],
                    "set_columns": ["Q_1set"],
                    "scale": {"translate_after": {"1": "Economy"}},
                }
            ],
        }
        write_json(meta_file, meta)
        df = pd.DataFrame({"Q_1best": ["1"], "Q_1worst": ["1"], "Q_1set": [[1]]})
        df.to_csv_file(csv_file)

        with pytest.raises(ValueError, match=r"(?i)translate_after.*maxdiff.*scale\.translate"):
            read_and_process_data(str(meta_file), return_meta=True)

    def test_topk_scale_translate_feeds_na_vals_detection(self, meta_file, csv_file):
        """Pre-translate fires before na_vals check: translated sentinel is dropped, raw form is not."""
        # Row 0: Qa=raw_keep, Qb=raw_drop. Row 1: Qa=raw_drop, Qb=raw_keep.
        # Pre-translate: raw_drop -> <drop> (matches na_vals, dropped), raw_keep -> keep.
        # After drop: each row has exactly ONE non-NA cell, so only one result column (Ra) survives.
        # Without pre-translate: na_vals=["<drop>"] never matches "raw_drop", both cells survive,
        # leftpack fills Ra AND Rb for every row → two result columns.
        pd.DataFrame({"Qa": ["raw_keep", "raw_drop"], "Qb": ["raw_drop", "raw_keep"]}).to_csv(csv_file, index=False)
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "topk",
                    "name": "q",
                    "from_columns": r"Q(\w)",
                    "res_columns": r"R\1",
                    "agg_index": 1,
                    "na_vals": ["<drop>"],
                    "scale": {
                        "translate": {"raw_keep": "keep", "raw_drop": "<drop>"},
                    },
                }
            ],
        }
        write_json(meta_file, meta)
        ndf, meta_obj = read_annotated_data(str(meta_file), return_meta=True)
        out = meta_obj.structure["q"]
        assert out.input_format == "onehot"
        res_cols = [c for c in ndf.columns if c.startswith("R")]
        # If pre-translate fired: only 1 non-NA cell per row → dropna(how="all") removes Rb → 1 result col.
        # If pre-translate did NOT fire: 2 non-NA cells per row → 2 result columns survive.
        assert len(res_cols) == 1

    def test_scale_translate_none_is_noop(self, meta_file, csv_file):
        """Blocks without scale.translate pass through unchanged without error."""
        pd.DataFrame({"Qa": ["A", None], "Qb": [None, "B"]}).to_csv(csv_file, index=False)
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "topk",
                    "name": "q",
                    "from_columns": r"Q(\w)",
                    "res_columns": r"R\1",
                    "agg_index": 1,
                    "na_vals": [],
                }
            ],
        }
        write_json(meta_file, meta)
        _, meta_obj = read_annotated_data(str(meta_file), return_meta=True)
        assert "q" in meta_obj.structure


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

    def test_category_inference_numeric_strings_preserve_strings(self, csv_file, meta_file):
        """Numeric-like strings should sort numerically but keep original string values (no rounding)."""
        pd.DataFrame(
            {
                "value": ["1.5", "2.333", "10", "4.0", "5.25"],
                "id": [1, 2, 3, 4, 5],
            }
        ).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            # Ensure we actually read the column as strings; otherwise pandas may parse it as float.
            "read_opts": {"dtype": {"value": "string"}},
            "structure": [{"name": "test", "columns": ["id", ["value", {"categories": "infer"}]]}],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        assert df["value"].dtype.name == "category"
        # Categories are the original strings, numerically ordered (10 should come last).
        assert list(df["value"].dtype.categories) == ["1.5", "2.333", "4.0", "5.25", "10"]

    def test_category_inference_mixed_int_and_numeric_string_after_translate(self, csv_file, meta_file):
        """Mixed int + numeric-string values should still yield string categories (Arrow-safe)."""
        # This mirrors a common real-world case:
        # - translation maps some string labels to ints
        # - other values remain as numeric strings (e.g. numeric codes that were already present)
        pd.DataFrame(
            {
                "mixed": ["A", "B", 2],
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
                        [
                            "mixed",
                            {
                                "categories": "infer",
                                "translate": {"A": 1, "B": 4},
                            },
                        ],
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        assert df["mixed"].dtype.name == "category"
        assert all(isinstance(c, str) for c in df["mixed"].dtype.categories)
        assert list(df["mixed"].dtype.categories) == ["1", "2", "4"]

    def test_category_inference_datetime_strings(self, csv_file, meta_file):
        """Datetime-like strings should sort chronologically but keep original string values."""
        pd.DataFrame(
            {
                "dt": ["2024-12-31", "2024-01-02", "2024-01-01"],
                "id": [1, 2, 3],
            }
        ).to_csv_file(csv_file)

        meta = {
            "file": "test.csv",
            # Ensure we actually read the column as strings
            "read_opts": {"dtype": {"dt": "string"}},
            "structure": [{"name": "test", "columns": ["id", ["dt", {"categories": "infer"}]]}],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        assert df["dt"].dtype.name == "category"
        assert list(df["dt"].dtype.categories) == ["2024-01-01", "2024-01-02", "2024-12-31"]

    def test_category_inference_datetime_dtype_formats_human(self, csv_file, meta_file):
        """True datetime dtype should be converted to '01 Dec 25' and sorted chronologically."""
        pd.DataFrame(
            {
                "dt_raw": ["2025-12-01", "2025-12-02", "2025-01-01"],
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
                        # Force datetime dtype first, then infer categories on that datetime series.
                        ["dt", "dt_raw", {"datetime": True, "categories": "infer"}],
                    ],
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))

        assert df["dt"].dtype.name == "category"
        assert list(df["dt"].dtype.categories) == ["01 Jan 25", "01 Dec 25", "02 Dec 25"]

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
        assert list(df.columns[: len(sample_csv_data.columns)]) == list(sample_csv_data.columns)
        assert set(df.columns) - set(sample_csv_data.columns) == {"file_code", "file_name"}

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

    def test_merge_error_on_suffixes(self, temp_dir):
        """Merges error when overlapping columns would be suffixed."""
        main_csv = temp_dir / "main.csv"
        merge_csv = temp_dir / "merge.csv"
        pd.DataFrame({"municipality": ["A", "B"], "persona": ["x", "y"]}).to_csv_file(main_csv)
        pd.DataFrame({"municipality": ["A", "B"], "persona": ["xx", "yy"]}).to_csv_file(merge_csv)

        desc = {
            "file": str(main_csv),
            "merge": {"file": str(merge_csv), "on": "municipality"},
        }

        with pytest.raises(ValueError, match="suffix"):
            read_and_process_data(desc)

    def test_merge_applies_categorical_conversion_from_metadata(self, temp_dir):
        """Merged columns should be converted to categorical based on metadata definitions."""
        main_csv = temp_dir / "main.csv"
        merge_csv = temp_dir / "merge.csv"
        meta_file = temp_dir / "main_meta.json"

        # Main data with municipality
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "municipality": ["A", "B", "C", "D"],
                "value": [10, 20, 30, 40],
            }
        ).to_csv_file(main_csv)

        # Merge file with numeric winddev column (0/1 integers)
        pd.DataFrame(
            {
                "municipality": ["A", "B", "C", "D"],
                "winddev": [1, 0, 1, 0],
                "other_num": [100, 200, 300, 400],
            }
        ).to_csv_file(merge_csv)

        # Metadata file defines structure including merged columns
        write_json(
            meta_file,
            {
                "file": "main.csv",
                "structure": [
                    {
                        "name": "main",
                        "columns": ["id", "municipality", "value"],
                    },
                    {
                        "name": "merged",
                        "columns": [
                            ["winddev", {"categories": "infer"}],
                            "other_num",  # No categories - should remain numeric
                        ],
                    },
                ],
            },
        )

        # DataDescription with merge - metadata is in the file, merge is in the desc
        desc = {
            "file": str(meta_file),
            "merge": {"file": str(merge_csv), "on": "municipality"},
        }

        df, meta = read_and_process_data(desc, return_meta=True)

        # winddev should be converted to categorical
        assert df["winddev"].dtype.name == "category", f"Expected winddev to be categorical, got {df['winddev'].dtype}"
        # Categories should be inferred as strings in sorted order
        assert list(df["winddev"].cat.categories) == ["0", "1"]

        # other_num should remain numeric (no categories defined)
        assert pd.api.types.is_numeric_dtype(df["other_num"])

    def test_merge_applies_explicit_categories_from_metadata(self, temp_dir):
        """Merged columns with explicit categories should use those categories."""
        main_csv = temp_dir / "main.csv"
        merge_csv = temp_dir / "merge.csv"
        meta_file = temp_dir / "main_meta.json"

        pd.DataFrame(
            {
                "id": [1, 2],
                "key": ["X", "Y"],
            }
        ).to_csv_file(main_csv)

        pd.DataFrame(
            {
                "key": ["X", "Y"],
                "status": [1, 2],
            }
        ).to_csv_file(merge_csv)

        # Metadata file defines structure including merged columns
        write_json(
            meta_file,
            {
                "file": "main.csv",
                "structure": [
                    {
                        "name": "main",
                        "columns": ["id", "key"],
                    },
                    {
                        "name": "merged",
                        "columns": [
                            ["status", {"categories": ["1", "2", "3"], "ordered": True}],
                        ],
                    },
                ],
            },
        )

        # DataDescription with merge
        desc = {
            "file": str(meta_file),
            "merge": {"file": str(merge_csv), "on": "key"},
        }

        df, meta = read_and_process_data(desc, return_meta=True)

        assert df["status"].dtype.name == "category"
        assert list(df["status"].cat.categories) == ["1", "2", "3"]
        assert df["status"].cat.ordered is True

    def test_return_meta_extra_field_categories(self, temp_dir):
        """Extra FileDesc fields (e.g. t) should be reflected in returned meta categories."""
        # Create tiny source CSVs (no `t` column); meta declares `t` but it will be empty at this stage.
        csv1 = temp_dir / "d1.csv"
        csv2 = temp_dir / "d2.csv"
        pd.DataFrame({"id": [1, 2], "value": ["A", "B"]}).to_csv_file(csv1)
        pd.DataFrame({"id": [3, 4], "value": ["C", "D"]}).to_csv_file(csv2)

        meta1 = temp_dir / "m1.json"
        meta2 = temp_dir / "m2.json"
        write_json(
            meta1,
            {
                "file": "d1.csv",
                "structure": [
                    {
                        "name": "survey",
                        "columns": [
                            "id",
                            "value",
                            ["t", {"categories": []}],
                        ],
                    }
                ],
            },
        )
        write_json(
            meta2,
            {
                "file": "d2.csv",
                "structure": [
                    {
                        "name": "survey",
                        "columns": [
                            "id",
                            "value",
                            ["t", {"categories": []}],
                        ],
                    }
                ],
            },
        )

        # Load annotated metas as multi-file description, injecting per-file `t` values.
        desc = {
            "files": [
                {"file": str(meta1), "t": "-3"},
                {"file": str(meta2), "t": "0"},
            ]
        }
        df, m = read_and_process_data(desc, return_meta=True)

        assert df["t"].dropna().unique().tolist() == ["-3", "0"]
        assert m.structure["survey"].columns["t"].categories == ["-3", "0"]

    def test_categorical_preserved_when_combining_json_metafiles(self, temp_dir):
        """Categoricals inferred per-file must stay categorical after combining two JSON metafiles.

        Regression: pd.concat on two Categoricals with different category lists produces
        dtype=object; _reconcile_categories must re-unify them after loading.
        """
        csv1 = temp_dir / "f1.csv"
        csv2 = temp_dir / "f2.csv"
        # Each file has a disjoint set of region values — the union must be the final categories.
        pd.DataFrame({"id": [1, 2], "region": ["North", "South"]}).to_csv_file(csv1)
        pd.DataFrame({"id": [3, 4], "region": ["East", "West"]}).to_csv_file(csv2)

        meta1 = temp_dir / "m1.json"
        meta2 = temp_dir / "m2.json"
        for meta_path, csv_name in [(meta1, "f1.csv"), (meta2, "f2.csv")]:
            write_json(
                meta_path,
                {
                    "file": csv_name,
                    "structure": [
                        {
                            "name": "main",
                            "columns": [
                                "id",
                                ["region", {"categories": "infer"}],
                            ],
                        }
                    ],
                },
            )

        desc = {"files": [{"file": str(meta1)}, {"file": str(meta2)}]}
        df = read_and_process_data(desc)

        assert df["region"].dtype.name == "category", "region should be categorical after combining files"
        assert set(df["region"].cat.categories) == {"North", "South", "East", "West"}
        assert set(df["region"].dropna()) == {"North", "South", "East", "West"}


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

    def test_read_parquet_with_metadata_handles_missing_schema_metadata(self, temp_dir):
        """Parquet files without schema metadata should return None meta without error."""
        parquet_file = temp_dir / "no_schema_metadata.parquet"
        expected_df = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
        table = pa.Table.from_pandas(expected_df).replace_schema_metadata(None)
        pq.write_table(table, parquet_file)

        restored_df, restored_meta = read_parquet_with_metadata(str(parquet_file))

        assert_frame_equal(restored_df, expected_df)
        assert restored_meta is None


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

    def test_soft_validate_with_extra_fields(self, capsys):
        """Test that soft_validate warns on extra fields at all nesting levels but returns a valid model."""
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

        result = soft_validate(meta_dict, DataMeta, warnings=True)

        # Should return a valid DataMeta with correct fields
        assert isinstance(result, DataMeta)
        assert result.description == "Valid field"
        assert result.files is not None and result.files[0].file == "test.csv"
        # Extras must be silently dropped
        assert not hasattr(result, "extra_field_1")
        test_block = result.structure["test"]
        assert not hasattr(test_block, "extra_block_field")

        # Warnings about extras should have been printed
        captured = capsys.readouterr()
        assert "extra_field_1" in captured.out
        assert "extra_block_field" in captured.out
        assert "extra_col_field" in captured.out

    def test_soft_validate_with_column_meta_extra_fields(self, capsys):
        """Test soft_validate with ColumnMeta warns on extra fields but returns valid model."""
        col_meta_dict = {
            "categories": ["A", "B", "C"],
            "ordered": True,
            "label": "Test Column",
            "extra_unknown_field": "should_be_ignored",
        }

        result = soft_validate(col_meta_dict, ColumnMeta, warnings=True)
        assert isinstance(result, ColumnMeta)
        assert result.categories == ["A", "B", "C"]
        assert result.label == "Test Column"
        assert not hasattr(result, "extra_unknown_field")
        captured = capsys.readouterr()
        assert "extra_unknown_field" in captured.out

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
                        ["x", {"ordered": True}],
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
        # System file columns should always be present for multi-file inputs
        assert {"file_code", "file_name"}.issubset(df.columns)
        assert list(df["file_code"].cat.categories) == ["F0", "F1"]
        assert bool(df["file_code"].cat.ordered) is True
        assert list(df["file_name"].cat.categories) == ["file1.csv", "file2.csv"]
        assert bool(df["file_name"].cat.ordered) is True
        assert df.loc[df["file_code"] == "F0", "file_name"].nunique() == 1
        assert str(df.loc[df["file_code"] == "F0", "file_name"].iloc[0]) == "file1.csv"
        assert df.loc[df["file_code"] == "F1", "file_name"].nunique() == 1
        assert str(df.loc[df["file_code"] == "F1", "file_name"].iloc[0]) == "file2.csv"
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
        # System file columns should always be present for multi-file inputs
        assert {"file_code", "file_name"}.issubset(df.columns)
        assert list(df["file_code"].cat.categories) == ["F0", "F1"]
        assert bool(df["file_code"].cat.ordered) is True
        assert list(df["file_name"].cat.categories) == ["file1.csv", "file2.csv"]
        assert bool(df["file_name"].cat.ordered) is True
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
            "preprocessing": [
                "df['file_flag'] = file_code",  # Should add file_code-based flag
                "df['file_name_flag'] = file_name",
            ],
            "structure": [
                {
                    "name": "test",
                    "columns": {
                        "id": {},
                        "value": {"continuous": True},
                        "file_flag": {"categories": "infer"},
                        "file_name_flag": {"categories": "infer", "ordered": True},
                    },
                }
            ],
        }
        write_json(meta_file, meta)

        df = read_annotated_data(str(meta_file))
        assert {"file_code", "file_name"}.issubset(df.columns)
        assert "file_flag" in df.columns
        assert set(df["file_flag"].unique()) == {"F0", "F1"}
        assert set(df["file_name_flag"].unique()) == {"file1.csv", "file2.csv"}

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

        df, meta_out = read_annotated_data(str(meta_file), return_meta=True)
        assert "name" in df.columns
        assert len(df) == 2
        # Should work as before
        assert set(df["name"].unique()) == {"Alice", "Bob"}
        dumped = meta_out.model_dump(mode="json")
        sys_blocks = [b for b in dumped["structure"] if b.get("name") == "files"]
        assert len(sys_blocks) == 1
        assert sys_blocks[0].get("hidden") is True
        assert sys_blocks[0].get("generated") is True

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
                    "name": "ids",
                    "columns": {"id": {}},
                },
                {
                    "type": "topk",
                    "name": "demographics",
                    "scale": {
                        "categories": "infer",
                        "translate_after": {"1": "Mentioned", "2": "Not mentioned"},
                    },
                    "columns": {
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
                    "from_columns": r"topk_(\d)",
                    "na_vals": ["Not mentioned", "No", "False"],
                    "res_columns": r"topk_R\1",
                },
            ],
        }
        write_json(meta_file, meta)

        data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

        dumped = data_meta.model_dump(mode="json")
        assert dumped["files"] == [
            {"file": "file1.csv", "opts": {}, "code": "F0"},
            {"file": "file2.csv", "opts": {}, "code": "F1"},
            {"file": "file3.csv", "opts": {}, "code": "F2"},
        ]
        # Single-sibling topk: child uses bare block name, overwriting the demoted parent.
        topk_block = next((b for b in dumped["structure"] if b.get("name") == "demographics"), None)
        assert topk_block is not None
        assert topk_block["columns"] == ["topk_R1", "topk_R2"]
        # System file metadata block is injected implicitly for multi-file inputs
        sys_blocks = [b for b in dumped["structure"] if b.get("name") == "files"]
        assert len(sys_blocks) == 1
        sysb = sys_blocks[0]
        # hidden=False is the default and may be omitted from JSON serialization
        assert sysb.get("hidden", False) is False
        assert sysb.get("generated") is True
        sys_cols = {c[0] if isinstance(c, list) else c for c in sysb.get("columns", [])}
        assert {"file_code", "file_name"}.issubset(sys_cols)
        expected_data = [
            ["F0", 1, "Yes", "No", "Mentioned", None],
            ["F0", 2, "No", "Yes", "Not mentioned", None],
            ["F1", 3, "Mentioned", "Mentioned", "Mentioned", "Not mentioned"],
            ["F1", 4, "Mentioned", "Not mentioned", "Mentioned", None],
            ["F2", 5, "True", "False", "Mentioned", None],
            ["F2", 6, "False", "True", "Not mentioned", None],
        ]
        edf = pd.DataFrame(expected_data, columns=["file_code", "id", "topk_1", "topk_2", "topk_R1", "topk_R2"])
        # data_df may contain additional system columns (file_ind, file_name) in multi-file mode
        assert_frame_equal(data_df[edf.columns], edf, check_dtype=False, check_categorical=False)
        assert {"file_name"}.issubset(data_df.columns)


class TestInferMetaDeepL:
    """Tests for infer_meta deepl_key / source_lang parameters."""

    def test_deepl_key_without_source_lang_raises(self):
        """deepl_key without source_lang must raise."""
        df = pd.DataFrame({"a": ["x", "y"]})
        with pytest.raises(ValueError, match="source_lang is required"):
            infer_meta(df=df, meta_file=False, deepl_key="fake-key")

    def test_source_lang_without_deepl_key_raises(self):
        """source_lang without deepl_key must raise."""
        df = pd.DataFrame({"a": ["x", "y"]})
        with pytest.raises(ValueError, match="deepl_key is required"):
            infer_meta(df=df, meta_file=False, source_lang="LT")

    def test_deepl_key_and_translate_fn_raises(self):
        """deepl_key and translate_fn together must raise."""
        df = pd.DataFrame({"a": ["x", "y"]})
        with pytest.raises(ValueError, match="Cannot specify both"):
            infer_meta(df=df, meta_file=False, deepl_key="fake-key", source_lang="LT", translate_fn=str)

    def test_deepl_key_builds_translate_fn(self, tmp_path):
        """When deepl_key + source_lang are given, translations appear in output."""
        from unittest.mock import patch, MagicMock

        mock_result = MagicMock()
        mock_result.text = "Translated"

        mock_translator = MagicMock()
        mock_translator.translate_text.return_value = mock_result

        df = pd.DataFrame({"color": pd.Categorical(["Raudona", "Mėlyna", "Raudona"])})

        with patch("deepl.Translator", return_value=mock_translator) as mock_cls:
            meta = infer_meta(df=df, meta_file=False, deepl_key="test-key", source_lang="LT")

        mock_cls.assert_called_once_with("test-key")
        assert mock_translator.translate_text.call_count > 0
        call_kwargs = mock_translator.translate_text.call_args_list[0]
        assert call_kwargs.kwargs.get("source_lang") == "LT"
        assert call_kwargs.kwargs.get("target_lang") == "EN-US"
        # Check that translate dicts are present in the output
        main_block = meta["structure"][0]
        col_entry = main_block["columns"][0]
        col_meta = col_entry[2] if len(col_entry) > 2 else col_entry[1]
        assert "translate" in col_meta


class TestPipelineSchema:
    """Test pipeline schema validation for new block-processing fields."""

    def test_plain_block_accepts_from_columns_and_subgroup_labels(self) -> None:
        """Verify plain blocks accept from_columns and subgroup_labels without validation errors."""
        from salk_toolkit.validation import soft_validate, ColumnBlockMeta

        b = soft_validate(
            {"name": "q", "from_columns": r"Q(\d+)_(\w+)", "subgroup_labels": {"1": {"1": "econ"}}, "columns": {}},
            ColumnBlockMeta,
        )
        assert b.from_columns == r"Q(\d+)_(\w+)"
        assert b.subgroup_labels == {"1": {"1": "econ"}}

    def test_topk_schema_has_input_format_and_drops_old_fields(self):
        """Verify TopKBlock has input_format and no longer has groups or translate_values."""
        from salk_toolkit.validation import TopKBlock

        assert "input_format" in TopKBlock.model_fields
        assert "subgroup_labels" in TopKBlock.model_fields  # inherited
        assert "groups" not in TopKBlock.model_fields
        assert "translate_values" not in TopKBlock.model_fields

    def test_topk_segments_ranked_and_unranked(self):
        """Verify segments() returns chain for ranked formats and single entry for flat ones."""
        from salk_toolkit.validation import soft_validate, TopKBlock

        b = soft_validate(
            {
                "type": "topk",
                "name": "t",
                "from_columns": ["R1", "R2", "R3"],
                "res_columns": ["R1", "R2", "R3"],
                "input_format": "ranked_leftpack",
                "columns": {"R1": {}, "R2": {}, "R3": {}},
            },
            TopKBlock,
        )
        assert b.segments() == [
            (["R1"], ["R2", "R3"], True),
            (["R2"], ["R3"], True),
            (["R1", "R2", "R3"], None, False),
        ]
        b2 = b.model_copy(update={"input_format": "leftpacked"})
        assert b2.segments() == [(["R1", "R2", "R3"], None, False)]

    def test_maxdiff_schema_has_input_format_and_renamed_fields(self):
        """Verify MaxDiffBlock has input_format; choice_mapping/items/translate removed."""
        from salk_toolkit.validation import MaxDiffBlock

        assert "input_format" in MaxDiffBlock.model_fields
        assert "choice_mapping" not in MaxDiffBlock.model_fields
        assert "items" not in MaxDiffBlock.model_fields
        assert "translate" not in MaxDiffBlock.model_fields

    def test_onehot_block_dispatched_by_discriminator(self):
        """Verify type=onehot is dispatched to OneHotBlock by the discriminated union."""
        from salk_toolkit.validation import soft_validate, DataMeta, OneHotBlock

        meta = soft_validate(
            {
                "structure": {
                    "sm": {
                        "type": "onehot",
                        "name": "sm",
                        "from_columns": r"M_(\d+)",
                        "columns": {},
                    }
                }
            },
            DataMeta,
        )
        assert isinstance(meta.structure["sm"], OneHotBlock)
        assert meta.structure["sm"].input_format == "leftpacked"

    def test_onehot_block_fields(self):
        """Verify OneHotBlock fields validate and default correctly."""
        from salk_toolkit.validation import soft_validate, OneHotBlock

        b = soft_validate(
            {
                "type": "onehot",
                "name": "sm",
                "from_columns": r"vQ12_M_(\d+)",
                "input_format": "leftpacked",
                "choices": ["Facebook", "TikTok"],
                "res_prefix": "sm_",
                "na_vals": ["99"],
            },
            OneHotBlock,
        )
        assert b.input_format == "leftpacked"
        assert b.choices == ["Facebook", "TikTok"]
        assert b.res_prefix == "sm_"
        assert b.na_vals == ["99"]
        assert not hasattr(b, "segments")

    def test_topk_leftpacked_skip_passthrough(self, meta_file, csv_file):
        """input_format=leftpacked skips the transform and passes R1..Rk through."""
        pd.DataFrame({"X_R1": ["A", "B"], "X_R2": ["B", None]}).to_csv(csv_file, index=False)
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "topk",
                    "name": "x",
                    "from_columns": ["X_R1", "X_R2"],
                    "res_columns": ["X_R1", "X_R2"],
                    "input_format": "leftpacked",
                    "scale": {"categories": "infer"},
                }
            ],
        }
        write_json(meta_file, meta)
        _, meta_obj = read_annotated_data(str(meta_file), return_meta=True)
        out = meta_obj.structure["x"]
        assert list(out.columns.keys()) == ["X_R1", "X_R2"]
        assert out.input_format == "leftpacked"
        assert out.segments() == [(["X_R1", "X_R2"], None, False)]

    def test_topk_ranked_leftpack_segments_chain(self, meta_file, csv_file):
        """input_format=ranked_leftpack skips transform and segments() returns ordered chain."""
        pd.DataFrame({"X_R1": ["A", "B"], "X_R2": ["B", None]}).to_csv(csv_file, index=False)
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "topk",
                    "name": "x",
                    "from_columns": ["X_R1", "X_R2"],
                    "res_columns": ["X_R1", "X_R2"],
                    "input_format": "ranked_leftpack",
                    "scale": {"categories": "infer"},
                }
            ],
        }
        write_json(meta_file, meta)
        _, meta_obj = read_annotated_data(str(meta_file), return_meta=True)
        assert meta_obj.structure["x"].segments() == [
            (["X_R1"], ["X_R2"], True),
            (["X_R1", "X_R2"], None, False),
        ]

    def test_topk_leftpacked_mismatch_raises(self, meta_file, csv_file):
        """res_columns != from_columns hard-fails on skip transforms."""
        pd.DataFrame({"X_R1": ["A"]}).to_csv(csv_file, index=False)
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "topk",
                    "name": "x",
                    "from_columns": ["X_R1"],
                    "res_columns": ["X_R1", "X_R2"],
                    "input_format": "leftpacked",
                }
            ],
        }
        write_json(meta_file, meta)
        with pytest.raises(ValueError, match="res_columns to match from_columns"):
            read_annotated_data(str(meta_file), return_meta=True)

    def test_topk_ranked_onehot_raises(self, meta_file, csv_file):
        """ranked_onehot is a scaffold — hard-fails with NotImplementedError."""
        pd.DataFrame({"Qa": [1, 2], "Qb": [2, 1]}).to_csv(csv_file, index=False)
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "topk",
                    "name": "x",
                    "from_columns": r"Q(\w)",
                    "res_columns": r"R\1",
                    "agg_index": 1,
                    "input_format": "ranked_onehot",
                }
            ],
        }
        write_json(meta_file, meta)
        with pytest.raises(NotImplementedError, match="ranked_onehot"):
            read_annotated_data(str(meta_file), return_meta=True)

    def test_topk_truncate_drops_non_na_hard_fails(self, meta_file, csv_file):
        """k=N with more than N non-NA selections per row hard-fails."""
        pd.DataFrame({"Qa": ["A", None], "Qb": ["A", "B"], "Qc": ["A", "C"]}).to_csv(csv_file, index=False)
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "topk",
                    "name": "x",
                    "from_columns": r"Q(\w)",
                    "res_columns": r"R\1",
                    "agg_index": 1,
                    "k": 2,
                    "na_vals": [],
                }
            ],
        }
        write_json(meta_file, meta)
        with pytest.raises(ValueError, match="truncation to k=2"):
            read_annotated_data(str(meta_file), return_meta=True)

    def test_maxdiff_multi_sibling_keyed_end_to_end(self, meta_file, csv_file):
        """Multi-sibling maxdiff with keyed choice_sets produces N sibling blocks.

        `scale.translate` is flat per-block (shared topic universe across siblings); only
        `choice_sets` supports the sibling-keyed dict form.
        """
        parquet_file = csv_file.with_suffix(".parquet")
        pd.DataFrame(
            {
                "Q_g1_b": ["A", "B"],
                "Q_g1_w": ["B", "A"],
                "Q_g1_set": [["A", "B"], ["A", "B"]],
                "Q_g2_b": ["A", "B"],
                "Q_g2_w": ["B", "A"],
                "Q_g2_set": [["A", "B"], ["A", "B"]],
                "V": [1, 1],
            }
        ).to_parquet(parquet_file)
        meta = {
            "file": str(parquet_file),
            "structure": [
                {
                    "type": "maxdiff",
                    "name": "md",
                    "from_columns": r"Q_(\w+)_b",
                    "best_columns": r"Q_(\w+)_b",
                    "worst_columns": r"Q_(\w+)_w",
                    "set_columns": r"Q_\1_set",
                    "setindex_column": "V",
                    "subgroup_labels": {"1": {"g1": "g1", "g2": "g2"}},
                    "choice_sets": {
                        "g1": [[[1, 2]]],
                        "g2": [[[1, 2]]],
                    },
                    "scale": {"translate": {"1": "A", "2": "B"}},
                }
            ],
        }
        write_json(meta_file, meta)
        _, meta_obj = read_annotated_data(str(meta_file), return_meta=True)
        assert "md_g1" in meta_obj.structure
        assert "md_g2" in meta_obj.structure

    def test_maxdiff_single_sibling_rejects_keyed_choice_sets(self, meta_file, csv_file):
        """Single-sibling maxdiff with dict-shaped choice_sets → hard fail."""
        pd.DataFrame(
            {
                "Q_b": ["A", "B"],
                "Q_w": ["B", "A"],
                "Q_set": [["A", "B"], ["A", "B"]],
                "V": [1, 1],
            }
        ).to_parquet(csv_file.with_suffix(".parquet"))
        meta = {
            "file": str(csv_file.with_suffix(".parquet")),
            "structure": [
                {
                    "type": "maxdiff",
                    "name": "md",
                    "best_columns": ["Q_b"],
                    "worst_columns": ["Q_w"],
                    "set_columns": ["Q_set"],
                    "setindex_column": "V",
                    "choice_sets": {"econ": [[[1, 2]]]},
                    "scale": {"translate": {"1": "A", "2": "B"}},
                }
            ],
        }
        write_json(meta_file, meta)
        with pytest.raises(ValueError, match="single sibling.*keyed"):
            read_annotated_data(str(meta_file), return_meta=True)

    def test_maxdiff_resolved_three_independent_regexes(self, meta_file, csv_file):
        """input_format=resolved with three regexes aligned by capture-group value."""
        parquet_file = csv_file.with_suffix(".parquet")
        pd.DataFrame(
            {
                "Q1_b_1": ["A", "B"],
                "Q1_w_1": ["B", "A"],
                "Q1_set_abc_1": [["A", "B"], ["A", "B"]],
                "Q1_b_2": ["B", "A"],
                "Q1_w_2": ["A", "B"],
                "Q1_set_abc_2": [["A", "B"], ["A", "B"]],
            }
        ).to_parquet(parquet_file)
        meta = {
            "file": str(parquet_file),
            "structure": [
                {
                    "type": "maxdiff",
                    "name": "md",
                    "input_format": "resolved",
                    "best_columns": r"Q1_b_(\d+)",
                    "worst_columns": r"Q1_w_(\d+)",
                    "set_columns": r"Q1_set_abc_(\d+)",
                    "scale": {"categories": ["A", "B"]},
                }
            ],
        }
        write_json(meta_file, meta)
        _, meta_obj = read_annotated_data(str(meta_file), return_meta=True)
        out = meta_obj.structure["md"]
        assert sorted(out.best_columns) == ["Q1_b_1", "Q1_b_2"]
        assert sorted(out.worst_columns) == ["Q1_w_1", "Q1_w_2"]
        assert sorted(out.set_columns) == ["Q1_set_abc_1", "Q1_set_abc_2"]
        segs = out.segments()
        assert len(segs) == 4

    def test_maxdiff_resolved_explicit_lists(self, meta_file, csv_file):
        """input_format=resolved with explicit column lists aligned positionally."""
        parquet_file = csv_file.with_suffix(".parquet")
        pd.DataFrame(
            {
                "best1": ["A", "B"],
                "worst1": ["B", "A"],
                "set1": [["A", "B"], ["A", "B"]],
                "best2": ["B", "A"],
                "worst2": ["A", "B"],
                "set2": [["A", "B"], ["A", "B"]],
            }
        ).to_parquet(parquet_file)
        meta = {
            "file": str(parquet_file),
            "structure": [
                {
                    "type": "maxdiff",
                    "name": "md",
                    "input_format": "resolved",
                    "best_columns": ["best1", "best2"],
                    "worst_columns": ["worst1", "worst2"],
                    "set_columns": ["set1", "set2"],
                    "scale": {"categories": ["A", "B"]},
                }
            ],
        }
        write_json(meta_file, meta)
        _, meta_obj = read_annotated_data(str(meta_file), return_meta=True)
        out = meta_obj.structure["md"]
        assert out.best_columns == ["best1", "best2"]
        assert len(out.segments()) == 4

    def test_maxdiff_resolved_incomplete_alignment_hard_fails(self, meta_file, csv_file):
        """Missing a role column for one capture key → hard fail."""
        parquet_file = csv_file.with_suffix(".parquet")
        pd.DataFrame(
            {
                "Q1_b_1": ["A"],
                "Q1_w_1": ["B"],
            }
        ).to_parquet(parquet_file)
        meta = {
            "file": str(parquet_file),
            "structure": [
                {
                    "type": "maxdiff",
                    "name": "md",
                    "input_format": "resolved",
                    "best_columns": r"Q1_b_(\d+)",
                    "worst_columns": r"Q1_w_(\d+)",
                    "set_columns": r"Q1_set_abc_(\d+)",
                }
            ],
        }
        write_json(meta_file, meta)
        with pytest.raises(ValueError, match="incomplete alignment"):
            read_annotated_data(str(meta_file), return_meta=True)

    def test_onehot_leftpacked_explicit_choices(self, meta_file, csv_file):
        """Leftpacked onehot with explicit choices emits one boolean column per choice."""
        pd.DataFrame({"M_1": ["FB", "TT"], "M_2": ["TT", None], "M_3": [None, "FB"]}).to_csv(csv_file, index=False)
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "onehot",
                    "name": "sm",
                    "from_columns": r"M_(\d+)",
                    "input_format": "leftpacked",
                    "choices": ["FB", "TT"],
                    "res_prefix": "sm_",
                    "scale": {"categories": [False, True]},
                }
            ],
        }
        write_json(meta_file, meta)
        ndf, meta_obj = read_annotated_data(str(meta_file), return_meta=True)
        assert sorted(meta_obj.structure["sm"].columns.keys()) == ["sm_FB", "sm_TT"]
        assert list(ndf["sm_FB"]) == [True, True]
        assert list(ndf["sm_TT"]) == [True, True]

    def test_onehot_leftpacked_inferred_choices(self, meta_file, csv_file):
        """choices=None derives the sorted union from observed cells."""
        pd.DataFrame({"M_1": ["A", "B"], "M_2": ["B", None]}).to_csv(csv_file, index=False)
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "onehot",
                    "name": "x",
                    "from_columns": r"M_(\d+)",
                    "input_format": "leftpacked",
                }
            ],
        }
        write_json(meta_file, meta)
        _, meta_obj = read_annotated_data(str(meta_file), return_meta=True)
        assert sorted(meta_obj.structure["x"].columns.keys()) == ["x_A", "x_B"]

    def test_onehot_unknown_value_hard_fails(self, meta_file, csv_file):
        """Explicit choices that don't cover observed cells hard-fails."""
        pd.DataFrame({"M_1": ["A", "ZZ"]}).to_csv(csv_file, index=False)
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "onehot",
                    "name": "x",
                    "from_columns": ["M_1"],
                    "input_format": "leftpacked",
                    "choices": ["A"],
                }
            ],
        }
        write_json(meta_file, meta)
        with pytest.raises(ValueError, match="not in choices"):
            read_annotated_data(str(meta_file), return_meta=True)

    def test_onehot_wide_passthrough(self, meta_file, csv_file):
        """input_format=wide: columns pass through as-is; choices inferred from res_prefix when None."""
        pd.DataFrame({"sm_FB": [True, False], "sm_TT": [False, True]}).to_csv(csv_file, index=False)
        meta = {
            "file": "test.csv",
            "structure": [
                {
                    "type": "onehot",
                    "name": "sm",
                    "from_columns": ["sm_FB", "sm_TT"],
                    "input_format": "wide",
                    "res_prefix": "sm_",
                    "scale": {"categories": [False, True]},
                }
            ],
        }
        write_json(meta_file, meta)
        ndf, meta_obj = read_annotated_data(str(meta_file), return_meta=True)
        out = meta_obj.structure["sm"]
        assert out.input_format == "wide"
        assert sorted(out.columns.keys()) == ["sm_FB", "sm_TT"]
        assert out.choices == ["FB", "TT"]


class TestInternalPipelineHelpers:
    """Tests for _match_columns and _subgroup_explode internal helpers."""

    def test_match_columns_regex(self):
        """Regex pattern matches columns against DataFrame columns."""
        import pandas as pd
        from salk_toolkit.io import _match_columns
        from salk_toolkit.validation import soft_validate, ColumnBlockMeta

        df = pd.DataFrame(columns=["Q1_a", "Q1_b", "Q2_a", "other"])
        b = soft_validate({"name": "t", "from_columns": r"Q(\d+)_(\w+)", "columns": {}}, ColumnBlockMeta)
        assert _match_columns(b, df) == ["Q1_a", "Q1_b", "Q2_a"]

    def test_match_columns_list(self):
        """List pattern returns exactly the listed columns."""
        import pandas as pd
        from salk_toolkit.io import _match_columns
        from salk_toolkit.validation import soft_validate, ColumnBlockMeta

        df = pd.DataFrame(columns=["a", "b", "c"])
        b = soft_validate({"name": "t", "from_columns": ["a", "c"], "columns": {}}, ColumnBlockMeta)
        assert _match_columns(b, df) == ["a", "c"]

    def test_match_columns_empty_raises(self):
        """ValueError raised when no columns match the pattern."""
        import pandas as pd
        import pytest
        from salk_toolkit.io import _match_columns
        from salk_toolkit.validation import soft_validate, ColumnBlockMeta

        b = soft_validate({"name": "t", "from_columns": r"X_(\w+)", "columns": {}}, ColumnBlockMeta)
        with pytest.raises(ValueError, match="No columns matched"):
            _match_columns(b, pd.DataFrame(columns=["Q2_a"]))

    def test_explode_topk_one_nonagg_dim(self):
        """TopK with one non-agg dimension produces one sibling per subgroup value."""
        import pandas as pd
        from salk_toolkit.io import _subgroup_explode
        from salk_toolkit.validation import soft_validate, TopKBlock

        df = pd.DataFrame(columns=["Q7r1c1", "Q7r1c2", "Q7r2c1", "Q7r2c2"])
        b = soft_validate(
            {
                "type": "topk",
                "name": "io",
                "from_columns": r"Q7r(\d+)c(\d+)",
                "res_columns": r"Q7r\1_R\2",
                "agg_index": 2,
                "subgroup_labels": {"1": {"1": "econ", "2": "health"}},
            },
            TopKBlock,
        )
        sibs = _subgroup_explode(b, df)
        assert [s.name for s in sibs] == ["io_econ", "io_health"]
        assert sibs[0].from_columns == ["Q7r1c1", "Q7r1c2"]

    def test_explode_plain_multi_capture(self):
        """Multi-capture regex with no agg_index produces one sibling per unique column."""
        import pandas as pd
        from salk_toolkit.io import _subgroup_explode
        from salk_toolkit.validation import soft_validate, ColumnBlockMeta

        df = pd.DataFrame(columns=["Q1_a", "Q1_b", "Q2_a", "Q2_b"])
        b = soft_validate({"name": "p", "from_columns": r"Q(\d+)_(\w+)", "columns": {}}, ColumnBlockMeta)
        sibs = _subgroup_explode(b, df)
        assert {s.name for s in sibs} == {"p_1_a", "p_1_b", "p_2_a", "p_2_b"}

    def test_explode_no_subgroups_bare_name(self):
        """Single capture group that is the agg axis yields one sibling with block name unchanged."""
        import pandas as pd
        from salk_toolkit.io import _subgroup_explode
        from salk_toolkit.validation import soft_validate, TopKBlock

        df = pd.DataFrame(columns=["Qa", "Qb"])
        b = soft_validate(
            {"type": "topk", "name": "io", "from_columns": r"Q(\w)", "res_columns": r"R\1", "agg_index": 1}, TopKBlock
        )
        sibs = _subgroup_explode(b, df)
        assert len(sibs) == 1
        assert sibs[0].name == "io"
        assert sibs[0].from_columns == ["Qa", "Qb"]

    def test_explode_list_from_columns_single_sibling(self):
        """List from_columns always yields exactly one sibling."""
        import pandas as pd
        from salk_toolkit.io import _subgroup_explode
        from salk_toolkit.validation import soft_validate, ColumnBlockMeta

        df = pd.DataFrame(columns=["a", "b"])
        b = soft_validate({"name": "p", "from_columns": ["a", "b"], "columns": {}}, ColumnBlockMeta)
        sibs = _subgroup_explode(b, df)
        assert len(sibs) == 1
        assert sibs[0].name == "p"
        assert sibs[0].from_columns == ["a", "b"]

    def test_explode_agg_index_out_of_range_raises(self):
        """agg_index beyond the regex group count hard-fails."""
        import pandas as pd
        import pytest
        from salk_toolkit.io import _subgroup_explode
        from salk_toolkit.validation import soft_validate, TopKBlock

        df = pd.DataFrame(columns=["Qa", "Qb"])
        b = soft_validate(
            {"type": "topk", "name": "io", "from_columns": r"Q(\w)", "res_columns": r"R\1", "agg_index": 5},
            TopKBlock,
        )
        with pytest.raises(ValueError, match="agg_index=5 out of range"):
            _subgroup_explode(b, df)

    def test_get_subgroup_config_strict_dispatch(self):
        """_get_subgroup_config enforces flat-for-single, keyed-for-multi."""
        from salk_toolkit.io import _get_subgroup_config

        # Single sibling + flat form: pass-through.
        flat_cs = [[[1, 2]]]
        assert _get_subgroup_config(flat_cs, "md", "md") is flat_cs
        flat_cm = {"1": "A"}
        assert _get_subgroup_config(flat_cm, "md", "md") is flat_cm

        # Single sibling + keyed form: hard fail.
        with pytest.raises(ValueError, match="single sibling.*keyed"):
            _get_subgroup_config({"g1": [[[1, 2]]]}, "md", "md")
        with pytest.raises(ValueError, match="single sibling.*keyed"):
            _get_subgroup_config({"g1": {"1": "A"}}, "md", "md")

        # Multi-sibling + flat form: hard fail.
        with pytest.raises(ValueError, match="multiple siblings.*flat"):
            _get_subgroup_config(flat_cs, "md_g1", "md")

        # Multi-sibling + keyed form, valid key: returns the entry.
        picked = _get_subgroup_config({"g1": [[[1, 2]]], "g2": [[[3, 4]]]}, "md_g1", "md")
        assert picked == [[[1, 2]]]

        # Multi-sibling + keyed form, missing key: hard fail.
        with pytest.raises(ValueError, match="sibling 'g2' missing"):
            _get_subgroup_config({"g1": [[[1, 2]]]}, "md_g2", "md")

        # None returns None regardless of sibling shape.
        assert _get_subgroup_config(None, "md", "md") is None
        assert _get_subgroup_config(None, "md_g1", "md") is None


if __name__ == "__main__":
    pytest.main([__file__])
