"""Tests for serialization functionality.

Tests cover:
- ColumnMeta serialization with and without block_scale context
- ColumnBlockMeta serialization with block_scale context
- Round-trip serialization (JSON -> DataMeta -> JSON)
"""

from typing import Any

from salk_toolkit.validation import (
    soft_validate,
    DataMeta,
    ColumnMeta,
    ColumnBlockMeta,
    BlockScaleMeta,
)

# Default fields that should be excluded from serialization
DEFAULT_COLUMN_META_FIELDS = [
    "datetime",
    "source",
    "transform",
    "translate",
    "translate_after",
    "modifiers",
    "nonordered",
    "labels",
    "groups",
    "colors",
    "num_values",
    "val_format",
    "val_range",
]

# Default fields for DataMeta that should be excluded
DEFAULT_DATA_META_FIELDS = [
    "description",
    "source",
    "restrictions",
    "collection_start",
    "collection_end",
    "author",
    "file",
    "read_opts",
    "files",
    "constants",
    "preprocessing",
    "postprocessing",
    "weight_col",
    "excluded",
    "total_size",
    "draws_data",
]

# Default fields for ColumnBlockMeta that should be excluded
DEFAULT_COLUMN_BLOCK_META_FIELDS = [
    "scale",
    "subgroup_transform",
    "generated",
    "hidden",
    "create",
]


class TestColumnMetaSerialization:
    """Test ColumnMeta serialization behavior."""

    def test_column_meta_serialization_with_block_scale(self):
        """Test that ColumnMeta excludes fields matching block_scale, not defaults, when block_scale is in context."""
        # Create a block scale with some values
        block_scale = BlockScaleMeta(
            categories=["A", "B", "C"],
            ordered=True,
            continuous=False,  # This is the default, but should still be compared
        )

        # Create a column meta that matches block_scale in some fields but not others
        col_meta = ColumnMeta(
            categories=["A", "B", "C"],  # Matches block_scale
            ordered=True,  # Matches block_scale
            continuous=False,  # Matches block_scale (default)
            label="Custom label",  # Different from block_scale (None)
            likert=False,  # Matches block_scale (default)
        )

        # Serialize with block_scale in context
        context = {"block_scale": block_scale}
        serialized = col_meta.model_dump(mode="json", context=context)

        # Fields matching block_scale should be excluded
        assert "categories" not in serialized  # Matches block_scale
        assert "ordered" not in serialized  # Matches block_scale
        assert "continuous" not in serialized  # Matches block_scale
        assert "likert" not in serialized  # Matches block_scale (default)

        # Fields different from block_scale should be included
        assert "label" in serialized
        assert serialized["label"] == "Custom label"

        # Default fields should also be excluded
        for field in DEFAULT_COLUMN_META_FIELDS:
            assert field not in serialized

    def test_column_meta_serialization_without_block_scale(self):
        """Test that ColumnMeta uses default behavior (exclude defaults) when no block_scale in context."""
        # Create a column meta with some defaults and some non-defaults
        col_meta = ColumnMeta(
            categories=["A", "B", "C"],  # Non-default
            ordered=True,  # Non-default
            continuous=False,  # Default
            datetime=False,  # Default
            likert=False,  # Default
        )

        # Serialize without block_scale in context
        serialized = col_meta.model_dump(mode="json")

        # Non-default fields should be included
        assert "categories" in serialized
        assert serialized["categories"] == ["A", "B", "C"]
        assert "ordered" in serialized
        assert serialized["ordered"] is True

        # Default fields should be excluded
        default_fields = ["continuous", "datetime", "likert", "label"] + DEFAULT_COLUMN_META_FIELDS
        for field in default_fields:
            assert field not in serialized


class TestColumnBlockMetaSerialization:
    """Test ColumnBlockMeta serialization behavior."""

    def test_column_block_meta_passes_block_scale_to_context(self):
        """Test that ColumnBlockMeta puts block_scale in context for ColumnMeta serialization."""
        # Create a block scale
        block_scale = BlockScaleMeta(
            categories=["Low", "Medium", "High"],
            ordered=True,
        )

        # Create column metas - one matching block_scale, one with differences
        col_meta1 = ColumnMeta(
            categories=["Low", "Medium", "High"],  # Matches block_scale
            ordered=True,  # Matches block_scale
            label="Question 1",  # Different
        )

        col_meta2 = ColumnMeta(
            categories=["Low", "Medium", "High"],  # Matches block_scale
            ordered=True,  # Matches block_scale
            label="Question 2",  # Different
            likert=True,  # Different from block_scale (default False)
        )

        # Create a ColumnBlockMeta with the scale
        block = ColumnBlockMeta(
            name="test_block",
            scale=block_scale,
            columns={
                "q1": col_meta1,
                "q2": col_meta2,
            },
        )

        # Serialize the block
        serialized = block.model_dump(mode="json")

        # Check that columns are serialized with block_scale context and in list format
        assert "columns" in serialized
        assert isinstance(serialized["columns"], list)

        # Find q1 and q2 in the list format
        q1_spec = next(col for col in serialized["columns"] if isinstance(col, list) and col[0] == "q1")
        q2_spec = next(col for col in serialized["columns"] if isinstance(col, list) and col[0] == "q2")

        # Extract metadata (last element of the list)
        q1_meta = q1_spec[-1] if len(q1_spec) > 1 else {}
        q2_meta = q2_spec[-1] if len(q2_spec) > 1 else {}

        # Fields matching block_scale should be excluded
        assert "categories" not in q1_meta  # Matches block_scale
        assert "ordered" not in q1_meta  # Matches block_scale

        # Fields different from block_scale should be included
        assert "label" in q1_meta
        assert q1_meta["label"] == "Question 1"

        # For q2, likert should be included (different from block_scale default)
        assert "categories" not in q2_meta  # Matches block_scale
        assert "ordered" not in q2_meta  # Matches block_scale
        assert "label" in q2_meta
        assert "likert" in q2_meta  # Different from block_scale
        assert q2_meta["likert"] is True

        # Default fields should be excluded from both q1 and q2
        default_fields = ["continuous", "datetime"] + DEFAULT_COLUMN_META_FIELDS
        for q_meta in [q1_meta, q2_meta]:
            for field in default_fields:
                assert field not in q_meta


class TestDataMetaSerialization:
    """Test DataMeta serialization behavior."""

    def test_data_meta_excludes_defaults(self):
        """Test that DataMeta excludes default fields from serialization."""
        # Create a DataMeta with only required fields and some non-defaults
        original_meta = {
            "file": "test.csv",  # Non-default (has a value)
            "structure": [
                {
                    "name": "test_block",
                    "columns": ["col1", "col2"],
                }
            ],
        }

        # Convert to DataMeta
        data_meta = soft_validate(original_meta, DataMeta)

        # Serialize
        serialized = data_meta.model_dump(mode="json")

        # Required/non-default fields should be included
        assert "file" in serialized
        assert serialized["file"] == "test.csv"
        assert "structure" in serialized

        # Default fields should be excluded
        for field in DEFAULT_DATA_META_FIELDS:
            if field != "file":  # file is non-default in this test
                assert field not in serialized

    def test_data_meta_includes_non_defaults(self):
        """Test that DataMeta includes non-default fields."""
        original_meta = {
            "file": "test.csv",
            "description": "Test dataset",  # Non-default
            "preprocessing": "df = df.dropna()",  # Non-default
            "structure": [
                {
                    "name": "test_block",
                    "columns": ["col1"],
                }
            ],
        }

        data_meta = soft_validate(original_meta, DataMeta)
        serialized = data_meta.model_dump(mode="json")

        # Non-default fields should be included
        assert "description" in serialized
        assert serialized["description"] == "Test dataset"
        assert "preprocessing" in serialized
        assert serialized["preprocessing"] == "df = df.dropna()"

        # Other defaults should still be excluded
        default_fields_to_check = [
            f for f in DEFAULT_DATA_META_FIELDS if f not in ["description", "preprocessing", "file"]
        ]
        for field in default_fields_to_check:
            assert field not in serialized


class TestColumnBlockMetaSerializationDefaults:
    """Test ColumnBlockMeta serialization excludes default fields."""

    def test_column_block_meta_excludes_defaults(self):
        """Test that ColumnBlockMeta excludes default fields from serialization."""
        # Create a block with only required fields
        block = ColumnBlockMeta(
            name="test_block",
            columns={
                "col1": ColumnMeta(categories=["A", "B"]),
                "col2": ColumnMeta(continuous=True),
            },
        )

        # Serialize
        serialized = block.model_dump(mode="json")

        # Required fields should be included
        assert "name" in serialized
        assert serialized["name"] == "test_block"
        assert "columns" in serialized

        # Default fields should be excluded
        for field in DEFAULT_COLUMN_BLOCK_META_FIELDS:
            assert field not in serialized

    def test_column_block_meta_includes_non_defaults(self):
        """Test that ColumnBlockMeta includes non-default fields."""
        block = ColumnBlockMeta(
            name="test_block",
            columns={
                "col1": ColumnMeta(categories=["A", "B"]),
            },
            generated=True,  # Non-default
            hidden=True,  # Non-default
        )

        # Serialize
        serialized = block.model_dump(mode="json")

        # Non-default fields should be included
        assert "generated" in serialized
        assert serialized["generated"] is True
        assert "hidden" in serialized
        assert serialized["hidden"] is True

        # Other defaults should still be excluded
        default_fields_to_check = [f for f in DEFAULT_COLUMN_BLOCK_META_FIELDS if f not in ["generated", "hidden"]]
        for field in default_fields_to_check:
            assert field not in serialized

    def test_column_block_meta_with_scale_excludes_defaults(self):
        """Test that ColumnBlockMeta with scale excludes defaults correctly."""
        block_scale = BlockScaleMeta(
            categories=["Low", "Medium", "High"],
            ordered=True,
        )

        block = ColumnBlockMeta(
            name="test_block",
            scale=block_scale,  # Non-default (has a value)
            columns={
                "col1": ColumnMeta(
                    categories=["Low", "Medium", "High"],
                    ordered=True,
                    label="Question 1",
                ),
            },
        )

        # Serialize
        serialized = block.model_dump(mode="json")

        # Scale should be included (non-default)
        assert "scale" in serialized

        # Other defaults should be excluded
        default_fields_to_check = [f for f in DEFAULT_COLUMN_BLOCK_META_FIELDS if f != "scale"]
        for field in default_fields_to_check:
            assert field not in serialized


def normalize_for_comparison(data: Any) -> Any:
    """Normalize data structures for comparison, handling format differences."""
    if isinstance(data, dict):
        # Remove None values and empty dicts/lists for comparison
        result = {}
        for key, value in data.items():
            normalized = normalize_for_comparison(value)
            if normalized is not None and normalized != {} and normalized != []:
                result[key] = normalized
        return result
    elif isinstance(data, list):
        return [normalize_for_comparison(item) for item in data if normalize_for_comparison(item) is not None]
    else:
        return data


class TestRoundTripSerialization:
    """Test round-trip serialization: JSON -> DataMeta -> JSON."""

    def test_round_trip_simple_meta(self):
        """Test round-trip with simple metadata."""
        original_meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "basic",
                    "columns": [
                        "id",
                        ["name", {"categories": "infer"}],
                        ["age", {"continuous": True}],
                    ],
                }
            ],
        }

        # Convert to DataMeta
        data_meta = soft_validate(original_meta, DataMeta)

        # Serialize back to JSON
        serialized = data_meta.model_dump(mode="json")

        # Check that DataMeta default fields are excluded
        default_data_meta_fields = [
            f for f in DEFAULT_DATA_META_FIELDS if f != "file"
        ]  # file is non-default in this test
        for field in default_data_meta_fields:
            assert field not in serialized

        # Normalize both for comparison (remove defaults, None values, etc.)
        serialized_normalized = normalize_for_comparison(serialized)

        # Structure should be preserved (as list)
        assert "structure" in serialized_normalized
        assert isinstance(serialized_normalized["structure"], list)
        assert len(serialized_normalized["structure"]) == 1

        # Block name should match
        block = serialized_normalized["structure"][0]
        assert block["name"] == "basic"

        # Check that ColumnBlockMeta default fields are excluded
        for field in DEFAULT_COLUMN_BLOCK_META_FIELDS:
            assert field not in block

        # Columns should be in list format
        assert "columns" in block
        assert isinstance(block["columns"], list)

        # Check that columns are present (may be in different formats)
        column_names = []
        for col in block["columns"]:
            if isinstance(col, str):
                column_names.append(col)
            elif isinstance(col, list) and len(col) > 0:
                column_names.append(col[0])

        assert "id" in column_names
        assert "name" in column_names
        assert "age" in column_names

        # Check that default fields are excluded
        for col in block["columns"]:
            if isinstance(col, list) and len(col) > 1:
                meta = col[-1]
                # Fields that should be completely absent (empty defaults)
                empty_default_fields = [
                    "translate",
                    "translate_after",
                    "modifiers",
                    "nonordered",
                    "labels",
                    "groups",
                    "colors",
                ]
                for field in empty_default_fields:
                    assert field not in meta
                # Fields that might be present only if non-default
                if "continuous" in meta:
                    assert meta["continuous"] is True  # If present, must be non-default
                if "datetime" in meta:
                    assert meta["datetime"] is True  # Only present if non-default
                if "source" in meta:
                    assert meta["source"] is not None
                if "transform" in meta:
                    assert meta["transform"] is not None
                if "ordered" in meta:
                    assert meta["ordered"] is True  # Only present if non-default

    def test_round_trip_with_block_scale(self):
        """Test round-trip with block scale that should be preserved."""
        original_meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "likert_scale",
                    "scale": {
                        "categories": ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
                        "ordered": True,
                    },
                    "columns": [
                        # q1 matches scale exactly, so categories and ordered will be excluded
                        [
                            "q1",
                            {
                                "label": "Question 1",
                                "categories": ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
                                "ordered": True,
                            },
                        ],
                        # q2 has likert=True which differs from scale default, but categories/ordered match scale
                        [
                            "q2",
                            {
                                "label": "Question 2",
                                "categories": ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
                                "ordered": True,
                                "likert": True,
                            },
                        ],
                    ],
                }
            ],
        }

        # Convert to DataMeta
        data_meta = soft_validate(original_meta, DataMeta)

        # Serialize back to JSON
        serialized = data_meta.model_dump(mode="json")

        # Check structure
        assert "structure" in serialized
        assert isinstance(serialized["structure"], list)
        block = serialized["structure"][0]
        assert block["name"] == "likert_scale"

        # Check that ColumnBlockMeta default fields are excluded (except scale which is non-default)
        default_block_fields = [f for f in DEFAULT_COLUMN_BLOCK_META_FIELDS if f != "scale"]
        for field in default_block_fields:
            assert field not in block

        # Scale should be preserved
        assert "scale" in block
        scale = block["scale"]
        assert scale["categories"] == ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
        assert scale["ordered"] is True

        # Columns should be in list format
        assert "columns" in block
        assert isinstance(block["columns"], list)

        # Find q1 and q2
        q1_spec = next(col for col in block["columns"] if isinstance(col, list) and col[0] == "q1")
        q2_spec = next(col for col in block["columns"] if isinstance(col, list) and col[0] == "q2")

        # q1 should only have label (categories and ordered match block_scale)
        q1_meta = q1_spec[-1] if len(q1_spec) > 1 else {}
        assert "label" in q1_meta
        assert q1_meta["label"] == "Question 1"
        assert "categories" not in q1_meta  # Matches block_scale
        assert "ordered" not in q1_meta  # Matches block_scale

        # q2 should have label and likert (likert differs from block_scale default)
        q2_meta = q2_spec[-1] if len(q2_spec) > 1 else {}
        assert "label" in q2_meta
        assert q2_meta["label"] == "Question 2"
        assert "likert" in q2_meta
        assert q2_meta["likert"] is True
        assert "categories" not in q2_meta  # Matches block_scale
        assert "ordered" not in q2_meta  # Matches block_scale

        # Default fields should be excluded from both q1 and q2
        default_fields = ["continuous", "datetime"] + DEFAULT_COLUMN_META_FIELDS
        for q_meta in [q1_meta, q2_meta]:
            for field in default_fields:
                assert field not in q_meta

    def test_round_trip_with_categories_and_colors(self):
        """Test round-trip with categories and colors."""
        original_meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "demographics",
                    "columns": [
                        [
                            "gender",
                            {
                                "categories": ["Male", "Female"],
                                "colors": {"Male": "blue", "Female": "red"},
                            },
                        ],
                        [
                            "age_group",
                            {
                                "categories": ["18-24", "25-34", "35-44"],
                                "ordered": True,
                            },
                        ],
                    ],
                }
            ],
        }

        # Convert to DataMeta
        data_meta = soft_validate(original_meta, DataMeta)

        # Serialize back to JSON
        serialized = data_meta.model_dump(mode="json")

        # Check structure
        block = serialized["structure"][0]
        assert block["name"] == "demographics"

        # Check that ColumnBlockMeta default fields are excluded
        for field in DEFAULT_COLUMN_BLOCK_META_FIELDS:
            assert field not in block

        # Find gender and age_group columns
        gender_spec = next(col for col in block["columns"] if isinstance(col, list) and col[0] == "gender")
        age_group_spec = next(col for col in block["columns"] if isinstance(col, list) and col[0] == "age_group")

        gender_meta = gender_spec[-1] if len(gender_spec) > 1 else {}
        age_group_meta = age_group_spec[-1] if len(age_group_spec) > 1 else {}

        # Gender should have categories and colors
        assert "categories" in gender_meta
        assert gender_meta["categories"] == ["Male", "Female"]
        assert "colors" in gender_meta
        assert gender_meta["colors"] == {"Male": "blue", "Female": "red"}

        # Age group should have categories and ordered
        assert "categories" in age_group_meta
        assert age_group_meta["categories"] == ["18-24", "25-34", "35-44"]
        assert age_group_meta["ordered"] is True

        # Default fields should be excluded from both columns
        # Note: colors is excluded from check for gender_meta since it has non-default values
        default_fields = ["continuous", "datetime", "likert", "label"] + [
            f for f in DEFAULT_COLUMN_META_FIELDS if f != "colors"
        ]
        for meta in [gender_meta, age_group_meta]:
            for field in default_fields:
                assert field not in meta
        # For gender_meta, colors should be present (non-default), but other defaults should be excluded
        assert "colors" in gender_meta
        # For age_group_meta, colors should be excluded (default empty dict)
        assert "colors" not in age_group_meta

    def test_round_trip_preserves_structure_format(self):
        """Test that structure is always serialized as a list, not a dict."""
        original_meta = {
            "file": "test.csv",
            "structure": [
                {"name": "block1", "columns": ["col1", "col2"]},
                {"name": "block2", "columns": [["col3", {"continuous": True}]]},
            ],
        }

        # Convert to DataMeta (internally uses dict format)
        data_meta = soft_validate(original_meta, DataMeta)

        # Verify internal format is dict
        assert isinstance(data_meta.structure, dict)
        assert "block1" in data_meta.structure
        assert "block2" in data_meta.structure

        # Serialize back to JSON (should be list format)
        serialized = data_meta.model_dump(mode="json")

        # Should be list format
        assert isinstance(serialized["structure"], list)
        assert len(serialized["structure"]) == 2

        # Block names should be preserved
        block_names = [block["name"] for block in serialized["structure"]]
        assert "block1" in block_names
        assert "block2" in block_names

    def test_round_trip_excludes_defaults(self):
        """Test that defaults are excluded in serialized output."""
        original_meta = {
            "file": "test.csv",
            "structure": [
                {
                    "name": "test",
                    "columns": [
                        ["col1", {"categories": ["A", "B"], "ordered": False}],  # ordered=False is default
                        ["col2", {"continuous": True}],  # continuous=True is non-default
                    ],
                }
            ],
        }

        # Convert to DataMeta
        data_meta = soft_validate(original_meta, DataMeta)

        # Serialize back
        serialized = data_meta.model_dump(mode="json")

        block = serialized["structure"][0]
        col1_spec = next(col for col in block["columns"] if isinstance(col, list) and col[0] == "col1")
        col2_spec = next(col for col in block["columns"] if isinstance(col, list) and col[0] == "col2")

        col1_meta = col1_spec[-1] if len(col1_spec) > 1 else {}
        col2_meta = col2_spec[-1] if len(col2_spec) > 1 else {}

        # col1: ordered=False is default, should be excluded
        assert "categories" in col1_meta
        assert "ordered" not in col1_meta  # ordered=False is default

        # col2: continuous=True is non-default, should be included
        assert "continuous" in col2_meta
        assert col2_meta["continuous"] is True

        # Default fields should be excluded from both columns
        default_fields = ["datetime", "likert", "label"] + DEFAULT_COLUMN_META_FIELDS
        for meta in [col1_meta, col2_meta]:
            for field in default_fields:
                assert field not in meta
