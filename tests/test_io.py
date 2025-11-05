"""
Comprehensive tests for read_annotated_data and read_and_process_data functions
covering all features of meta parsing.
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from pandas.testing import assert_frame_equal

from salk_toolkit.io import (
    read_annotated_data, 
    read_and_process_data,
    write_parquet_with_metadata,
    reset_file_tracking,
    get_loaded_files,
    extract_column_meta,
    group_columns_dict,
    replace_data_meta_in_parquet
)

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
    with open(file_path, 'w') as f:
        json.dump(data, f)

# Extend DataFrame with convenient CSV writing
def df_to_csv(self, file_path):
    """Write DataFrame to CSV file (no index)"""
    self.to_csv(file_path, index=False)
    return file_path

pd.DataFrame.to_csv_file = df_to_csv

@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin'],
        'score': ['High', 'Medium', 'Low', 'High', 'Medium'],
        'date_str': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'is_active': ['Yes', 'No', 'Yes', 'Yes', 'No']
    })

class TestReadAnnotatedData:
    """Test read_annotated_data function"""
    
    def test_json_file_loading_basic(self, csv_file, meta_file, sample_csv_data):
        """Test basic JSON metafile loading"""
        sample_csv_data.to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "basic",
                "columns": [
                    "id",
                    ["name", {"categories": "infer"}],
                    ["age", {"continuous": True}],
                    ["city", {"categories": "infer"}]
                ]
            }]
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file), return_raw=False)
        
        assert len(df) == 5
        assert 'id' in df.columns
        assert 'name' in df.columns
        assert 'age' in df.columns
        assert 'city' in df.columns
        
        # Check data types
        assert df['age'].dtype in [np.int64, np.float64]  # continuous
        assert df['name'].dtype.name == 'category'  # inferred categories
        assert df['city'].dtype.name == 'category'  # inferred categories
    
    def test_return_raw_parameter(self, csv_file, meta_file, sample_csv_data):
        """Test return_raw parameter"""
        sample_csv_data.to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "basic", 
                "columns": [["score", {"translate": {"High": "Good", "Medium": "OK", "Low": "Bad"}}]]
            }]
        }
        write_json(meta_file, meta)
        
        # Test return_raw=True (should return raw data before processing)
        raw_df = read_annotated_data(str(meta_file), return_raw=True)
        assert 'High' in raw_df['score'].values  # Original values
        
        # Test return_raw=False (should return processed data)
        processed_df = read_annotated_data(str(meta_file), return_raw=False)
        assert 'Good' in processed_df['score'].values  # Translated values
    
    def test_parquet_file_loading(self, temp_dir, sample_csv_data):
        """Test loading parquet files with embedded metadata"""
        parquet_file = temp_dir / "test.parquet"
        meta = {
            "data": {
                "structure": [{
                    "name": "test",
                    "columns": ["id", "name", "age"]
                }]
            },
            "model": {"test_model": "info"}
        }
        
        write_parquet_with_metadata(sample_csv_data, meta, str(parquet_file))
        
        df, data_meta = read_annotated_data(str(parquet_file), return_meta=True)
        
        assert len(df) == 5
        assert data_meta is not None
        assert data_meta == meta['data']

    def test_meta_inference(self, csv_file, sample_csv_data):
        """Test automatic meta inference when no meta exists"""
        sample_csv_data.to_csv_file(csv_file)
        
        df, meta = read_annotated_data(str(csv_file), return_meta=True, infer=True)
        
        assert len(df) == 5
        assert meta is not None
        assert 'structure' in meta
        assert len(meta['structure']) > 0

    def test_topk_create_block(self, meta_file, csv_file):
        """Test top k create block."""
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "topk",
                "columns": ["id","q1_1","q1_2","q1_3","q2_1","q2_2","q2_3"],
                "create": {
                    "type": "topk",
                    "from_columns": r"q(\d+)_(\d+)",
                    "name": "issue_importance_raw",
                    "na_vals": ["not_selected"],
                    "res_cols": r"q\1_R\2",
                    "scale": {
                        "categories": "infer",
                        "translate": {
                            "1": "USA",
                            "2": "Canada",
                            "3": "Mexico"
                        }
                    }
                }
            }]
        }
        write_json(meta_file, meta)

        df = pd.DataFrame({
            'q1_1': ['selected', 'not_selected', 'not_selected'],
            'q1_2': ['not_selected', 'selected', 'not_selected'],
            'q1_3': ['not_selected', 'not_selected', 'selected'],
            'q2_1': ['selected', 'not_selected', 'selected'],
            'q2_2': ['selected', 'selected', 'selected'],
            'q2_3': ['selected', 'not_selected', 'not_selected'],
            'id': ['a', 'b', 'c']
        })
        df.to_csv_file(csv_file)
        data_df, data_meta = read_and_process_data(str(meta_file),return_meta=True)
        newcols = data_df.columns.difference(df.columns)
        diffs = data_df[newcols].replace('<NA>',pd.NA)
        expected_result = pd.DataFrame([
            ["USA","USA","Canada","Mexico"],
            ["Canada","Canada",pd.NA, pd.NA],
            ["Mexico","USA","Canada",pd.NA]
            ], columns=newcols, 
            dtype=pd.CategoricalDtype(categories=['USA', 'Canada','Mexico']))
        expected_meta = {
            'file': 'test.csv',
            'structure': [
                {'name': 'topk', 'columns': ['id', 'q1_1', 'q1_2', 'q1_3', 'q2_1', 'q2_2', 'q2_3']},
                {'name': 'issue_importance_raw_1', 'scale': {'categories': ['USA', 'Canada', 'Mexico'], 'translate': {'1': 'USA', '2': 'Canada', '3': 'Mexico'}}, 'columns': ['q1_R1']},
                {'name': 'issue_importance_raw_2', 'scale': {'categories': ['USA', 'Canada', 'Mexico'], 'translate': {'1': 'USA', '2': 'Canada', '3': 'Mexico'}}, 'columns': ['q2_R1', 'q2_R2', 'q2_R3']}
            ]
        }
        assert_frame_equal(diffs.fillna(pd.NA), expected_result.fillna(pd.NA), check_dtype=False, check_categorical=False)
        assert data_meta == expected_meta


class TestColumnTransformations:
    """Test various column transformation features"""
    
    def test_translate_transformation(self, csv_file, meta_file):
        """Test translate transformation"""
        pd.DataFrame({
            'status': ['A', 'B', 'C', 'A', 'B'],
            'id': [1, 2, 3, 4, 5]
        }).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": [
                    "id",
                    ["status", {
                        "translate": {"A": "Active", "B": "Blocked", "C": "Cancelled"},
                        "categories": ["Active", "Blocked", "Cancelled"]
                    }]
                ]
            }]
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file))
        
        assert 'Active' in df['status'].values
        assert 'Blocked' in df['status'].values
        assert 'Cancelled' in df['status'].values
        assert 'A' not in df['status'].values
    
    def test_transform_code_execution(self, csv_file, meta_file):
        """Test transform code execution"""
        pd.DataFrame({
            'value': [10, 20, 30, 40, 50],
            'id': [1, 2, 3, 4, 5]
        }).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": [
                    "id",
                    ["doubled_value", "value", {"transform": "s * 2", "continuous": True}]
                ]
            }]
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file))
        assert df['doubled_value'].tolist() == [20, 40, 60, 80, 100]
    
    def test_translate_after_transformation(self, csv_file, meta_file):
        """Test translate_after transformation"""
        pd.DataFrame({
            'value': [1, 2, 3, 4, 5],
            'id': [1, 2, 3, 4, 5]
        }).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": [
                    "id",
                    ["category", "value", {
                        "transform": "s * 10",  # First multiply by 10
                        "translate_after": {"10": "Low", "20": "Medium", "30": "High", "40": "Very High", "50": "Max"},
                        "categories": ["Low", "Medium", "High", "Very High", "Max"]
                    }]
                ]
            }]
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file))
        
        assert 'Low' in df['category'].values
        assert 'Medium' in df['category'].values
        assert 'High' in df['category'].values
    
    def test_datetime_transformation(self, csv_file, meta_file):
        """Test datetime transformation"""
        pd.DataFrame({
            'date_str': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'id': [1, 2, 3]
        }).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": [
                    "id",
                    ["date", "date_str", {"datetime": True}]
                ]
            }]
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file))
        assert pd.api.types.is_datetime64_any_dtype(df['date'])
    
    def test_continuous_transformation(self, csv_file, meta_file):
        """Test continuous transformation"""
        pd.DataFrame({
            'value_str': ['10.5', '20.3', '30.7'],
            'id': [1, 2, 3]
        }).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": [
                    "id",
                    ["value", "value_str", {"continuous": True}]
                ]
            }]
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file))
        
        assert pd.api.types.is_numeric_dtype(df['value'])
        assert df['value'].tolist() == [10.5, 20.3, 30.7]

class TestCategoricalFeatures:
    """Test categorical data features"""
    
    def test_category_inference(self, csv_file, meta_file):
        """Test category inference"""
        pd.DataFrame({
            'status': ['Active', 'Inactive', 'Pending', 'Active', 'Inactive'],
            'id': [1, 2, 3, 4, 5]
        }).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": [
                    "id",
                    ["status", {"categories": "infer"}]
                ]
            }]
        }
        write_json(meta_file, meta)
        
        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)
        
        assert df['status'].dtype.name == 'category'
        # Check that categories were inferred and stored in meta
        status_meta = None
        for group in result_meta['structure']:
            for col in group['columns']:
                if isinstance(col, list) and col[0] == 'status':
                    status_meta = col[-1]
                    break
        
        assert status_meta is not None
        assert 'categories' in status_meta
        assert set(status_meta['categories']) == {'Active', 'Inactive', 'Pending'}
    
    def test_ordered_categories(self, csv_file, meta_file):
        """Test ordered categories"""
        pd.DataFrame({
            'rating': ['Poor', 'Good', 'Excellent', 'Good', 'Poor'],
            'id': [1, 2, 3, 4, 5]
        }).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": [
                    "id",
                    ["rating", {
                        "categories": ["Poor", "Good", "Excellent"],
                        "ordered": True
                    }]
                ]
            }]
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file))
        
        assert df['rating'].dtype.name == 'category'
        assert df['rating'].dtype.ordered == True
        assert list(df['rating'].dtype.categories) == ["Poor", "Good", "Excellent"]
    
    def test_numeric_categories_mapping(self, csv_file, meta_file):
        """Test numeric categories mapping to nearest values"""
        pd.DataFrame({
            'score': [1.1, 2.9, 4.8, 1.2, 3.1],
            'id': [1, 2, 3, 4, 5]
        }).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": [
                    "id",
                    ["score", {"categories": ["1", "3", "5"], "ordered": True}]
                ]
            }]
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file))
        
        # Should map to nearest categories
        expected_mapping = ['1', '3', '5', '1', '3']  # 1.1->1, 2.9->3, 4.8->5, 1.2->1, 3.1->3
        assert df['score'].tolist() == expected_mapping

class TestAdvancedFeatures:
    """Test advanced meta parsing features"""
    
    def test_constants_replacement(self, csv_file, meta_file):
        """Test constants replacement"""
        pd.DataFrame({
            'code': ['A', 'B', 'C'],
            'id': [1, 2, 3]
        }).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "constants": {
                "code_mapping": {"A": "Alpha", "B": "Beta", "C": "Gamma"}
            },
            "structure": [{
                "name": "test",
                "columns": [
                    "id",
                    ["code", {"translate": "code_mapping", "categories": ["Alpha", "Beta", "Gamma"]}]
                ]
            }]
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file))
        
        assert 'Alpha' in df['code'].values
        assert 'Beta' in df['code'].values
        assert 'Gamma' in df['code'].values
    
    def test_list_preprocessing(self, csv_file, meta_file):
        """Test preprocessing as list of strings"""
        pd.DataFrame({'value': [1, 2, 3], 'text': ['a', 'b', 'c']}).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "preprocessing": [
                "df['doubled'] = df['value'] * 2",
                "df['upper'] = df['text'].str.upper()",
                "df['combined'] = df['doubled'].astype(str) + df['upper']"
            ],
            "structure": [{"name": "test", "columns": ["value", "doubled", "upper", "combined"]}]
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file))
        assert df['doubled'].tolist() == [2, 4, 6]
        assert df['upper'].tolist() == ['A', 'B', 'C']
        assert df['combined'].tolist() == ['2A', '4B', '6C']
    
    def test_scale_num_values(self, csv_file, meta_file):
        """Test num_values metadata preservation at scale level"""
        pd.DataFrame({'rating': ['Poor', 'Good', 'Excellent']}).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "scale": {
                    "categories": ["Poor", "Good", "Excellent"],
                    "ordered": True,
                    "num_values": [-1, 0, 1]
                },
                "columns": [["rating_num", "rating"]]
            }]
        }
        write_json(meta_file, meta)
        
        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)
        # Data should still be categorical
        assert df['rating_num'].tolist() == ['Poor', 'Good', 'Excellent']
        assert df['rating_num'].dtype.name == 'category'
        assert df['rating_num'].dtype.ordered == True
        # But num_values should be preserved in metadata for later use
        test_group = next(group for group in result_meta['structure'] if group['name'] == 'test')
        assert test_group['scale']['num_values'] == [-1, 0, 1]
    
    def test_colors_parameter(self, csv_file, meta_file):
        """Test colors parameter referencing constants"""
        pd.DataFrame({'party': ['A', 'B', 'C']}).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "constants": {
                "test_colors": {"A": "red", "B": "blue", "C": "green"}
            },
            "structure": [{
                "name": "test",
                "columns": [["party", {"categories": ["A", "B", "C"], "colors": "test_colors"}]]
            }]
        }
        write_json(meta_file, meta)
        
        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)
        # Verify colors are preserved in metadata using extract_column_meta
        column_meta = extract_column_meta(result_meta)
        assert column_meta['party']['colors'] == {"A": "red", "B": "blue", "C": "green"}
        
        # Also verify group_columns_dict works
        group_cols = group_columns_dict(result_meta)
        assert group_cols == {"test": ["party"]}
    
    def test_groups_definition(self, csv_file, meta_file):
        """Test groups parameter"""
        pd.DataFrame({'category': ['A', 'B', 'C', 'Other']}).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": [["category", {
                    "categories": ["A", "B", "C", "Other"],
                    "groups": {"main": ["A", "B", "C"]}
                }]]
            }]
        }
        write_json(meta_file, meta)
        
        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)
        # Verify groups are preserved using extract_column_meta
        column_meta = extract_column_meta(result_meta)
        assert column_meta['category']['groups'] == {"main": ["A", "B", "C"]}
        
        # Verify group_columns_dict functionality
        group_cols = group_columns_dict(result_meta)
        assert group_cols == {"test": ["category"]}
    
    def test_hidden_columns(self, csv_file, meta_file):
        """Test hidden column metadata"""
        pd.DataFrame({'visible': [1, 2, 3], 'hidden': [4, 5, 6]}).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "visible_group",
                "columns": ["visible"]
            }, {
                "name": "hidden_group",
                "hidden": True,
                "columns": ["hidden"]
            }]
        }
        write_json(meta_file, meta)
        
        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)
        # Data should still be loaded
        assert 'hidden' in df.columns
        # But metadata should preserve hidden flag
        hidden_group = next(group for group in result_meta['structure'] if group['name'] == 'hidden_group')
        assert hidden_group.get('hidden') == True
    
    def test_label_metadata(self, csv_file, meta_file):
        """Test label metadata preservation"""
        pd.DataFrame({'question': ['Yes', 'No', 'Maybe']}).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": [["question", {
                    "categories": ["Yes", "No", "Maybe"],
                    "label": "Do you agree with this statement?"
                }]]
            }]
        }
        write_json(meta_file, meta)
        
        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)
        # Verify label is preserved
        question_col = next(col for group in result_meta['structure'] for col in group['columns'] if isinstance(col, list) and col[0] == 'question')
        assert question_col[-1]['label'] == "Do you agree with this statement?"
    
    def test_complex_likert_scales(self, csv_file, meta_file):
        """Test complex likert scale with all features"""
        pd.DataFrame({'response': ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree']}).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": [["response", {
                    "categories": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
                    "ordered": True,
                    "likert": True,
                    "num_values": [-2, -1, 0, 1, 2],
                    "label": "How much do you agree?"
                }]]
            }]
        }
        write_json(meta_file, meta)
        
        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)
        assert df['response'].dtype.ordered == True
        # Verify all metadata is preserved
        resp_col = next(col for group in result_meta['structure'] for col in group['columns'] if isinstance(col, list) and col[0] == 'response')
        metadata = resp_col[-1]
        assert metadata['likert'] == True
        assert metadata['num_values'] == [-2, -1, 0, 1, 2]
        assert metadata['label'] == "How much do you agree?"
    
    def test_preprocessing_execution(self, csv_file, meta_file):
        """Test preprocessing execution"""
        pd.DataFrame({
            'value': [1, 2, 3, 4, 5],
            'multiplier': [2, 2, 2, 2, 2]
        }).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "preprocessing": "df['computed'] = df['value'] * df['multiplier']",
            "structure": [{
                "name": "test",
                "columns": ["value", "computed"]
            }]
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file))
        
        assert 'computed' in df.columns
        assert df['computed'].tolist() == [2, 4, 6, 8, 10]
    
    def test_postprocessing_execution(self, csv_file, meta_file):
        """Test postprocessing execution"""
        pd.DataFrame({'value': [1, 2, 3, 4, 5]}).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": ["value"]
            }],
            "postprocessing": "df['final'] = df['value'] + 100"
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file))
        
        assert 'final' in df.columns
        assert df['final'].tolist() == [101, 102, 103, 104, 105]
    
    def test_exclusions_handling(self, csv_file, meta_file):
        """Test exclusions handling"""
        pd.DataFrame({
            'value': [1, 2, 3, 4, 5],
            'id': [1, 2, 3, 4, 5]
        }).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": ["id", "value"]
            }],
            "excluded": [[1, "test exclusion"], [3, "another exclusion"]]  # Exclude rows 1 and 3
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
        pd.DataFrame({'value': [1, 2, 3, 4, 5]}).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": ["value"]
            }],
            "excluded": [[1, "test exclusion"], [3, "another exclusion"]]
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file), ignore_exclusions=True)
        
        # Should have all 5 rows when ignoring exclusions
        assert len(df) == 5
    
    def test_column_prefixing(self, csv_file, meta_file):
        """Test column prefixing"""
        pd.DataFrame({
            'q1': ['A', 'B', 'C'],
            'q2': ['X', 'Y', 'Z'],
            'id': [1, 2, 3]
        }).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "questions",
                "scale": {"col_prefix": "survey_"},
                "columns": ["q1", "q2"]
            }, {
                "name": "basic",
                "columns": ["id"]
            }]
        }
        write_json(meta_file, meta)
        
        df, result_meta = read_annotated_data(str(meta_file), return_meta=True)
        
        assert 'survey_q1' in df.columns
        assert 'survey_q2' in df.columns
        assert 'id' in df.columns
        
        # Verify prefixing works correctly with utility functions
        group_cols = group_columns_dict(result_meta)
        assert group_cols == {
            "questions": ["survey_q1", "survey_q2"],
            "basic": ["id"]
        }
        
        column_meta = extract_column_meta(result_meta)
        assert 'survey_q1' in column_meta
        assert 'survey_q2' in column_meta
        assert column_meta['survey_q1']['col_prefix'] == 'survey_'
    
    def test_subgroup_transform(self, csv_file, meta_file):
        """Test subgroup transform"""
        pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        }).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": ["a", "b", "c"],
                "subgroup_transform": "gdf + 10"  # Add 10 to all columns in group
            }]
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file))
        
        assert df['a'].tolist() == [11, 12, 13]
        assert df['b'].tolist() == [14, 15, 16]
        assert df['c'].tolist() == [17, 18, 19]

class TestMultipleFiles:
    """Test multiple file handling"""
    
    def test_multiple_files_concatenation(self, temp_dir, meta_file):
        """Test concatenation of multiple files"""
        # Create two CSV files
        csv_file1 = temp_dir / "test1.csv"
        csv_file2 = temp_dir / "test2.csv"
        
        pd.DataFrame({
            'id': [1, 2], 'value': ['A', 'B'], 'source': ['file1', 'file1']
        }).to_csv_file(csv_file1)
        
        pd.DataFrame({
            'id': [3, 4], 'value': ['C', 'D'], 'source': ['file2', 'file2']
        }).to_csv_file(csv_file2)
        
        meta = {
            "files": [
                {"file": "test1.csv"},
                {"file": "test2.csv"}
            ],
            "structure": [{
                "name": "test",
                "columns": ["id", "value", "source"]
            }]
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file))
        
        assert len(df) == 4
        assert set(df['source'].values) == {'file1', 'file2'}
        assert df['id'].tolist() == [1, 2, 3, 4]
    
    def test_multiple_files_with_extra_columns(self, temp_dir, meta_file):
        """Test multiple files with extra metadata columns"""
        csv_file1 = temp_dir / "test1.csv"
        csv_file2 = temp_dir / "test2.csv"
        
        pd.DataFrame({'id': [1, 2], 'value': ['A', 'B']}).to_csv_file(csv_file1)
        pd.DataFrame({'id': [3, 4], 'value': ['C', 'D']}).to_csv_file(csv_file2)
        
        meta = {
            "files": [
                {"file": "test1.csv", "wave": 1, "survey_date": "2023-01-01"},
                {"file": "test2.csv", "wave": 2, "survey_date": "2023-02-01"}
            ],
            "structure": [{
                "name": "test",
                "columns": ["id", "value", "wave", "survey_date"]
            }]
        }
        write_json(meta_file, meta)
        
        df = read_annotated_data(str(meta_file))
        
        assert len(df) == 4
        assert 'wave' in df.columns
        assert 'survey_date' in df.columns
        assert df[df['id'].isin([1, 2])]['wave'].iloc[0] == 1
        assert df[df['id'].isin([3, 4])]['wave'].iloc[0] == 2

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
            "filter": "df.age > 25"
        }
        
        df = read_and_process_data(desc)
        
        assert 'new_col' in df.columns
        assert len(df) < len(sample_csv_data)  # Filter should reduce rows
        assert all(df['age'] > 25)  # Filter condition
        assert df['new_col'].equals(df['age'] * 2)  # Preprocessing
    
    def test_direct_data_input(self):
        """Test direct data input via 'data' parameter"""
        desc = {
            "data": {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "value": [10, 20, 30]
            },
            "preprocessing": "df['doubled'] = df['value'] * 2"
        }
        
        df = read_and_process_data(desc)
        
        assert len(df) == 3
        assert 'doubled' in df.columns
        assert df['doubled'].tolist() == [20, 40, 60]
    
    def test_postprocessing_execution(self, csv_file, sample_csv_data):
        """Test postprocessing execution"""
        sample_csv_data.to_csv_file(csv_file)
        
        desc = {
            "file": str(csv_file),
            "postprocessing": "df['age_category'] = df['age'].apply(lambda x: 'Young' if x < 30 else 'Old')"
        }
        
        df = read_and_process_data(desc)
        
        assert 'age_category' in df.columns
        assert set(df['age_category'].values) == {'Young', 'Old'}
    
    def test_skip_postprocessing_parameter(self, csv_file, sample_csv_data):
        """Test skip_postprocessing parameter"""
        sample_csv_data.to_csv_file(csv_file)
        
        desc = {
            "file": str(csv_file),
            "postprocessing": "df['should_not_exist'] = 1"
        }
        
        df = read_and_process_data(desc, skip_postprocessing=True)
        
        assert 'should_not_exist' not in df.columns
    
    def test_constants_parameter(self, csv_file, sample_csv_data):
        """Test constants parameter"""
        sample_csv_data.to_csv_file(csv_file)
        
        desc = {
            "file": str(csv_file),
            "preprocessing": "df['multiplied'] = df['age'] * multiplier"
        }
        
        constants = {"multiplier": 5}
        
        df = read_and_process_data(desc, constants=constants)
        
        assert 'multiplied' in df.columns
        assert df['multiplied'].equals(df['age'] * 5)

class TestFileTracking:
    """Test file tracking functionality"""
    
    def test_file_tracking(self, csv_file, meta_file, sample_csv_data):
        """Test that loaded files are tracked"""
        reset_file_tracking()
        
        sample_csv_data.to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": ["id", "name"]
            }]
        }
        write_json(meta_file, meta)
        
        read_annotated_data(str(meta_file))
        
        loaded_files = get_loaded_files()
        assert len(loaded_files) > 0
        assert str(csv_file) in loaded_files

class TestMetadataUtilities:
    """Test metadata utility functions"""
    
    def test_extract_column_meta_basic(self):
        """Test basic extract_column_meta functionality"""
        meta = {
            'structure': [
                {
                    'name': 'demographics',
                    'scale': {'categories': ['A', 'B', 'C'], 'ordered': True},
                    'columns': ['age', ['gender', {'categories': ['M', 'F']}]]
                },
                {
                    'name': 'voting',
                    'columns': ['party', 'vote_prob']
                }
            ]
        }
        
        result = extract_column_meta(meta)
        
        # Check group-level metadata
        assert 'demographics' in result
        assert result['demographics']['categories'] == ['A', 'B', 'C']
        assert result['demographics']['ordered'] == True
        assert result['demographics']['columns'] == ['age', 'gender']
        
        assert 'voting' in result
        assert result['voting']['columns'] == ['party', 'vote_prob']
        
        # Check individual column metadata
        assert 'age' in result
        assert result['age']['categories'] == ['A', 'B', 'C']  # Inherits from scale
        assert result['age']['ordered'] == True
        assert result['age']['label'] == None
        
        assert 'gender' in result
        assert result['gender']['categories'] == ['M', 'F']  # Override from column spec
        assert result['gender']['ordered'] == True  # Still inherits ordered from scale
        
        assert 'party' in result
        assert result['party']['label'] == None
        
        assert 'vote_prob' in result
        assert result['vote_prob']['label'] == None
    
    def test_extract_column_meta_with_prefix(self):
        """Test extract_column_meta with col_prefix"""
        meta = {
            'structure': [
                {
                    'name': 'questions',
                    'scale': {'col_prefix': 'q_', 'categories': ['Yes', 'No']},
                    'columns': ['1', '2', ['3', {'categories': ['A', 'B', 'C']}]]
                }
            ]
        }
        
        result = extract_column_meta(meta)
        
        # Group should have prefixed column names
        assert result['questions']['columns'] == ['q_1', 'q_2', 'q_3']
        
        # Individual columns should have prefixed names as keys
        assert 'q_1' in result
        assert result['q_1']['categories'] == ['Yes', 'No']
        assert result['q_1']['col_prefix'] == 'q_'
        
        assert 'q_2' in result
        assert result['q_2']['categories'] == ['Yes', 'No']
        
        assert 'q_3' in result
        assert result['q_3']['categories'] == ['A', 'B', 'C']  # Column-level override
        assert result['q_3']['col_prefix'] == 'q_'
    
    def test_extract_column_meta_complex_features(self):
        """Test extract_column_meta with complex features"""
        meta = {
            'structure': [
                {
                    'name': 'likert_scale',
                    'scale': {
                        'categories': ['Disagree', 'Neutral', 'Agree'],
                        'ordered': True,
                        'likert': True,
                        'num_values': [-1, 0, 1]
                    },
                    'columns': [
                        ['q1', {'label': 'Question 1'}],
                        ['q2', {'label': 'Question 2', 'categories': ['No', 'Yes']}]
                    ]
                }
            ]
        }
        
        result = extract_column_meta(meta)
        
        # Group metadata
        group_meta = result['likert_scale']
        assert group_meta['categories'] == ['Disagree', 'Neutral', 'Agree']
        assert group_meta['ordered'] == True
        assert group_meta['likert'] == True
        assert group_meta['num_values'] == [-1, 0, 1]
        assert group_meta['columns'] == ['q1', 'q2']
        
        # Column metadata inheritance and override
        assert result['q1']['categories'] == ['Disagree', 'Neutral', 'Agree']  # From scale
        assert result['q1']['ordered'] == True
        assert result['q1']['likert'] == True
        assert result['q1']['num_values'] == [-1, 0, 1]
        assert result['q1']['label'] == 'Question 1'  # Column-level label is preserved
        
        assert result['q2']['categories'] == ['No', 'Yes']  # Column override
        assert result['q2']['ordered'] == True  # Still inherits from scale
        assert result['q2']['likert'] == True
        assert result['q2']['label'] == 'Question 2'  # Column-level label is preserved
    
    def test_group_columns_dict_basic(self):
        """Test basic group_columns_dict functionality"""
        meta = {
            'structure': [
                {
                    'name': 'demographics',
                    'columns': ['age', 'gender', 'location']
                },
                {
                    'name': 'voting',
                    'columns': ['party', 'vote_prob']
                }
            ]
        }
        
        result = group_columns_dict(meta)
        
        assert result == {
            'demographics': ['age', 'gender', 'location'],
            'voting': ['party', 'vote_prob']
        }
    
    def test_group_columns_dict_with_prefix(self):
        """Test group_columns_dict with col_prefix"""
        meta = {
            'structure': [
                {
                    'name': 'survey_questions',
                    'scale': {'col_prefix': 'q_'},
                    'columns': ['1', '2', '3']
                },
                {
                    'name': 'demographics',
                    'columns': ['age', 'gender']
                }
            ]
        }
        
        result = group_columns_dict(meta)
        
        assert result == {
            'survey_questions': ['q_1', 'q_2', 'q_3'],
            'demographics': ['age', 'gender']
        }
    
    def test_group_columns_dict_mixed_column_specs(self):
        """Test group_columns_dict with mixed column specifications"""
        meta = {
            'structure': [
                {
                    'name': 'mixed_group',
                    'columns': [
                        'simple_col',
                        ['complex_col', {'categories': ['A', 'B']}],
                        ['renamed_col', 'original_col', {'transform': 's * 2'}]
                    ]
                }
            ]
        }
        
        result = group_columns_dict(meta)
        
        assert result == {
            'mixed_group': ['simple_col', 'complex_col', 'renamed_col']
        }
    
    def test_extract_column_meta_label_isolation(self):
        """Test that scale-level labels don't propagate to individual columns"""
        meta = {
            'structure': [
                {
                    'name': 'test_group',
                    'scale': {
                        'label': 'Group Label',
                        'categories': ['A', 'B']
                    },
                    'columns': [
                        'col1',
                        ['col2', {'label': 'Column 2 Label'}]
                    ]
                }
            ]
        }
        
        result = extract_column_meta(meta)
        
        # Group should keep its label
        assert 'label' in result['test_group']
        assert result['test_group']['label'] == 'Group Label'
        
        # Individual columns should have label set to None (cleared from scale) unless explicitly set
        assert result['col1']['label'] == None  # No column-level label specified
        assert result['col2']['label'] == 'Column 2 Label'  # Column-level label is preserved
    
    def test_get_original_column_names_bug_fix(self):
        """Test that get_original_column_names handles strings correctly (bug fix)"""
        from salk_toolkit.io import get_original_column_names
        
        # Test simple string columns (this was the bug case)
        meta_simple = {
            'structure': [{
                'name': 'test_group',
                'columns': ['column1', 'column2', 'column_with_long_name']
            }]
        }
        
        result_simple = get_original_column_names(meta_simple)
        expected_simple = {
            'column1': 'column1',
            'column2': 'column2', 
            'column_with_long_name': 'column_with_long_name'
        }
        assert result_simple == expected_simple
        
        # Test mixed column formats
        meta_mixed = {
            'structure': [{
                'name': 'test_group',
                'columns': [
                    'simple_string_col',  # Simple string
                    ['single_element'],   # Single-element list
                    ['renamed_col', 'original_col', {'metadata': 'value'}],  # Rename format
                    ['col_with_meta', {'categories': ['A', 'B']}]  # Column with metadata
                ]
            }]
        }
        
        result_mixed = get_original_column_names(meta_mixed)
        expected_mixed = {
            'simple_string_col': 'simple_string_col',
            'single_element': 'single_element',
            'original_col': 'renamed_col',  # old_name -> new_name mapping
            'col_with_meta': 'col_with_meta'  # Regular column with metadata
        }
        assert result_mixed == expected_mixed
        
        # Test multiple groups
        meta_multi_group = {
            'structure': [
                {
                    'name': 'group1',
                    'columns': ['col1', 'col2']
                },
                {
                    'name': 'group2', 
                    'columns': [['new_name', 'old_name'], 'col3']
                }
            ]
        }
        
        result_multi = get_original_column_names(meta_multi_group)
        expected_multi = {
            'col1': 'col1',
            'col2': 'col2',
            'old_name': 'new_name',
            'col3': 'col3'
        }
        assert result_multi == expected_multi

class TestReplaceDataMetaInParquet:
    """Test replace_data_meta_in_parquet function"""
    
    def test_replace_data_meta_basic_operations(self, temp_dir):
        """Test basic metadata operations: ordered flag changes, metadata-only updates, and category reordering"""
        parquet_file = temp_dir / "test_basic.parquet"
        new_meta_file = temp_dir / "basic_meta.json"
        
        # Create initial data with mixed column types
        df = pd.DataFrame({
            'status': pd.Categorical(['A', 'B', 'C', 'A', 'B'], categories=['A', 'B', 'C'], ordered=False),
            'score': [1, 2, 3, 4, 5],
            'rating': pd.Categorical(['Good', 'Bad', 'Excellent', 'Good', 'Bad'],
                                   categories=['Good', 'Bad', 'Excellent'], ordered=False)
        })
        
        original_meta = {
            "data": {
                "structure": [{
                    "name": "test_group",
                    "columns": [
                        ["status", {"categories": ["A", "B", "C"], "ordered": False}],
                        ["score", {"continuous": True}],
                        ["rating", {"categories": ["Good", "Bad", "Excellent"], "ordered": False}]
                    ]
                }]
            },
            "model": {"version": "1.0"}
        }
        
        write_parquet_with_metadata(df, original_meta, str(parquet_file))
        
        # New metadata with multiple changes: ordered flags, metadata additions, category reordering
        new_meta = {
            "structure": [{
                "name": "test_group", 
                "columns": [
                    ["status", {"categories": ["A", "B", "C"], "ordered": True, "label": "Status Level"}],  # Add ordered + label
                    ["score", {"continuous": True, "label": "Numeric Score"}],  # Add label only
                    ["rating", {"categories": ["Bad", "Good", "Excellent"], "ordered": True, "colors": {"Bad": "red", "Good": "yellow", "Excellent": "green"}}]  # Reorder + ordered + colors
                ]
            }]
        }
        write_json(new_meta_file, new_meta)
        
        # Replace metadata
        result_df, result_meta = replace_data_meta_in_parquet(str(parquet_file), str(new_meta_file))
        
        # Verify data transformations
        assert result_df['status'].dtype.ordered == True  # Should now be ordered
        assert result_df['rating'].dtype.ordered == True  # Should now be ordered
        assert list(result_df['rating'].dtype.categories) == ["Bad", "Good", "Excellent"]  # Should be reordered
        assert set(result_df['status'].values) == {'A', 'B', 'C'}  # Status values unchanged
        assert set(result_df['rating'].values) == {'Good', 'Bad', 'Excellent'}  # Rating values unchanged
        
        # Verify metadata structure preservation
        assert 'original_data' in result_meta  # Original metadata should be preserved
        assert result_meta['data'] != result_meta['original_data']  # Should be different
        assert result_meta['model']['version'] == "1.0"  # Model metadata should be preserved
        
        # Verify new metadata is applied using utility functions
        column_meta = extract_column_meta(result_meta['data'])
        assert column_meta['status']['ordered'] == True
        assert column_meta['status']['label'] == "Status Level"
        assert column_meta['score']['label'] == "Numeric Score"
        assert column_meta['rating']['categories'] == ["Bad", "Good", "Excellent"]
        assert column_meta['rating']['ordered'] == True
        assert column_meta['rating']['colors'] == {"Bad": "red", "Good": "yellow", "Excellent": "green"}
        
        # Verify group_columns_dict works
        group_cols = group_columns_dict(result_meta['data'])
        assert group_cols == {"test_group": ["status", "score", "rating"]}
    
    def test_replace_data_meta_data_transformations(self, temp_dir):
        """Test data transformations: column renaming, translations, and constants"""
        parquet_file = temp_dir / "test_transformations.parquet"
        new_meta_file = temp_dir / "transformations_meta.json"
        
        # Create initial data with translated values A,B,C (after X,Y,Z -> A,B,C translation)
        df = pd.DataFrame({
            'old_name': pd.Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C']),
            'keep_same': [1, 2, 3],
            'response': pd.Categorical(['Yes', 'No', 'Maybe'], categories=['Yes', 'No', 'Maybe']),
            'party': pd.Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C'])  # Translated values
        })
        
        original_meta = {
            "data": {
                "structure": [{
                    "name": "test_group",
                    "columns": [
                        ["old_name", {"categories": ["A", "B", "C"]}],
                        ["keep_same", {"continuous": True}],
                        ["response", {"categories": ["Yes", "No", "Maybe"]}],
                        ["party", {"categories": ["A", "B", "C"], "translate": {"X": "A", "Y": "B", "Z": "C"}}]
                    ]
                }]
            }
        }
        
        write_parquet_with_metadata(df, original_meta, str(parquet_file))
        
        # Test case: Should work - X,Y,Z are raw values
        new_meta = {
            "constants": {
                "party_mapping": {"X": "Reform", "Y": "EKRE", "Z": "Center"}  # X,Y,Z are raw values
            },
            "structure": [{
                "name": "test_group",
                "columns": [
                    ["renamed_col", "old_name", {"categories": ["A", "B", "C"]}],
                    ["keep_same", {"continuous": True}],
                    ["response", {
                        "categories": ["Jah", "Ei", "Võib-olla"],
                        "translate": {"Yes": "Jah", "No": "Ei", "Maybe": "Võib-olla"}
                    }],
                    ["party", {
                        "categories": ["Reform", "EKRE", "Center"],
                        "translate": "party_mapping"
                    }]
                ]
            }]
        }
        write_json(new_meta_file, new_meta)
        
        # This should work
        result_df, result_meta = replace_data_meta_in_parquet(str(parquet_file), str(new_meta_file), advanced=True)
        
        # Verify results
        assert set(result_df.columns) == {'renamed_col', 'keep_same', 'response', 'party'}
        assert 'old_name' not in result_df.columns
        assert result_df['renamed_col'].tolist() == ['A', 'B', 'C']
        assert result_df['keep_same'].tolist() == [1, 2, 3]
        assert set(result_df['response'].values) == {'Jah', 'Ei', 'Võib-olla'}
        assert set(result_df['party'].values) == {'Reform', 'EKRE', 'Center'}
        
        # Verify metadata
        column_meta = extract_column_meta(result_meta['data'])
        assert 'renamed_col' in column_meta
        assert 'old_name' not in column_meta
        assert column_meta['response']['categories'] == ["Jah", "Ei", "Võib-olla"]
        assert column_meta['party']['categories'] == ["Reform", "EKRE", "Center"]
        
        group_cols = group_columns_dict(result_meta['data'])
        assert group_cols == {"test_group": ["renamed_col", "keep_same", "response", "party"]}
        
    
class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_missing_file_error(self, meta_file):
        """Test error when file doesn't exist"""
        meta = {
            "file": "nonexistent.csv",
            "structure": [{"name": "test", "columns": ["id"]}]
        }
        write_json(meta_file, meta)
        
        with pytest.raises(FileNotFoundError):
            read_annotated_data(str(meta_file))
    
    def test_invalid_transform_code(self, csv_file, meta_file):
        """Test error handling for invalid transform code"""
        pd.DataFrame({'value': [1, 2, 3]}).to_csv_file(csv_file)
        
        meta = {
            "file": "test.csv",
            "structure": [{
                "name": "test",
                "columns": [["value", {"transform": "invalid_python_code()"}]]
            }]
        }
        write_json(meta_file, meta)
        
        with pytest.raises((NameError, SyntaxError)):
            read_annotated_data(str(meta_file))

if __name__ == "__main__":
    pytest.main([__file__])
