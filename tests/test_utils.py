"""Comprehensive unit tests for salk_toolkit.utils module."""

import pytest
import numpy as np
import pandas as pd
import altair as alt
import warnings
from unittest.mock import patch
from collections import defaultdict, OrderedDict

from salk_toolkit.utils import (
    factorize_w_codes, batch, loc2iloc, match_sum_round, min_diff, continify,
    replace_cat_with_dummies, match_data, replace_constants, approx_str_match,
    index_encoder, to_alt_scale, multicol_to_vals_cats, gradient_to_discrete_color_scale,
    gradient_subrange, gradient_from_color, gradient_from_color_alt, split_to_neg_neutral_pos,
    is_datetime, rel_wave_times, stable_draws, deterministic_draws, clean_kwargs,
    call_kwsafe, censor_dict, cut_nice_labels, cut_nice, rename_cats, str_replace,
    merge_series, aggregate_multiselect, deaggregate_multiselect, gb_in, gb_in_apply,
    stk_defaultdict, cached_fn, scores_to_ordinal_rankings, dict_cache, get_size,
    escape_vega_label, unescape_vega_label, warn, default_color,
    default_bidirectional_gradient, redblue_gradient, greyscale_gradient
)


class TestBasicUtilities:
    """Test basic utility functions."""
    
    def test_factorize_w_codes(self):
        """Test factorize_w_codes function."""
        # Basic functionality
        s = pd.Series(['a', 'b', 'c', 'a'])
        codes = ['a', 'b', 'c']
        result = factorize_w_codes(s, codes)
        expected = np.array([0, 1, 2, 0], dtype='int')
        assert np.array_equal(result, expected)
        
        # With NaN values
        s = pd.Series(['a', 'b', None, 'c'])
        result = factorize_w_codes(s, codes)
        expected = np.array([0, 1, -1, 2], dtype='int')
        assert np.array_equal(result, expected)
        
        # Error case - value not in codes
        s = pd.Series(['a', 'b', 'd'])
        with pytest.raises(Exception, match="Codes for .* do not match all values"):
            factorize_w_codes(s, codes)
    
    def test_batch(self):
        """Test batch function."""
        # Basic batching
        data = [1, 2, 3, 4, 5, 6, 7]
        result = list(batch(data, 3))
        expected = [[1, 2, 3], [4, 5, 6], [7]]
        assert result == expected
        
        # Single element batches
        result = list(batch(data, 1))
        expected = [[1], [2], [3], [4], [5], [6], [7]]
        assert result == expected
        
        # Batch size larger than data
        result = list(batch(data, 10))
        expected = [[1, 2, 3, 4, 5, 6, 7]]
        assert result == expected
        
        # Empty data
        result = list(batch([], 3))
        assert result == []
    
    def test_loc2iloc(self):
        """Test loc2iloc function."""
        index = ['a', 'b', 'c', 'd']
        vals = ['c', 'a', 'd']
        result = loc2iloc(index, vals)
        expected = [2, 0, 3]
        assert result == expected
        
        # With numeric index
        index = [10, 20, 30]
        vals = [30, 10]
        result = loc2iloc(index, vals)
        expected = [2, 0]
        assert result == expected
    
    def test_match_sum_round(self):
        """Test match_sum_round function."""
        # Basic test case from notebook
        result = match_sum_round([0.7, 0.7, 0.6])
        expected = np.array([1, 1, 0])
        assert np.array_equal(result, expected)
        
        # Integer values should remain unchanged
        result = match_sum_round([1, 2, 3])
        expected = np.array([1, 2, 3])
        assert np.array_equal(result, expected)
        
        # Test with different fractional parts
        result = match_sum_round([2.3, 1.8, 0.9])
        assert result.sum() == 5  # Should preserve sum
        assert result.dtype == int
        
        # Single value
        result = match_sum_round([2.7])
        expected = np.array([3])
        assert np.array_equal(result, expected)


class TestNumericalUtilities:
    """Test numerical utility functions."""
    
    def test_min_diff(self):
        """Test min_diff function."""
        # Basic case from notebook
        result = min_diff([0, 0, 2, 1.5, 3, 3])
        assert result == 0.5
        
        # Empty array
        assert min_diff([]) == 0.0
        
        # All same values
        assert min_diff([1, 1, 1]) == 0.0
        
        # Two different values
        assert min_diff([1, 3]) == 2
        
        # Single value
        assert min_diff([5]) == 0.0
    
    def test_continify(self):
        """Test continify function."""
        np.random.seed(42)  # For reproducible tests
        
        # Basic functionality
        ar = np.array([0, 2, 4])
        result = continify(ar, bounded=False)
        assert len(result) == len(ar)
        assert not np.array_equal(result, ar)  # Should add noise
        
        # Bounded version
        ar = np.array([0, 2, 4, 1, 5])
        result = continify(ar, bounded=True)
        assert result.min() >= ar.min()
        assert result.max() <= ar.max()
        
        # With delta
        result = continify(ar, bounded=True, delta=0.1)
        assert len(result) == len(ar)


class TestDataManipulation:
    """Test data manipulation functions."""
    
    def test_replace_cat_with_dummies(self):
        """Test replace_cat_with_dummies function."""
        df = pd.DataFrame({'cat': ['a', 'b', 'c'], 'other': [1, 2, 3]})
        cs = ['a', 'b', 'c']
        result = replace_cat_with_dummies(df, 'cat', cs)
        
        assert 'cat' not in result.columns
        assert 'other' in result.columns
        assert 'b' in result.columns
        assert 'c' in result.columns
        assert 'a' not in result.columns  # First category is dropped
        assert len(result) == 3
    
    def test_replace_constants(self):
        """Test replace_constants function."""
        # Test case from notebook
        d = {
            'constants': {'a': {'a': 1}, 'b': ['b']},
            'test1': 'a',
            'test2': [1, 'b'],
            'test3': {'xy': 'a'},
            'test4': {'xy': [2, 'b']},
            'test5': {'constants': {'a': ['a']}, 'x': 'a'},
            'test6': 'a'
        }
        result = replace_constants(d)
        expected = {
            'test1': {'a': 1}, 
            'test2': [1, ['b']], 
            'test3': {'xy': {'a': 1}}, 
            'test4': {'xy': [2, ['b']]}, 
            'test5': {'x': ['a']}, 
            'test6': {'a': 1}
        }
        assert result == expected
        
        # Test inplace=False doesn't modify original
        original = {'constants': {'a': 1}, 'test': 'a'}
        result = replace_constants(original, inplace=False)
        assert 'constants' in original  # Original unchanged
        assert 'constants' not in result
    
    def test_approx_str_match(self):
        """Test approx_str_match function."""
        # Test case from notebook
        result = approx_str_match(['aaabc', 'xyz'], ['xxy', 'ac', 'dfg'])
        expected = {'aaabc': 'ac', 'xyz': 'xxy'}
        assert result == expected
        
        # Perfect matches
        result = approx_str_match(['a', 'b'], ['a', 'b'])
        expected = {'a': 'a', 'b': 'b'}
        assert result == expected


class TestStringUtilities:
    """Test string manipulation utilities."""
    
    def test_str_replace(self):
        """Test str_replace function."""
        # Test case from notebook
        s = pd.Series(['abc', 'bca', 'def'])
        result = str_replace(s, {'a': 'x', 'bc': 'y'})
        expected = pd.Series(['xy', 'yx', 'def'])
        assert (result == expected).all()
        
        # Empty replacement dict
        result = str_replace(s, {})
        assert (result == s).all()
        
        # No matches
        result = str_replace(s, {'z': 'w'})
        assert (result == s).all()
    
    def test_merge_series(self):
        """Test merge_series function."""
        # Test case from notebook
        df = pd.DataFrame({'a': ['a', 'a', 'a'], 'b': ['x', None, None], 'c': ['d', 'e', 'f']})
        result = merge_series(df['a'], df['b'], (df['c'], ['f', 'g']))
        expected = pd.Series(['x', 'a', 'f'])
        assert (result == expected).all()
        
        # Simple merge without whitelist
        s1 = pd.Series(['a', None, 'c'])
        s2 = pd.Series(['x', 'y', None])
        result = merge_series(s1, s2)
        expected = pd.Series(['x', 'y', 'c'])
        assert (result == expected).all()


class TestColorUtilities:
    """Test color and visualization utilities."""
    
    def test_gradient_to_discrete_color_scale(self):
        """Test gradient_to_discrete_color_scale function."""
        # Test case from notebook
        grad = ['#ff0000', '#ffff00', '#00ff00']
        result = gradient_to_discrete_color_scale(grad, 4)
        expected = ['#ff0000', '#ffaa00', '#aaff00', '#00ff00']
        assert result == expected
        
        # Two colors (minimum for valid gradient)
        result = gradient_to_discrete_color_scale(['#ff0000', '#00ff00'], 3)
        assert len(result) == 3
        assert all(color.startswith('#') for color in result)
    
    def test_gradient_subrange(self):
        """Test gradient_subrange function."""
        grad = ['#ff0000', '#00ff00']
        result = gradient_subrange(grad, 4, range=[-0.5, 0.5], bidirectional=True)
        # The function creates more colors and then subsets them
        assert len(result) >= 2  # Should have at least some colors
        assert all(color.startswith('#') for color in result)
    
    def test_gradient_from_color(self):
        """Test gradient_from_color function."""
        result = gradient_from_color('#ff0000', n_points=5)
        assert len(result) == 5
        assert all(color.startswith('#') for color in result)
        
        # Test with different parameters
        result = gradient_from_color('#ff0000', l_value=0.5, range=[0.2, 0.8])
        assert len(result) == 7  # default n_points
    
    def test_gradient_from_color_alt(self):
        """Test gradient_from_color_alt function."""
        result = gradient_from_color_alt('#ff0000', n_points=3)
        assert len(result) == 3
        assert all(color.startswith('#') for color in result)
    
    def test_to_alt_scale(self):
        """Test to_alt_scale function."""
        # None input
        result = to_alt_scale(None)
        assert result == alt.Undefined
        
        # Dictionary input
        scale_dict = {'a': '#ff0000', 'b': '#00ff00'}
        result = to_alt_scale(scale_dict)
        assert isinstance(result, alt.Scale)
        
        # Dictionary with order
        result = to_alt_scale(scale_dict, order=['b', 'a'])
        assert isinstance(result, alt.Scale)
        
        # Non-dict input should pass through
        result = to_alt_scale('linear')
        assert result == 'linear'
    
    def test_escape_unescape_vega_label(self):
        """Test escape_vega_label and unescape_vega_label functions."""
        label = 'test.with[brackets]'
        escaped = escape_vega_label(label)
        assert escaped == 'test․with［brackets］'
        
        unescaped = unescape_vega_label(escaped)
        assert unescaped == label


class TestDataProcessing:
    """Test data processing functions."""
    
    def test_multicol_to_vals_cats(self):
        """Test multicol_to_vals_cats function."""
        # Test case from notebook
        df = pd.DataFrame({'q1': ['a', 'b', 'c', None, None, None], 
                          'q1b': [None, None, None, 'c', 'b', 'a']})
        result = multicol_to_vals_cats(df, col_prefix='q1', reverse_suffixes=['1b'], 
                                     cat_order=['a', 'b', 'c'])
        expected_vals = ['a', 'b', 'c', 'a', 'b', 'c']
        assert (result['vals'] == expected_vals).all()
        
        # Test without reverse
        df = pd.DataFrame({'test1': ['x', None], 'test2': [None, 'y']})
        result = multicol_to_vals_cats(df, col_prefix='test')
        assert 'vals' in result.columns
        assert 'cats' in result.columns
    
    def test_cut_nice_labels(self):
        """Test cut_nice_labels function."""
        breaks = [0, 20, 30, 40]
        mi, ma = 5, 35
        result_breaks, labels = cut_nice_labels(breaks, mi, ma, isint=True)
        
        assert len(labels) == len(result_breaks) - 1
        assert all(isinstance(label, str) for label in labels)
    
    def test_cut_nice(self):
        """Test cut_nice function."""
        # Test case from notebook
        s = [19, 20, 29, 30, 39, 199]
        breaks = [0, 20.0, 30, 40, 50, 60, 70]
        result = cut_nice(s, breaks)
        expected = ['0 - 19', '20 - 29', '20 - 29', '30 - 39', '30 - 39', '70+']
        assert (result == expected).all()
        
        # Test with values outside breaks
        s = [19, 20, 29, 30, 39]
        breaks = [20.0, 30, 40, 50, 60, 70]
        result = cut_nice(s, breaks)
        expected = ['<20', '20 - 29', '20 - 29', '30 - 39', '30 - 39']
        assert (result == expected).all()
    
    def test_aggregate_multiselect(self):
        """Test aggregate_multiselect function."""
        df = pd.DataFrame({
            'q1_opt1': ['selected', 'not_selected', 'selected'],
            'q1_opt2': ['not_selected', 'selected', 'not_selected'],
            'other': ['a', 'b', 'c']
        })
        
        # Should work when na_vals are found
        aggregate_multiselect(df, 'q1_', 'result_', na_vals=['not_selected'], 
                            colnames_as_values=True, inplace=True)
        
        assert 'result_1' in df.columns
        
        # Should raise error when no na_vals found
        df2 = pd.DataFrame({
            'q1_opt1': ['a', 'b', 'c'],
            'q1_opt2': ['d', 'e', 'f']
        })
        with pytest.raises(ValueError, match="No na_vals found"):
            aggregate_multiselect(df2, 'q1_', 'result_', na_vals=['not_found'])
    
    def test_deaggregate_multiselect(self):
        """Test deaggregate_multiselect function."""
        df = pd.DataFrame({
            'choice1': ['a', 'b', None],
            'choice2': ['b', None, 'c'],
            'other': [1, 2, 3]
        })
        
        deaggregate_multiselect(df, 'choice', 'out_')
        
        assert 'out_a' in df.columns
        assert 'out_b' in df.columns
        assert 'out_c' in df.columns
        assert df['out_a'].iloc[0] == True
        assert df['out_b'].iloc[0] == True


class TestTimeAndRandomUtilities:
    """Test time and random utility functions."""
    
    def test_stable_draws(self):
        """Test stable_draws function."""
        # Test case from notebook
        result = stable_draws(20, 5, 'test')
        expected = np.array([1, 2, 3, 3, 2, 3, 2, 2, 0, 0, 0, 3, 4, 4, 1, 1, 1, 0, 4, 4])
        assert np.array_equal(result, expected)
        
        # Same input should give same result
        result2 = stable_draws(20, 5, 'test')
        assert np.array_equal(result, result2)
        
        # Different uid should give different result
        result3 = stable_draws(20, 5, 'different')
        assert not np.array_equal(result, result3)
    
    def test_deterministic_draws(self):
        """Test deterministic_draws function."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        result = deterministic_draws(df, 3, 'test')
        
        assert 'draw' in result.columns
        assert len(result) == 5
        assert result['draw'].max() < 3
        assert result['draw'].min() >= 0
    
    def test_rel_wave_times(self):
        """Test rel_wave_times function."""
        ws = pd.Series([1, 1, 2, 2, 3])
        dts = pd.Series(['2020-01-01', '2020-01-02', '2020-02-01', '2020-02-02', '2020-03-01'])
        
        result = rel_wave_times(ws, dts)
        
        assert len(result) == len(ws)
        assert result.name == 't'
        assert result.max() <= 0  # Last wave should be reference (0 or negative)
    
    def test_is_datetime(self):
        """Test is_datetime function."""
        # Datetime column
        dt_col = pd.Series(pd.date_range('2020-01-01', periods=3))
        assert is_datetime(dt_col)
        
        # String dates that can be parsed
        str_col = pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'])
        assert is_datetime(str_col)
        
        # Non-datetime column
        num_col = pd.Series([1, 2, 3])
        assert not is_datetime(num_col)
        
        # Mixed column with some dates
        mixed_col = pd.Series(['2020-01-01', 'not_a_date', '2020-01-03'])
        assert is_datetime(mixed_col)  # Should return True if any can be parsed


class TestHelperFunctions:
    """Test helper functions and utilities."""
    
    def test_clean_kwargs(self):
        """Test clean_kwargs function."""
        def test_fn(a, b, c=None):
            return a + b
        
        kwargs = {'a': 1, 'b': 2, 'c': 3, 'd': 4}  # 'd' not in function signature
        result = clean_kwargs(test_fn, kwargs)
        expected = {'a': 1, 'b': 2, 'c': 3}
        assert result == expected
        
        # Function with **kwargs should pass everything through
        def test_fn_varkw(a, **kwargs):
            return a
        
        result = clean_kwargs(test_fn_varkw, kwargs)
        assert result == kwargs
    
    def test_call_kwsafe(self):
        """Test call_kwsafe function."""
        def test_fn(a, b):
            return a + b
        
        result = call_kwsafe(test_fn, 1, 2, c=3, d=4)  # Extra kwargs should be ignored
        assert result == 3
    
    def test_censor_dict(self):
        """Test censor_dict function."""
        d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        result = censor_dict(d, ['b', 'd'])
        expected = {'a': 1, 'c': 3}
        assert result == expected
    
    def test_rename_cats(self):
        """Test rename_cats function."""
        # Categorical column
        df = pd.DataFrame({'cat': pd.Categorical(['a', 'b', 'c'])})
        rename_cats(df, 'cat', {'a': 'x', 'b': 'y'})
        assert 'x' in df['cat'].cat.categories
        assert 'y' in df['cat'].cat.categories
        
        # Non-categorical column
        df = pd.DataFrame({'col': ['a', 'b', 'c']})
        rename_cats(df, 'col', {'a': 'x', 'b': 'y'})
        assert (df['col'] == ['x', 'y', 'c']).all()
    
    def test_gb_in(self):
        """Test gb_in function."""
        df = pd.DataFrame({'group': ['a', 'a', 'b'], 'value': [1, 2, 3]})
        
        # With groupby columns
        result = gb_in(df, ['group'])
        assert hasattr(result, 'groups')  # Should be a GroupBy object
        
        # Without groupby columns
        result = gb_in(df, [])
        assert isinstance(result, pd.DataFrame)  # Should be the original DataFrame
    
    def test_gb_in_apply(self):
        """Test gb_in_apply function."""
        df = pd.DataFrame({'group': ['a', 'a', 'b'], 'value': [1, 2, 3]})
        
        # With groupby - specify numeric columns only
        result = gb_in_apply(df, ['group'], lambda x: x.mean(), cols=['value'])
        assert len(result) == 2  # Two groups
        
        # Without groupby
        result = gb_in_apply(df, [], lambda x: x.mean(), cols=['value'])
        assert len(result) == 1  # Single result
    
    def test_stk_defaultdict(self):
        """Test stk_defaultdict function."""
        # Simple default value
        dd = stk_defaultdict(42)
        assert dd['nonexistent'] == 42
        
        # Dictionary with default
        dd = stk_defaultdict({'default': 'def', 'a': 'val_a'})
        assert dd['a'] == 'val_a'
        assert dd['nonexistent'] == 'def'
    
    def test_cached_fn(self):
        """Test cached_fn function."""
        call_count = 0
        
        def expensive_fn(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        cached = cached_fn(expensive_fn)
        
        # First call
        result1 = cached(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call with same input - should use cache
        result2 = cached(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
        
        # Call with different input
        result3 = cached(3)
        assert result3 == 6
        assert call_count == 2
    
    def test_scores_to_ordinal_rankings(self):
        """Test scores_to_ordinal_rankings function."""
        # Test case from notebook
        df = pd.DataFrame({'a': [1, 2], 'b': [2, 2], 'c': [np.nan, 1]})
        result = scores_to_ordinal_rankings(df, ['a', 'b', 'c'], 'test')
        
        assert 'test_orank' in result.columns
        assert 'test_ties' in result.columns
        
        expected_orank = [['b', 'a'], ['a', 'b', 'c']]
        expected_ties = [[0, 0], [1, 0, 0]]
        
        assert result['test_orank'].iloc[0] == expected_orank[0]
        assert result['test_orank'].iloc[1] == expected_orank[1]
        assert result['test_ties'].iloc[0] == expected_ties[0]
        assert result['test_ties'].iloc[1] == expected_ties[1]


class TestAdvancedUtilities:
    """Test advanced utility classes and functions."""
    
    def test_dict_cache(self):
        """Test dict_cache class."""
        cache = dict_cache(size=2)
        
        # Add items
        cache['a'] = 1
        cache['b'] = 2
        assert len(cache) == 2
        
        # Add third item - should evict first
        cache['c'] = 3
        assert len(cache) == 2
        assert 'a' not in cache
        assert 'b' in cache
        assert 'c' in cache
        
        # Access should move to end
        _ = cache['b']
        cache['d'] = 4
        assert 'c' not in cache  # 'c' should be evicted, not 'b'
        assert 'b' in cache
        assert 'd' in cache
    
    def test_get_size(self):
        """Test get_size function."""
        # Simple objects
        assert get_size(42) > 0
        assert get_size("hello") > 0
        assert get_size([1, 2, 3]) > get_size([])
        
        # Dictionary
        d = {'a': 1, 'b': [1, 2, 3]}
        size = get_size(d)
        assert size > 0
        
        # Circular reference handling
        d1 = {'ref': None}
        d2 = {'ref': d1}
        d1['ref'] = d2
        size = get_size(d1)  # Should not infinite loop
        assert size > 0
    
    def test_index_encoder(self):
        """Test index_encoder function."""
        # Pandas Index
        idx = pd.Index(['a', 'b', 'c'])
        result = index_encoder(idx)
        assert result == ['a', 'b', 'c']
        
        # Non-Index object should raise TypeError
        with pytest.raises(TypeError, match="Object of type .* is not serializable"):
            index_encoder(42)
    
    def test_split_to_neg_neutral_pos(self):
        """Test split_to_neg_neutral_pos function."""
        # Odd number of categories, no neutrals specified
        cats = ['strongly_disagree', 'disagree', 'neutral', 'agree', 'strongly_agree']
        neg, neutral, pos = split_to_neg_neutral_pos(cats, [])
        assert len(neg) == 2
        assert len(neutral) == 1
        assert len(pos) == 2
        assert neutral == ['neutral']
        
        # Even number, no neutrals
        cats = ['disagree', 'somewhat_disagree', 'somewhat_agree', 'agree']
        neg, neutral, pos = split_to_neg_neutral_pos(cats, [])
        assert len(neg) == 2
        assert len(neutral) == 0
        assert len(pos) == 2
        
        # With specified neutrals
        cats = ['bad', 'neutral1', 'good', 'neutral2']
        neutrals = ['neutral1', 'neutral2']
        neg, neutral, pos = split_to_neg_neutral_pos(cats, neutrals)
        assert neg == ['bad']
        assert pos == ['good']
        assert set(neutral) == set(neutrals)


class TestConstants:
    """Test module constants."""
    
    def test_constants_exist(self):
        """Test that all expected constants are defined."""
        assert default_color == 'lightgrey'
        assert isinstance(default_bidirectional_gradient, list)
        assert isinstance(redblue_gradient, list)
        assert isinstance(greyscale_gradient, list)
        
        # Check gradients have valid hex colors
        for gradient in [default_bidirectional_gradient, redblue_gradient, greyscale_gradient]:
            assert all(color.startswith('#') for color in gradient)
    
    def test_warn_function(self):
        """Test warn function."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn("Test warning message")
            assert len(w) == 1
            assert "Test warning message" in str(w[0].message)


if __name__ == "__main__":
    pytest.main([__file__])
