"""Utilities
---------

Cross-cutting helpers that used to be scattered across `10_utils.ipynb` now
live here.  The module groups:

- colour helpers, gradients, translation utilities, and Altair niceties
- dataframe mangling helpers (factorisation, batching, deterministic draws)
- caching/warning helpers that IO, plotting, dashboards, and election modules
  all depend on

If you need a generic helper, check this file before adding another bespoke
version elsewhere.
"""

from __future__ import annotations

__all__ = [
    "aggregate_multiselect",
    "approx_str_match",
    "batch",
    "cached_fn",
    "call_kwsafe",
    "censor_dict",
    "clean_kwargs",
    "continify",
    "cut_nice",
    "cut_nice_labels",
    "deaggregate_multiselect",
    "deterministic_draws",
    "dict_cache",
    "escape_vega_label",
    "factorize_w_codes",
    "gb_in",
    "gb_in_apply",
    "get_categories",
    "get_ordered",
    "get_size",
    "gradient_from_color",
    "gradient_from_color_alt",
    "gradient_subrange",
    "gradient_to_discrete_color_scale",
    "index_encoder",
    "is_datetime",
    "loc2iloc",
    "match_data",
    "match_sum_round",
    "merge_pydantic_models",
    "merge_series",
    "min_diff",
    "multicol_to_vals_cats",
    "read_json",
    "read_yaml",
    "rel_wave_times",
    "rename_cats",
    "replace_cat_with_dummies",
    "replace_constants",
    "scores_to_ordinal_rankings",
    "split_to_neg_neutral_pos",
    "stable_draws",
    "stable_rng",
    "stk_defaultdict",
    "str_replace",
    "to_alt_scale",
    "unescape_vega_label",
    "warn",
    "plot_matrix_html",
]

import json
import re
import warnings
import math
import inspect
import sys
import yaml
from collections import defaultdict, OrderedDict
from copy import deepcopy
from hashlib import sha256
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    cast,
    List,
    Mapping,
    MutableSequence,
    Sequence,
    TypeAlias,
    TypeVar,
)

if TYPE_CHECKING:
    from salk_toolkit.pp import AltairChart


import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import altair as alt
import matplotlib.colors as mpc
import hsluv  # type: ignore[import-untyped]

import Levenshtein
from pydantic import BaseModel

pd.set_option("future.no_silent_downcasting", True)


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | dict[str, "JSONValue"] | list["JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]
JSONArray: TypeAlias = list[JSONValue]


ModelT = TypeVar("ModelT", bound=BaseModel)


def merge_pydantic_models(
    defaults: ModelT | None, overrides: ModelT, *, context: dict[str, object] | None = None
) -> ModelT:
    """Merge two BaseModel instances, similar to ``{**defaults, **overrides}``."""

    if defaults is None:
        return overrides

    defaults_dict = defaults.model_dump(mode="python")
    # Important semantic: fields explicitly set on the override should replace defaults,
    # including explicit `None` (JSON `null`) which should clear inherited values.
    #
    # Note: Many of our Pydantic models use custom serializers that drop default-valued
    # fields, so `model_dump(exclude_unset=True)` is insufficient to detect "explicit None".
    # Use `model_fields_set` to preserve the user's intent.
    overrides_dict = {k: getattr(overrides, k) for k in overrides.model_fields_set}
    merged_dict = {**defaults_dict, **overrides_dict}
    return overrides.__class__.model_validate(merged_dict, context=context)


# convenience for warnings that gives a more useful stack frame (fn calling the warning, not warning fn itself)
def warn(msg: str, *args: object) -> None:
    """Emit a warning while pointing at the caller instead of this helper.

    Args:
        msg: Warning message to display.
        *args: Additional positional arguments forwarded to `warnings.warn`.
    """
    # mypy doesn't handle *args well with warn overloads
    warnings.warn(msg, *args, stacklevel=3)  # type: ignore[call-overload]


def factorize_w_codes(s: pd.Series, codes: Sequence[Any]) -> np.ndarray:
    """Return integer codes for `s` using an explicit category ordering.

    Args:
        s: Series with categorical values to encode.
        codes: Ordered collection describing the allowed category list.

    Returns:
        Integer numpy array matching the new categorical codes (missing → -1).

    Raises:
        Exception: If `s` contains values not present in `codes`.
    """

    res = s.astype("object").replace(dict(zip(codes, range(len(codes)))))
    if not s.dropna().isin(codes).all():  # Throw an exception if all values were not replaced
        vals = set(s) - set(codes)
        raise Exception(f"Codes for {s.name} do not match all values: {vals}")
    return res.fillna(-1).to_numpy(dtype="int")


T = TypeVar("T")


def batch(iterable: Sequence[T], n: int = 1) -> Iterator[Sequence[T]]:
    """Yield slices of ``n`` items from a sequence-like iterable.

    Args:
        iterable: Sequence to chunk; must support ``len`` and slicing.
        n: Maximum chunk size.

    Yields:
        Consecutive slices from ``iterable`` no larger than ``n`` items.
    """

    total = len(iterable)
    for ndx in range(0, total, n):
        yield iterable[ndx : min(ndx + n, total)]


def loc2iloc(index: Sequence[Any], vals: Sequence[Any]) -> List[int]:
    """Convert label positions from ``index`` into zero-based integer offsets.

    Args:
        index: Original index labels defining order.
        vals: Values whose integer positions should be resolved.

    Returns:
        Integer offsets corresponding to ``vals`` within ``index``.
    """

    d = dict(zip(np.array(index), range(len(index))))
    return [d[v] for v in vals]


def match_sum_round(s: Sequence[float]) -> np.ndarray:
    """Round the values while preserving the original sum exactly.

    Args:
        s: Sequence of floats that should sum to the same value after rounding.

    Returns:
        Integer numpy array that maintains the rounded sum of ``s``.
    """

    s_arr = np.array(s)
    fs = np.floor(s_arr)
    diff = round(s_arr.sum() - fs.sum())
    residues = np.argsort(-(s_arr % 1))[:diff]
    fs[residues] = fs[residues] + 1
    return fs.astype("int")


def min_diff(arr: Sequence[float]) -> float:
    """Return the smallest strictly positive difference between sorted values.

    Args:
        arr: Sequence of numeric values.

    Returns:
        Minimum positive pairwise distance; ``0`` if all values are identical.
    """

    b = np.diff(np.sort(arr))
    if len(b) == 0 or b.max() == 0.0:
        return 0
    else:
        return b[b > 0].min()


def continify(ar: np.ndarray, bounded: bool = False, delta: float = 0.0) -> np.ndarray:
    """Add Gaussian noise to discrete values to mimic a continuous distribution.

    Args:
        ar: Numpy array to smooth.
        bounded: Whether to reflect noise at min/max bounds.
        delta: Shift applied when computing the bounds to avoid clipping.

    Returns:
        Array of the same shape with jitter applied.
    """

    mi, ma = ar.min() + delta, ar.max() - delta
    noise = np.random.normal(0, 0.5 * min_diff(ar), size=len(ar))
    res = ar + noise
    if bounded:  # Reflect the noise on the boundaries
        res[res > ma] = ma - (res[res > ma] - ma)
        res[res < mi] = mi + (mi - res[res < mi])
    return res


def replace_cat_with_dummies(df: pd.DataFrame, c: str, cs: Sequence[str]) -> pd.DataFrame:
    """Expand a categorical column into dummy variables (dropping the baseline).

    Args:
        df: Input dataframe.
        c: Column name to expand.
        cs: Category order; the first entry is dropped to avoid collinearity.

    Returns:
        Dataframe with ``c`` removed and dummy columns appended.
    """

    return pd.concat([df.drop(columns=[c]), pd.get_dummies(df[c])[cs[1:]].astype(float)], axis=1)


def match_data(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    cols: Sequence[str] | None = None,
) -> tuple[Sequence[int], Sequence[int]]:
    """Pair rows between two frames by minimizing Mahalanobis distance.

    Args:
        data1: First dataframe.
        data2: Second dataframe.
        cols: Columns to align on. Must exist in both frames. If None, uses all columns.

    Returns:
        Tuple of row indices describing optimal matches for ``data1`` and ``data2``.
    """
    if cols is None:
        cols = list(data1.columns)
    d1 = data1[cols].copy().dropna()
    d2 = data2[cols].copy().dropna()

    if len(d1) == 0 or len(d2) == 0:
        return [], []

    ccols = [c for c in cols if d1[c].dtype.name == "category"]
    for c in ccols:
        d1_dtype = d1[c].dtype
        d2_dtype = d2[c].dtype
        d1_ordered = get_ordered(d1_dtype)
        d1_categories = get_categories(d1_dtype)
        d2_categories = get_categories(d2_dtype)
        if d1_ordered:  # replace categories with their index
            s1, s2 = set(d1_categories), set(d2_categories)
            if s1 - s2 and s2 - s1:  # one-way imbalance is fine
                raise Exception(f"Ordered categorical columns differ in their categories on: {s1 - s2} vs {s2 - s1}")

            md = d1 if len(s2 - s1) == 0 else d2
            md_dtype = md[c].dtype
            md_categories = get_categories(md_dtype)
            mdict = dict(
                zip(
                    md_categories,
                    np.linspace(0, 2, len(md_categories)),
                )
            )
            d1[c] = d1[c].astype("object").replace(mdict)
            d2[c] = d2[c].astype("object").replace(mdict)
        else:  # Use one-hot encoding instead
            cs = list(set(d1[c].unique()) | set(d2[c].unique()))
            d1[c], d2[c] = pd.Categorical(d1[c], cs), pd.Categorical(d2[c], cs)
            # Use all but the first category as otherwise mahalanobis fails because of full colinearity
            d1 = replace_cat_with_dummies(d1, c, cs)
            d2 = replace_cat_with_dummies(d2, c, cs)

    # Use pseudoinverse in case we have collinear columns
    cov = np.cov(np.vstack([d1.values, d2.values]).T.astype("float"))
    cov += np.eye(len(cov)) * 1e-5  # Add a small amount of noise to avoid singular matrix
    pVI = np.linalg.pinv(cov).T
    dmat = cdist(d1, d2, "mahalanobis", VI=pVI)
    i1, i2 = linear_sum_assignment(dmat, maximize=False)
    ind1, ind2 = d1.index[i1], d2.index[i2]
    return list(ind1), list(ind2)


JSONStructure = TypeVar("JSONStructure", JSONScalar, JSONObject, JSONArray)


def replace_constants(
    d: JSONStructure,
    constants: Mapping[str, JSONValue] | None = None,
    inplace: bool = False,
    keep: bool = False,
) -> JSONStructure:
    """Recursively expand ``"constants"`` references inside annotation dicts.

    Args:
        d: Arbitrary nested structure containing optional ``"constants"`` blocks.
        constants: Pre-existing constant definitions to seed recursion with.
        inplace: Whether to mutate the provided structure.
        keep: Whether to preserve the constants key (for DataMeta).

    Returns:
        Structure with string references swapped for their constant values.
    """

    if not inplace:
        d = deepcopy(d)

    constants_map: Dict[str, JSONValue] = dict(constants or {})

    # Handle dict case
    if isinstance(d, dict):
        if "constants" in d and d["constants"] is not None:
            constants_map.update(d["constants"])  # type: ignore[arg-type]
            if not keep:
                del d["constants"]
        for k, v in d.items():
            if isinstance(v, str) and v in constants_map:
                d[k] = constants_map[v]
            elif isinstance(v, (dict, list)):
                d[k] = replace_constants(v, constants_map, inplace=True, keep=keep)
    # Handle list case
    elif isinstance(d, list):
        for i, v in enumerate(d):
            if isinstance(v, str) and v in constants_map:
                d[i] = constants_map[v]
            elif isinstance(v, (dict, list)):
                d[i] = replace_constants(v, constants_map, inplace=True, keep=keep)

    return d


def approx_str_match(
    frm: Iterable[str],
    to: Iterable[str],
    dist_fn: Callable[[str, str], float] | None = None,
    lower: bool = True,
) -> Dict[str, str]:
    """Return the minimum-distance mapping between two label collections.

    Args:
        frm: Source iterable of labels.
        to: Target iterable of labels.
        dist_fn: Optional distance function (defaults to Levenshtein distance).
        lower: Whether to compare strings case-insensitively.

    Returns:
        Dictionary mapping each item in ``frm`` to its best match in ``to``.
    """

    frm_list = list(frm)
    to_list = list(to)
    distance_fn = dist_fn or Levenshtein.distance  # type: ignore[assignment]
    if lower:
        original = distance_fn
        distance_fn = lambda x, y: original(x.lower(), y.lower())

    dmat = scipy.spatial.distance.cdist(
        np.array(frm_list)[:, None],
        np.array(to_list)[:, None],
        lambda x, y: distance_fn(x[0], y[0]),
    )
    idx_from, idx_to = scipy.optimize.linear_sum_assignment(dmat)
    return dict(zip([frm_list[i] for i in idx_from], [to_list[i] for i in idx_to]))


def index_encoder(z: object) -> list[Any]:
    """Convert pandas indices to JSON-serialisable lists.

    Args:
        z: Arbitrary object (ideally a pandas Index).

    Returns:
        List representation of ``z`` if it is an Index.

    Raises:
        TypeError: If ``z`` cannot be serialised.
    """

    if isinstance(z, pd.Index):
        return list(z)
    type_name = z.__class__.__name__
    raise TypeError(f"Object of type {type_name} is not serializable")


default_color = "lightgrey"  # Something that stands out so it is easy to notice a missing color


def to_alt_scale(
    scale: Mapping[str, str] | alt.Scale | None,
    order: Sequence[str] | None = None,
) -> alt.Scale | alt.utils.schemapi.UndefinedType:
    """Convert a mapping into an Altair scale (or None into alt.Undefined) while preserving category order.

    Preserving order matters because scale order overrides sort argument.
    """

    if scale is None:
        scale = alt.Undefined  # type: ignore[assignment]
    if isinstance(scale, dict):
        if order is None:
            order = list(scale.keys())
        # else: order = [ c for c in order if c in scale ]
        scale = alt.Scale(
            domain=list(order),
            range=[(scale[c] if c in scale else default_color) for c in order],
        )
    return scale


def multicol_to_vals_cats(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
    col_prefix: str | None = None,
    reverse_cols: Sequence[str] | None = None,
    reverse_suffixes: Sequence[str] | None = None,
    cat_order: Sequence[str] | None = None,
    vals_name: str = "vals",
    cats_name: str = "cats",
    inplace: bool = False,
) -> pd.DataFrame:
    """Pivot question variants spread across multiple columns into long form."""

    if not inplace:
        df = df.copy()
    if cols is None:
        if not col_prefix:
            raise ValueError("Either cols or col_prefix must be provided")
        cols = [c for c in df.columns if c.startswith(col_prefix)]
    reverse_cols = list(reverse_cols or [])

    if not reverse_cols and reverse_suffixes is not None:
        reverse_cols = list({c for c in cols for rs in reverse_suffixes if c.endswith(rs)})

    if len(reverse_cols) > 0:
        if cat_order is None:
            raise ValueError("cat_order must be provided when reverse columns are specified")
        remap = dict(zip(cat_order, reversed(cat_order)))
        df.loc[:, reverse_cols] = df.loc[:, reverse_cols].astype("object").replace(remap)

    tdf = df[cols]
    cinds = np.argmax(tdf.notna(), axis=1)
    df.loc[:, vals_name] = np.array(tdf)[range(len(tdf)), cinds]
    df.loc[:, cats_name] = np.array(tdf.columns)[cinds]
    return df


default_bidirectional_gradient = ["#c30d24", "#e67f6c", "#c3b6af", "#74b0ce", "#1770ab"]
redblue_gradient = ["#8D0E26", "#EA9379", "#F2EFEE", "#8FC1DC", "#134C85"]
greyscale_gradient = ["#444444", "#ffffff"]

# Grad is a list of colors


def gradient_to_discrete_color_scale(grad: Sequence[str], num_colors: int) -> list[str]:
    """Sample ``num_colors`` evenly spaced colours from a gradient definition."""

    cmap = mpc.LinearSegmentedColormap.from_list("grad", grad)
    return [mpc.to_hex(cmap(i)) for i in np.linspace(0, 1, num_colors)]


def gradient_subrange(
    grad: Sequence[str],
    num_colors: int,
    range: Sequence[float] = (-1, 1),
    bidirectional: bool = True,
) -> list[str]:
    """Extract a sub-range of a gradient while keeping perceptual spacing."""

    base = [-1, 1] if bidirectional else [0, 1]
    wr = (range[1] - range[0]) / (base[1] - base[0])
    nt = round(num_colors / wr)
    grad = gradient_to_discrete_color_scale(grad, nt)

    mi, ma = (
        round(nt * (range[0] - base[0]) / (base[1] - base[0])),
        round(nt * (range[1] - base[0]) / (base[1] - base[0])),
    )
    return grad[mi:ma]


def gradient_from_color(
    color: str,
    l_value: float = 0.3,
    n_points: int = 7,
    range: Sequence[float] = (0, 1),
) -> list[str]:
    """Derive a bi-directional gradient anchored at a single colour."""

    # Get hue and saturation for color (ignoring luminosity to make scales uniform on that)
    ch, cs, _ = hsluv.hex_to_hsluv(mpc.to_hex(color))

    max_l = 94  # Set max luminosity to be slightly below pure white
    l_diff = max_l - 100 * l_value  # Difference between max and min luminosity

    beg_s, end_s = (
        3 * cs * range[0],
        3 * cs * range[1],
    )  # As we use min(cs, s), this just desaturates on first 1/3 of the range
    beg_l, end_l = max_l - l_diff * range[0], max_l - l_diff * range[1]

    ls = np.linspace(0, 1, n_points)  # Create n_points steps in hsluv space
    return [hsluv.hsluv_to_hex((ch, min(cs, w * end_s + (1 - w) * beg_s), (w * end_l + (1 - w) * beg_l))) for w in ls]


def gradient_from_color_alt(
    color: str,
    l_value: float = 0.6,
    n_points: int = 7,
    range: Sequence[float] = (0, 1),
) -> list[str]:
    """Variant of ``_gradient_from_color`` tuned for lighter palettes."""

    # Get hue and saturation for color (ignoring luminosity to make scales uniform on that)
    ch, cs, cl = hsluv.hex_to_hsluv(mpc.to_hex(color))

    max_l = 94  # Set max luminosity to be slightly below pure white
    if cs < 50:
        l_value = 0.3  # For very washed out tones, make sure we have enough luminosity contrast
    l_diff = max_l - min(cl, 100 * l_value)  # Difference between max and min luminosity

    beg_s, end_s = (
        3 * cs * range[0],
        3 * cs * range[1],
    )  # As we use min(cs, s), this just desaturates on lower part of range
    beg_l, end_l = max_l - l_diff * range[0], max_l - l_diff * range[1]

    ls = np.linspace(0, 1, n_points)  # Create n_points steps in hsluv space
    return [hsluv.hsluv_to_hex((ch, min(cs, w * end_s + (1 - w) * beg_s), (w * end_l + (1 - w) * beg_l))) for w in ls]


def split_to_neg_neutral_pos(
    cats: Sequence[str],
    neutrals: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    """Partition Likert categories into negative/neutral/positive lists."""

    cats, mid = list(cats), len(cats) // 2
    neutrals_list = list(neutrals)  # Make a copy to avoid modifying input down the line
    if not neutrals_list:
        if len(cats) % 2 == 1:
            return cats[:mid], [cats[mid]], cats[mid + 1 :]
        else:
            return cats[:mid], [], cats[mid:]

    # Find a neutral that is not at start or end
    bi, ei = 0, 0
    while cats[bi] in neutrals_list:
        bi += 1
    while cats[-ei - 1] in neutrals_list:
        ei += 1
    cn = [c for c in neutrals_list if c in cats[bi : len(cats) - ei]]

    # If no such neutral, split evenly between positive and negative
    if not cn:
        posneg = [c for c in cats if c not in neutrals_list]
        pnmid = len(posneg) // 2
        if len(posneg) % 2 == 1:
            return posneg[:pnmid], neutrals_list + [posneg[pnmid]], posneg[pnmid + 1 :]
        else:
            return posneg[:pnmid], neutrals_list, posneg[pnmid:]
    else:  # Split around the first central neutral found
        ci = cats.index(cn[0])
        neg = [c for c in cats[:ci] if c not in neutrals_list]
        pos = [c for c in cats[ci:] if c not in neutrals_list]
        return neg, neutrals_list, pos


def is_datetime(col: pd.Series) -> bool:
    """Return True if a pandas Series behaves like a datetime column."""

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=UserWarning)
        result = pd.api.types.is_datetime64_any_dtype(col) or (
            col.dtype.name in ["str", "object"] and pd.to_datetime(col, errors="coerce").notna().any()
        )
        return bool(result)


def rel_wave_times(ws: Sequence[int], dts: Sequence[Any], dt0: pd.Timestamp | None = None) -> pd.Series:
    """Convert survey wave codes + dates into a relative time axis (months)."""

    df = pd.DataFrame({"wave": ws, "dt": pd.to_datetime(dts)})
    adf = df.groupby("wave")["dt"].median()
    if dt0 is None:
        dt0 = adf.max()  # use last wave date as the reference

    assert dt0 is not None, "dt0 must be set"
    w_to_time = dict(((adf - dt0).dt.days / 30).items())

    return pd.Series(df["wave"].replace(w_to_time), name="t")


def stable_rng(seed: int | str | bytes) -> np.random.Generator:
    """Return a platform-stable RNG using numpy's SFC64 bit generator."""

    return np.random.Generator(np.random.SFC64(seed))


def stable_draws(n: int, n_draws: int, uid: str | int) -> np.ndarray:
    """Generate deterministic draw assignments keyed by ``uid``."""

    # Initialize a random generator with a hash of uid
    bgen = np.random.SFC64(np.frombuffer(sha256(str(uid).encode("utf-8")).digest(), dtype="uint32"))
    gen = np.random.Generator(bgen)

    n_samples = int(math.ceil(n / n_draws))
    draws = np.tile(np.arange(n_draws), n_samples)[:n]
    return gen.permuted(draws)


def deterministic_draws(
    df: pd.DataFrame,
    n_draws: int,
    uid: str | int,
    n_total: int | None = None,
) -> pd.DataFrame:
    """Attach a deterministic ``draw`` column to ``df`` using ``stable_draws``."""

    if n_total is None:
        n_total = len(df)
    df.loc[:, "draw"] = pd.Series(stable_draws(n_total, n_draws, uid), index=np.arange(n_total))
    return df


def clean_kwargs(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to only those accepted by ``fn``."""

    aspec = inspect.getfullargspec(fn)
    result = {k: v for k, v in kwargs.items() if k in aspec.args} if aspec.varkw is None else kwargs
    return dict(result)


def call_kwsafe(fn: Callable[..., T], *args: object, **kwargs: object) -> T:
    """Call ``fn`` after trimming unsupported keyword arguments."""

    return fn(*args, **clean_kwargs(fn, kwargs))


def censor_dict(d: Mapping[str, Any], vs: Sequence[str]) -> Dict[str, Any]:
    """Return a dict copy without keys in ``vs``."""

    return {k: v for k, v in d.items() if k not in vs}


def cut_nice_labels(
    breaks: MutableSequence[float],
    mi: float = -np.inf,
    ma: float = np.inf,
    isint: bool = False,
    format: str = "",
    separator: str = " - ",
) -> tuple[list[float], list[str]]:
    """Build human-friendly interval labels (used by both pandas and polars)."""

    # Extend breaks if needed
    lopen, ropen = False, False
    if ma > breaks[-1]:
        breaks.append(ma + 1)
        ropen = True
    if mi < breaks[0]:
        breaks.insert(0, mi)
        lopen = True

    obreaks = breaks.copy()

    if isint:
        breaks = list(map(int, breaks))
        format = ""  # No need for decimal places if all integers
        breaks[-1] += 1  # to counter the -1 applied below

    tuples = [(breaks[i], breaks[i + 1] - (1 if isint else 0)) for i in range(len(breaks) - 1)]
    labels = [f"{t[0]:{format}}{separator}{t[1]:{format}}" if t[0] != t[1] else f"{t[0]:{format}}" for t in tuples]

    if lopen:
        labels[0] = f"<{breaks[1]:{format}}"
    if ropen:
        labels[-1] = f"{breaks[-2]:{format}}+"

    return obreaks, labels


def cut_nice(
    s: Sequence[float],
    breaks: MutableSequence[float],
    format: str = "",
    separator: str = " - ",
) -> pd.Categorical:
    """Wrapper around ``pd.cut`` that keeps prettier labels and inclusivity."""

    s_arr = np.array(s)
    mi, ma = s_arr.min(), s_arr.max()
    isint = np.issubdtype(s_arr.dtype, np.integer) or (s_arr % 1 == 0.0).all()
    breaks, labels = cut_nice_labels(breaks, mi, ma, isint, format, separator)
    return pd.cut(s_arr, breaks, right=False, labels=labels, ordered=False)


def rename_cats(df: pd.DataFrame, col: str, cat_map: Mapping[str, str]) -> None:
    """Rename categorical levels regardless of dtype quirks."""

    if df[col].dtype.name == "category":
        df[col] = df[col].cat.rename_categories(cat_map)
    else:
        df[col] = df[col].replace(cat_map)


def str_replace(s: pd.Series, d: Mapping[str, str]) -> pd.Series:
    """Apply a sequence of string replacements in order."""

    s = s.astype("object")
    for k, v in d.items():
        s = s.str.replace(k, v)
    return s


def merge_series(*lst: pd.Series | tuple[pd.Series, Sequence[Any]]) -> pd.Series:
    """Merge multiple series, preferentially taking non-null values."""

    s = lst[0].astype("object").copy()
    for t in lst[1:]:
        if isinstance(t, tuple):
            ns, whitelist = t
            inds = ~ns.isna() & ns.isin(whitelist)
            s.loc[inds] = ns[inds]
        else:
            ns = t
            s.loc[~ns.isna()] = ns[~ns.isna()]
    return s


def aggregate_multiselect(
    df: pd.DataFrame,
    prefix: str,
    out_prefix: str,
    na_vals: Sequence[Any] | None = None,
    colnames_as_values: bool = False,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """Collect multi-select responses stored as separate columns."""

    warn("This functionality is now built into create block.")
    cols = [c for c in df.columns if c.startswith(prefix)]
    na_vals = list(na_vals or [])
    dfc = df[cols].astype("object").replace(dict(zip(na_vals, [None] * len(na_vals))))

    if dfc.isna().sum().sum() == 0:
        raise ValueError(f"No na_vals found by _aggregate_multiselect in {prefix}")

    # Turn column names into the values - this is sometimes necessary
    # as values in col might be "mentioned"/"not mentioned"
    if colnames_as_values:
        dfc = dfc.copy()
        for c in dfc.columns:
            dfc.loc[~dfc[c].isna(), c] = c.removeprefix(prefix)

    lst = list(map(lambda row: [v for v in row if v is not None], dfc.values.tolist()))
    n_res = max(map(len, lst))
    columns = [f"{out_prefix}{i + 1}" for i in range(n_res)]
    result = pd.DataFrame(lst, columns=columns)
    if inplace:
        df[columns] = pd.DataFrame(lst)
        return None
    return result


def deaggregate_multiselect(df: pd.DataFrame, prefix: str, out_prefix: str = "") -> pd.DataFrame:
    """Expand multi-select columns into one-hot encoded booleans."""

    cols = [c for c in df.columns if c.startswith(prefix)]

    # Determine all categories
    ocols = set()
    for c in cols:
        ocols.update(df[c].dropna().unique())

    # Create a one-hot column for each
    for oc in ocols:
        df[out_prefix + oc] = (df[cols] == oc).any(axis=1)

    return df


def gb_in(df: pd.DataFrame, gb_cols: Sequence[str]) -> pd.core.groupby.generic.DataFrameGroupBy | pd.DataFrame:
    """Return ``df.groupby`` when ``gb_cols`` not empty; otherwise return ``df``."""

    # Convert to list for pandas groupby overload matching
    return df.groupby(list(gb_cols), observed=False) if len(gb_cols) > 0 else df  # type: ignore[call-overload]


def gb_in_apply(
    df: pd.DataFrame,
    gb_cols: Sequence[str],
    fn: Callable[..., pd.DataFrame | pd.Series],
    cols: Sequence[str] | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """Groupby apply if needed - similar to gb_in but for apply.

    Apply ``fn`` either to the whole frame or grouped subsets.
    """

    if cols is None:
        cols = list(df.columns)
    if len(gb_cols) == 0:
        res = fn(df[cols], **kwargs)
        if isinstance(res, pd.Series):
            res = pd.DataFrame(res).T
    else:
        # Convert to list for pandas groupby overload matching
        res = df.groupby(list(gb_cols), observed=False)[cols].apply(fn, **kwargs)  # type: ignore[call-overload]
    return res


def stk_defaultdict(dv: object) -> defaultdict[str, Any]:
    """Return a ``defaultdict`` whose fallback value can be configured."""

    if not isinstance(dv, dict):
        dv = {"default": dv}
    return defaultdict(lambda: dv["default"], dv)


def cached_fn(fn: Callable[[Any], T]) -> Callable[[Any], T]:
    """Memoise a single-argument function."""

    cache: Dict[Any, T] = {}

    def _cf(x: object) -> T:
        if x not in cache:
            cache[x] = fn(x)
        return cache[x]

    return _cf


def scores_to_ordinal_rankings(
    df: pd.DataFrame,
    cols: Sequence[str] | str,
    name: str,
    prefix: str = "",
) -> pd.DataFrame:
    """Convert score columns into ordinal rankings with tie handling."""

    # If cols is a string, treat it as a prefix and find all columns that start with it
    if isinstance(cols, str):
        prefix = prefix or cols
        cols = [c for c in df.columns if c.startswith(cols)]

    sinds = np.argsort(-df[cols].values, axis=1)

    rmat = df[cols].rank(method="max", ascending=False, axis=1).values
    # rmat = np.concatenate([rmat,np.full((len(rmat),1),0)],axis=1)
    rvals = rmat[np.tile(np.arange(len(rmat)), (len(rmat[0]), 1)).T, sinds]

    names_a, ties_a = [], []
    for cns, rs in zip(np.array(cols)[sinds], rvals):
        if np.isnan(rs[-1]):
            missing_idx = np.where(np.isnan(rs))[0][0]
            cns, rs = cns[:missing_idx], rs[:missing_idx]
        ties = (rs - np.arange(len(rs)) - 1).astype(int)
        names_a.append([c[len(prefix) :] for c in cns])
        ties_a.append(list(ties))
    df[f"{name}_orank"] = names_a
    df[f"{name}_ties"] = ties_a
    return df


class dict_cache(OrderedDict):
    """LRU-ish OrderedDict with a configurable size limit."""

    def __init__(self, size: int = 10, *args: object, **kwargs: object) -> None:
        """Create a cache with the provided max ``size``."""

        self.size = size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: object, value: object) -> None:
        """Drop the oldest item when the capacity is reached."""

        if len(self) >= self.size:
            old = next(iter(self))
            super().__delitem__(old)

        super().__setitem__(key, value)

    def __getitem__(self, key: object) -> object:
        """Mark the entry as recently used before returning it."""

        res = super().__getitem__(key)
        super().move_to_end(key)  # Move it up to indicate recent use
        return res


def get_size(obj: object, seen: set[int] | None = None) -> int:
    """Recursively approximate the memory footprint of ``obj``."""

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in cast(Iterable[Any], obj)])
    return size


def get_categories(dtype: object) -> list[object]:
    """Safely get categories from a pandas dtype.

    Args:
        dtype: A pandas dtype object (may be CategoricalDtype or other).

    Returns:
        List of categories if dtype has categories attribute, empty list otherwise.
    """
    if hasattr(dtype, "categories"):
        categories = dtype.categories
        if categories is not None:
            return list(categories)
    return []


def get_ordered(dtype: object) -> bool:
    """Safely get ordered status from a pandas dtype.

    Args:
        dtype: A pandas dtype object (may be CategoricalDtype or other).

    Returns:
        True if dtype is ordered, False otherwise.
    """
    if hasattr(dtype, "ordered"):
        return bool(dtype.ordered)
    return False


def escape_vega_label(label: str) -> str:
    """Escape characters that confuse Vega Lite (Altair)."""

    return label.replace(".", "․").replace("[", "［").replace("]", "］")


def unescape_vega_label(label: str) -> str:
    """Undo ``escape_vega_label``."""

    return label.replace("․", ".").replace("［", "[").replace("］", "]")


def read_json(fname: str) -> JSONValue:
    """Load JSON file with extension sanity checks."""

    if ".json" not in fname:
        raise FileNotFoundError(f"Expecting {fname} to have a .json extension")
    with open(fname, "r") as jf:
        meta = json.load(jf)
    return meta


def read_yaml(model_desc_file: str) -> JSONValue:
    """Load YAML file with extension sanity checks."""

    if ".yaml" not in model_desc_file:
        raise FileNotFoundError(f"Expecting {model_desc_file} to have a .yaml extension")
    with open(model_desc_file) as stream:
        try:
            yaml_desc = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_desc


# Altair default configuration for plot styling
altair_default_config = {
    "font": '"Source Sans Pro", sans-serif',
    "background": "#ffffff",
    "fieldTitle": "verbal",
    "autosize": {"type": "fit", "contains": "padding"},
    "title": {
        "align": "left",
        "anchor": "start",
        "color": "#31333F",
        "titleFontStyle": "normal",
        "fontWeight": 600,
        "fontSize": 16,
        "orient": "top",
        "offset": 26,
    },
    "header": {
        "titleFontWeight": 400,
        "titleFontSize": 16,
        "titleColor": "#808495",
        "titleFontStyle": "normal",
        "labelFontSize": 12,
        "labelFontWeight": 400,
        "labelColor": "#808495",
        "labelFontStyle": "normal",
    },
    "axis": {
        "labelFontSize": 12,
        "labelFontWeight": 400,
        "labelColor": "#808495",
        "labelFontStyle": "normal",
        "titleFontWeight": 400,
        "titleFontSize": 14,
        "titleColor": "#808495",
        "titleFontStyle": "normal",
        "ticks": False,
        "gridColor": "#e6eaf1",
        "domain": False,
        "domainWidth": 1,
        "domainColor": "#e6eaf1",
        "labelFlush": True,
        "labelFlushOffset": 1,
        "labelBound": False,
        "labelLimit": 100,
        "titlePadding": 16,
        "labelPadding": 16,
        "labelSeparation": 4,
        "labelOverlap": True,
    },
    "legend": {
        "labelFontSize": 14,
        "labelFontWeight": 400,
        "labelColor": "#808495",
        "titleFontSize": 14,
        "titleFontWeight": 400,
        "titleFontStyle": "normal",
        "titleColor": "#808495",
        "titlePadding": 5,
        "labelPadding": 16,
        "columnPadding": 8,
        "rowPadding": 4,
        "padding": 7,
        "symbolStrokeWidth": 4,
    },
    "range": {
        "category": [
            "#0068c9",
            "#83c9ff",
            "#ff2b2b",
            "#ffabab",
            "#29b09d",
            "#7defa1",
            "#ff8700",
            "#ffd16a",
            "#6d3fc0",
            "#d5dae5",
        ],
        "diverging": [
            "#7d353b",
            "#bd4043",
            "#ff4b4b",
            "#ff8c8c",
            "#ffc7c7",
            "#a6dcff",
            "#60b4ff",
            "#1c83e1",
            "#0054a3",
            "#004280",
        ],
        "ramp": [
            "#e4f5ff",
            "#c7ebff",
            "#a6dcff",
            "#83c9ff",
            "#60b4ff",
            "#3d9df3",
            "#1c83e1",
            "#0068c9",
            "#0054a3",
            "#004280",
        ],
        "heatmap": [
            "#e4f5ff",
            "#c7ebff",
            "#a6dcff",
            "#83c9ff",
            "#60b4ff",
            "#3d9df3",
            "#1c83e1",
            "#0068c9",
            "#0054a3",
            "#004280",
        ],
    },
    "view": {
        "columns": 1,
        "strokeWidth": 0,
        "stroke": "transparent",
        "continuousHeight": 350,
        "continuousWidth": 400,
        "discreteHeight": {"step": 20},
    },
    "concat": {"columns": 1},
    "facet": {"columns": 1},
    "mark": {"tooltip": True, "color": "#0068c9"},
    "bar": {"binSpacing": 4, "discreteBandSize": {"band": 0.85}},
    "axisDiscrete": {"grid": False},
    "axisXPoint": {"grid": False},
    "axisTemporal": {"grid": False},
    "axisXBand": {"grid": False},
}

# HTML template for embedding Altair plots
html_template = """
<!DOCTYPE html>
<html>
<head></head>
<body>
  <div id="UID">SUBDIVS</div>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <script type="text/javascript">
    UID_delta = 0
    function draw_plot() {
        width = document.getElementById("UID").parentElement.clientWidth;
        var specs = %s;
        var opt = {"renderer": "canvas", "actions": false};
        specs.forEach(function(spec,i){ vegaEmbed("#UID-"+i, spec, opt); });
    };
    draw_plot();
    // This is a hack to fix facet plot width issues
    setTimeout(function() {
        wc = %s;
        wp = document.getElementById("UID").offsetWidth;
        UID_delta = wp-wc;
        if (UID_delta!=0) draw_plot();
    }, 5);
    %s
   </script>
</body>
</html>
"""


def plot_matrix_html(
    pmat: AltairChart | list[list[AltairChart]] | None,
    uid: str = "viz",
    width: int | None = None,
    responsive: bool = True,
) -> str | None:
    """Generate HTML for a matrix of Altair plots.

    Args:
        pmat: Matrix of plots (list of lists) or single plot, or None.
        uid: Unique identifier for HTML elements.
        width: Optional plot width in pixels.
        responsive: Whether plots should be responsive to container width.

    Returns:
        HTML string containing the plots, or None if pmat is None.
    """
    if not pmat:
        return None
    if not isinstance(pmat, list):
        pmat = [[pmat]]

    # Sanitize uid so it can be used as a variable name in JavaScript
    # - replace all whitespace and non-alphanumeric characters with underscores
    uid = re.sub(r"\W+", "_", str(uid))

    template = html_template.replace("UID", uid)

    rstring = "XYZresponsiveXZY"  # Something we can replace easy
    specs, ncols = [], len(pmat[0])
    for i, p in enumerate(pmat):
        for j, pp in enumerate(p):
            pdict = json.loads(pp.to_json())
            pdict["autosize"] = {"type": "fit-x", "contains": "padding"}
            pdict["config"] = altair_default_config

            if responsive:
                cwidth = pdict["spec"]["width"] if "spec" in pdict else pdict["width"]
                repl = f"(width-{uid}_delta/{ncols})/{width / cwidth}"
                if "spec" in pdict:
                    pdict["spec"]["width"] = rstring
                else:
                    pdict["width"] = rstring
                pjson = json.dumps(pdict).replace(f'"{rstring}"', repl)
            else:
                pjson = json.dumps(pdict)
            specs.append(pjson)

    if responsive:
        goal_width = f'document.getElementById("{uid}").parentElement.clientWidth'
        resp_code = 'window.addEventListener("resize", draw_plot);'
    else:
        goal_width, resp_code = str(width), ""

    html = template % (f"[{','.join(specs)}]", goal_width, resp_code)

    # Add subdivs after the plots - otherwise width% needs complex escaping
    subdivs = "".join(
        [f'<div id="{uid}-{i}" style="width: {0.99 / ncols:.3%}"></div>' for i in range(sum(map(len, pmat)))]
    )
    html = html.replace("SUBDIVS", subdivs)

    if responsive:
        html = html.replace(f'"{rstring}"', repl)
    return html
