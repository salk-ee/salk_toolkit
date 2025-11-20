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
    "warn",
    "default_color",
    "default_bidirectional_gradient",
    "redblue_gradient",
    "greyscale_gradient",
    "factorize_w_codes",
    "batch",
    "loc2iloc",
    "match_sum_round",
    "min_diff",
    "continify",
    "replace_cat_with_dummies",
    "match_data",
    "replace_constants",
    "approx_str_match",
    "index_encoder",
    "to_alt_scale",
    "multicol_to_vals_cats",
    "gradient_to_discrete_color_scale",
    "gradient_subrange",
    "gradient_from_color",
    "gradient_from_color_alt",
    "split_to_neg_neutral_pos",
    "is_datetime",
    "rel_wave_times",
    "stable_rng",
    "stable_draws",
    "deterministic_draws",
    "clean_kwargs",
    "call_kwsafe",
    "censor_dict",
    "cut_nice_labels",
    "cut_nice",
    "rename_cats",
    "str_replace",
    "merge_series",
    "aggregate_multiselect",
    "deaggregate_multiselect",
    "gb_in",
    "gb_in_apply",
    "stk_defaultdict",
    "cached_fn",
    "scores_to_ordinal_rankings",
    "dict_cache",
    "get_size",
    "escape_vega_label",
    "unescape_vega_label",
    "read_json",
    "read_yaml",
]

import json
import warnings
import math
import inspect
import sys
import yaml
from collections import defaultdict, OrderedDict
from copy import deepcopy
from hashlib import sha256
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableSequence,
    Sequence,
    TypeVar,
)


import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import altair as alt
import matplotlib.colors as mpc
import hsluv

import Levenshtein

pd.set_option("future.no_silent_downcasting", True)


JSONValue = str | int | float | bool | None | dict[str, "JSONValue"] | list["JSONValue"]


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

    s = np.array(s)
    fs = np.floor(s)
    diff = round(s.sum() - fs.sum())
    residues = np.argsort(-(s % 1))[:diff]
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
        cols: Columns to align on. Must exist in both frames.

    Returns:
        Tuple of row indices describing optimal matches for ``data1`` and ``data2``.
    """

    d1 = data1[cols].copy().dropna()
    d2 = data2[cols].copy().dropna()

    if len(d1) == 0 or len(d2) == 0:
        return [], []

    ccols = [c for c in cols if d1[c].dtype.name == "category"]
    for c in ccols:
        if d1[c].dtype.ordered:  # replace categories with their index
            s1, s2 = set(d1[c].dtype.categories), set(d2[c].dtype.categories)
            if s1 - s2 and s2 - s1:  # one-way imbalance is fine
                raise Exception(f"Ordered categorical columns differ in their categories on: {s1 - s2} vs {s2 - s1}")

            md = d1 if len(s2 - s1) == 0 else d2
            mdict = dict(
                zip(
                    md[c].dtype.categories,
                    np.linspace(0, 2, len(md[c].dtype.categories)),
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
    return ind1, ind2


def replace_constants(
    d: JSONValue | MutableSequence[JSONValue] | MutableSequence[MutableSequence[JSONValue]],
    constants: Mapping[str, JSONValue] | None = None,
    inplace: bool = False,
) -> JSONValue | MutableSequence[JSONValue] | MutableSequence[MutableSequence[JSONValue]]:
    """Recursively expand ``"constants"`` references inside annotation dicts.

    Args:
        d: Arbitrary nested structure containing optional ``"constants"`` blocks.
        constants: Pre-existing constant definitions to seed recursion with.
        inplace: Whether to mutate the provided structure.

    Returns:
        Structure with string references swapped for their constant values.
    """

    if not inplace:
        d = deepcopy(d)

    constants_map: Dict[str, JSONValue] = dict(constants or {})
    if isinstance(d, dict) and "constants" in d:
        constants_map.update(d["constants"])
        del d["constants"]

    iterator = d.items() if isinstance(d, dict) else enumerate(d)
    for k, v in iterator:
        if isinstance(v, str) and v in constants_map:
            d[k] = constants_map[v]
        elif isinstance(v, (dict, list)):
            d[k] = replace_constants(v, constants_map, inplace=True)

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
    distance_fn = dist_fn or Levenshtein.distance
    if lower:
        original = distance_fn

        def distance_fn(x: str, y: str) -> float:  # type: ignore[redefined-outer-name]
            return original(x.lower(), y.lower())

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

# Helper function to turn a dictionary into an Altair scale (or None into alt.Undefined)
# Also: preserving order matters because scale order overrides sort argument


def to_alt_scale(
    scale: Mapping[str, str] | alt.Scale | None,
    order: Sequence[str] | None = None,
) -> alt.Scale | alt.utils.schemapi.UndefinedType:
    """Convert a mapping into an Altair scale while preserving category order."""

    if scale is None:
        scale = alt.Undefined
    if isinstance(scale, dict):
        if order is None:
            order = scale.keys()
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
    """Variant of ``gradient_from_color`` tuned for lighter palettes."""

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
    neutrals = neutrals.copy()  # Make a copy to avoid modifying input down the line
    if not neutrals:
        if len(cats) % 2 == 1:
            return cats[:mid], [cats[mid]], cats[mid + 1 :]
        else:
            return cats[:mid], [], cats[mid:]

    # Find a neutral that is not at start or end
    bi, ei = 0, 0
    while cats[bi] in neutrals:
        bi += 1
    while cats[-ei - 1] in neutrals:
        ei += 1
    cn = [c for c in neutrals if c in cats[bi : len(cats) - ei]]

    # If no such neutral, split evenly between positive and negative
    if not cn:
        posneg = [c for c in cats if c not in neutrals]
        pnmid = len(posneg) // 2
        if len(posneg) % 2 == 1:
            return posneg[:pnmid], neutrals + [posneg[pnmid]], posneg[pnmid + 1 :]
        else:
            return posneg[:pnmid], neutrals, posneg[pnmid:]
    else:  # Split around the first central neutral found
        ci = cats.index(cn[0])
        neg = [c for c in cats[:ci] if c not in neutrals]
        pos = [c for c in cats[ci:] if c not in neutrals]
        return neg, neutrals, pos


def is_datetime(col: pd.Series) -> bool:
    """Return True if a pandas Series behaves like a datetime column."""

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=UserWarning)
        return pd.api.types.is_datetime64_any_dtype(col) or (
            col.dtype.name in ["str", "object"] and pd.to_datetime(col, errors="coerce").notna().any()
        )


def rel_wave_times(ws: Sequence[int], dts: Sequence[Any], dt0: pd.Timestamp | None = None) -> pd.Series:
    """Convert survey wave codes + dates into a relative time axis (months)."""

    df = pd.DataFrame({"wave": ws, "dt": pd.to_datetime(dts)})
    adf = df.groupby("wave")["dt"].median()
    if dt0 is None:
        dt0 = adf.max()  # use last wave date as the reference

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
    return {k: v for k, v in kwargs.items() if k in aspec.args} if aspec.varkw is None else kwargs


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


# A nicer behaving wrapper around pd.cut


def cut_nice(
    s: Sequence[float],
    breaks: MutableSequence[float],
    format: str = "",
    separator: str = " - ",
) -> pd.Categorical:
    """Wrapper around ``pd.cut`` that keeps prettier labels and inclusivity."""

    s = np.array(s)
    mi, ma = s.min(), s.max()
    isint = np.issubdtype(s.dtype, np.integer) or (s % 1 == 0.0).all()
    breaks, labels = cut_nice_labels(breaks, mi, ma, isint, format, separator)
    return pd.cut(s, breaks, right=False, labels=labels, ordered=False)


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
        raise ValueError(f"No na_vals found by aggregate_multiselect in {prefix}")

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


def gb_in(df: pd.DataFrame, gb_cols: Sequence[str]) -> pd.core.groupby.generic.DataFrameGroupBy | pd.DataFrame:
    """Return ``df.groupby`` when ``gb_cols`` not empty; otherwise return ``df``."""

    return df.groupby(gb_cols, observed=False) if len(gb_cols) > 0 else df


# Groupby apply if needed - similar to gb_in but for apply


def gb_in_apply(
    df: pd.DataFrame,
    gb_cols: Sequence[str],
    fn: Callable[..., pd.DataFrame | pd.Series],
    cols: Sequence[str] | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """Apply ``fn`` either to the whole frame or grouped subsets."""

    if cols is None:
        cols = list(df.columns)
    if len(gb_cols) == 0:
        res = fn(df[cols], **kwargs)
        if isinstance(res, pd.Series):
            res = pd.DataFrame(res).T
    else:
        res = df.groupby(gb_cols, observed=False)[cols].apply(fn, **kwargs)
    return res


def stk_defaultdict(dv: object) -> defaultdict[str, Any]:
    """Return a ``defaultdict`` whose fallback value can be configured."""

    if not isinstance(dv, dict):
        dv = {"default": dv}
    return defaultdict(lambda: dv["default"], dv)


def cached_fn(fn: Callable[[Any], T]) -> Callable[[Any], T]:
    """Memoise a single-argument function."""

    cache: Dict[Any, T] = {}

    def cf(x: object) -> T:
        if x not in cache:
            cache[x] = fn(x)
        return cache[x]

    return cf


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
        size += sum([get_size(i, seen) for i in obj])
    return size


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
