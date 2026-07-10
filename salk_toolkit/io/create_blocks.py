"""Specialized block processing (topk/maxdiff/onehot): expand grouped raw columns
into derived blocks via the block stages:

1. Match: identify candidate columns from the dataframe.
2. Explode: fan out regex-matched columns into subgroup siblings.
3. Pre-translate: map raw cell values using scale.translate.
4. Transform: dispatch to the type-specific transform + output block builder.
5. Post-translate: map output cells and categories using scale.translate_after.

Plain blocks are processed column-by-column by :mod:`salk_toolkit.io.pipeline`;
only the specialized block types come through here.
"""

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from copy import deepcopy
from typing import Any, cast

import numpy as np
import pandas as pd

from salk_toolkit.utils import warn
from salk_toolkit.validation import (
    ColumnBlockMeta,
    ColumnMeta,
    MaxDiffBlock,
    OneHotBlock,
    TopKBlock,
    soft_validate,
)

from salk_toolkit.io.core import _is_series_of_lists


def _throw_vals_left(df: pd.DataFrame) -> None:
    """Move all NaN values to the right in each row (in-place)."""
    # Helper fun to move inplace all nan values to right.
    df.iloc[:, :] = df.apply(lambda row: sorted(row, key=pd.isna), axis=1).to_list()


def _check_topk_na_vals_after_replace(sdf: pd.DataFrame, *, block_name: str) -> None:
    """Require na_vals to match somewhere overall; warn on columns with no NA after replace."""
    if not sdf.isna().to_numpy().any():
        raise ValueError(f"No na_vals found in topk block {block_name!r}")

    cols_no_na = [c for c in sdf.columns if not sdf[c].isna().any()]
    if cols_no_na:
        listed = ", ".join(map(str, cols_no_na))
        warn(
            f"Topk block {block_name!r}: no NA in column(s) after na_vals replace ({listed})",
            UserWarning,
        )


def _apply_pre_transform_translate(block: ColumnBlockMeta, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Stage 3: map raw cell values through scale.translate before the transform runs."""
    if block.scale is None or not block.scale.translate:
        return df
    # TODO: consider str-coercing cells before .replace() — CSV round-trips turn
    # integer-string index cells into int64, so .replace({"1": "Economy"}) silently
    # no-ops. See follow-up from 2026-04-23 block-processing refactor.
    translate = cast("dict[object, object]", dict(block.scale.translate))
    df = df.copy()

    def _map_list(lst: object) -> object:
        if lst is None:
            return None
        if isinstance(lst, float) and pd.isna(lst):
            return None
        return [translate.get(x, x) for x in cast(Iterable[object], lst)]

    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if isinstance(block, MaxDiffBlock) and _is_series_of_lists(s):
            df[c] = s.map(_map_list)
        else:
            df[c] = s.astype("object").replace(translate)
    return df


def _apply_post_transform_translate(
    block: ColumnBlockMeta,
    sdf: pd.DataFrame,
    meta: ColumnBlockMeta,
) -> tuple[pd.DataFrame, ColumnBlockMeta]:
    """Universal stage 5. If `scale.translate_after` is set on the *output* meta:
    - scalar cells: element-wise `.replace` via translate_after dict per column.
    - list-valued cells (future MaxDiff set-column support): map inside the list.
    - Categorical columns: rebuild with translated categories in the same order.
    - Rewrite `meta.scale.categories` to the translated values.
    Blocks with no scale or no translate_after pass through unchanged.
    MaxDiff blocks reject translate_after at pydantic validation (the
    `_reject_translate_after` model validator), so this branch is a defensive
    safety net — unreachable in practice."""
    scale = meta.scale
    if scale is None or not scale.translate_after:
        return sdf, meta
    if isinstance(block, MaxDiffBlock):
        raise ValueError(
            f"MaxDiffBlock {block.name!r}: scale.translate_after is not supported on maxdiff; "
            f"use scale.translate (pre-transform) instead."
        )
    t = dict(scale.translate_after)

    def _map_scalar(v: object) -> object:
        if isinstance(v, str):
            return t.get(v, v)
        return v

    def _map_list(lst: object, _t: dict[str, str] = t) -> object:
        if lst is None:
            return None
        if isinstance(lst, float) and pd.isna(lst):
            return None
        return [_t.get(x, x) if isinstance(x, str) else x for x in cast(Iterable[object], lst)]

    for col in sdf.columns:
        s = sdf[col]
        if _is_series_of_lists(s):
            sdf[col] = s.map(_map_list)
        elif isinstance(s.dtype, pd.CategoricalDtype):
            new_cats = [t.get(c, c) if isinstance(c, str) else c for c in s.cat.categories]
            sdf[col] = s.cat.rename_categories(new_cats)
        else:
            sdf[col] = s.map(_map_scalar)

    scale_dict = scale.model_dump(mode="python")
    if not scale_dict.get("categories") or scale_dict.get("categories") == "infer":
        new_categories = list(dict.fromkeys(t.values()))
    else:
        new_categories = [t.get(c, c) if isinstance(c, str) else c for c in scale_dict["categories"]]
    scale_dict["categories"] = new_categories
    new_scale = type(scale).model_validate(scale_dict)
    # Propagate the post-translate categories onto each column too, mirroring what
    # `merge_scale_with_columns` does at block-validation time. Columns whose
    # categories were inherited from the pre-translate scale (`"infer"` or the
    # pre-translate list) get re-synced to the new categories.
    updated_columns: dict[str, ColumnMeta] = {}
    pre_cats = scale.categories
    for cn, col_meta in meta.columns.items():
        col_update: dict[str, object] = {}
        if col_meta.categories == "infer" or col_meta.categories == pre_cats:
            col_update["categories"] = new_categories
        if col_meta.translate_after == scale.translate_after:
            col_update["translate_after"] = new_scale.translate_after
        updated_columns[cn] = col_meta.model_copy(update=col_update) if col_update else col_meta
    meta_out = meta.model_copy(update={"scale": new_scale, "columns": updated_columns})
    return sdf, meta_out


def _process_block(
    block: TopKBlock | MaxDiffBlock | OneHotBlock, df: pd.DataFrame
) -> Iterator[tuple[pd.DataFrame, ColumnBlockMeta]]:
    """Driver for specialized blocks: explode into siblings, then run the
    pre-translate -> transform -> post-translate stages on each."""
    siblings: list[ColumnBlockMeta]
    if isinstance(block, MaxDiffBlock) and block.from_columns is None:
        siblings = [_apply_role_resolution(block, block, df)]
    elif isinstance(block, OneHotBlock):
        siblings = [block]
    else:
        siblings = _subgroup_explode(block, df)

    for sib in siblings:
        cols = sib.input_df_columns(df)
        df_t = _apply_pre_transform_translate(sib, df, cols)
        sdf, meta = _apply_transform(sib, df_t, source_block=block)
        sdf, meta = _apply_post_transform_translate(sib, sdf, meta)
        yield sdf, meta


def _apply_transform(
    block: ColumnBlockMeta,
    df: pd.DataFrame,
    *,
    source_block: ColumnBlockMeta,
) -> tuple[pd.DataFrame, ColumnBlockMeta]:
    """Dispatch to the per-type transform."""
    if isinstance(block, TopKBlock):
        assert isinstance(source_block, TopKBlock)
        source_pattern = source_block.from_columns if isinstance(source_block.from_columns, str) else None
        return _topk_apply_transform(block, df, source_pattern=source_pattern, source_block=source_block)
    if isinstance(block, MaxDiffBlock):
        assert isinstance(source_block, MaxDiffBlock)
        cs = _get_subgroup_config(source_block.choice_sets, block.name, source_block.name)
        return _maxdiff_apply_transform(block, df, cs, source_block=source_block)
    if isinstance(block, OneHotBlock):
        assert isinstance(source_block, OneHotBlock)
        return _onehot_apply_transform(block, df, source_block.choices)
    raise TypeError(f"Unsupported block type for _apply_transform: {type(block)}")


def _get_subgroup_config(value: object, sibling_name: str, source_name: str) -> object:
    """Extract a sibling-specific configuration from a parent field.

    A parent field (like `choice_sets`) can be either:
    1. A 'flat' structure (list or dict) that is shared by all siblings.
    2. A 'keyed' dictionary where keys correspond to subgroup labels (e.g.,
       'economics', 'politics') and values are the sibling-specific configs.

    This helper extracts the correct config based on the sibling's name suffix.

    Args:
        value: The configuration value from the parent block.
        sibling_name: The name of the narrowed sibling block.
        source_name: The name of the original parent block.

    Returns:
        The configuration (flat or picked from the keyed dict) for this sibling.
    """
    if value is None:
        return None

    sibling_label = sibling_name.removeprefix(source_name).lstrip("_")
    is_keyed = isinstance(value, dict) and len(value) > 0 and all(isinstance(v, (list, dict)) for v in value.values())

    if not sibling_label:
        if is_keyed:
            raise ValueError(f"Block {source_name!r}: single sibling but field is keyed; expected flat")
        return value

    if not is_keyed:
        raise ValueError(f"Block {source_name!r}: multiple siblings but field is flat; expected dict keyed by label")
    keyed = cast(dict[str, object], value)
    if sibling_label not in keyed:
        raise ValueError(
            f"Block {source_name!r}: sibling {sibling_label!r} missing from field keys {list(keyed.keys())}"
        )
    return keyed[sibling_label]


def _match_columns(block: ColumnBlockMeta, df: pd.DataFrame) -> list[str]:
    """Stage 1: resolve `from_columns` to the concrete df columns it selects, raising if
    none match. Uses the base `from_columns` resolver explicitly, so it stays correct even
    for MaxDiffBlock (whose `input_df_columns` override means best/worst/set, not this)."""
    cols = ColumnBlockMeta.input_df_columns(block, df)
    if not cols:
        raise ValueError(f"No columns matched for block {block.name!r} (from_columns={block.from_columns!r})")
    return cols


def _block_scale_dict(block: ColumnBlockMeta) -> dict[str, Any]:
    """A deep-copied plain dict of the block's scale (empty dict when no scale),
    safe to mutate while building an output block."""
    return deepcopy(block.scale.model_dump(mode="python") if block.scale else {})


def _narrow_sibling(block: ColumnBlockMeta, cols: list[str], *, label_suffix: str) -> ColumnBlockMeta:
    new_name = block.name if not label_suffix else f"{block.name}_{label_suffix}"
    return block.model_copy(
        update={
            "name": new_name,
            "from_columns": cols,
            "subgroup_labels": None,
        }
    )


def _subgroup_explode(block: ColumnBlockMeta, df: pd.DataFrame) -> list[ColumnBlockMeta]:
    """Stage 2: fan out regex-matched columns into subgroup siblings.

    If from_columns is a regex with capture groups, this function identifies every
    unique combination of captured values and returns a list of 'narrowed' siblings.
    """
    matched_cols = _match_columns(block, df)
    pattern = block.from_columns
    if not isinstance(pattern, str):
        siblings = [_narrow_sibling(block, matched_cols, label_suffix="")]
        return [_apply_role_resolution(s, block, df) for s in siblings]

    regex = re.compile(pattern)
    first = regex.match(matched_cols[0])
    assert first is not None
    n_groups = len(first.groups())

    # TOPK-specific: skip one group for sibling identity if aggregating.
    # Otherwise, use all groups.
    agg_pos = None
    if isinstance(block, TopKBlock):
        agg_idx = block.agg_index
        agg_pos = agg_idx - 1 if agg_idx > 0 else agg_idx
        if agg_pos < 0:
            agg_pos = n_groups + agg_pos

        if not (0 <= agg_pos < n_groups):
            raise ValueError(f"Block {block.name!r}: agg_index={agg_idx} out of range for {n_groups} capture group(s)")

    non_agg_positions = [i for i in range(n_groups) if i != agg_pos] if agg_pos is not None else list(range(n_groups))

    if not non_agg_positions:
        siblings = [_narrow_sibling(block, matched_cols, label_suffix="")]
        return [_apply_role_resolution(s, block, df) for s in siblings]

    def _key(col: str) -> tuple[str, ...]:
        m = regex.match(col)
        assert m is not None
        g = m.groups()
        return tuple(g[i] for i in non_agg_positions)

    sibling_cols: dict[tuple[str, ...], list[str]] = {}
    for c in matched_cols:
        sibling_cols.setdefault(_key(c), []).append(c)

    labels = block.subgroup_labels or {}

    def _label(key: tuple[str, ...]) -> str:
        parts = []
        for val, pos in zip(key, non_agg_positions, strict=True):
            parts.append(str(labels.get(str(pos + 1), {}).get(val, val)))
        return "_".join(parts)

    return [
        _apply_role_resolution(_narrow_sibling(block, cols, label_suffix=_label(key)), block, df)
        for key, cols in sibling_cols.items()
    ]


def _apply_role_resolution(sib: ColumnBlockMeta, source: ColumnBlockMeta, df: pd.DataFrame) -> ColumnBlockMeta:
    """Apply per-type role-column resolution. Label is derived from sibling vs source name."""
    sib_label = sib.name.removeprefix(source.name).lstrip("_")
    updates = sib.resolve_role_columns(df, sib_label)
    return sib.model_copy(update=updates) if updates else sib


def _topk_apply_transform(
    block: TopKBlock,
    df: pd.DataFrame,
    *,
    source_pattern: str | None,
    source_block: TopKBlock,
) -> tuple[pd.DataFrame, TopKBlock]:
    """Dispatch TopK transformation based on input_format."""
    fmt = block.input_format
    if fmt in ("leftpacked", "ranked_leftpack"):
        # No transform needed, just extract columns
        return _topk_transform_passthrough(block, df, source_block=source_block)
    if fmt == "onehot":
        return _topk_transform_onehot(block, df, source_pattern=source_pattern, source_block=source_block)
    if fmt == "ranked_onehot":
        raise NotImplementedError("ranked_onehot transform not yet implemented")
    raise ValueError(f"unknown TopK input_format: {fmt!r}")


def _topk_transform_passthrough(
    block: TopKBlock,
    df: pd.DataFrame,
    *,
    source_block: TopKBlock,
) -> tuple[pd.DataFrame, TopKBlock]:
    """TopK passthrough: input is already in TopK format (one row per response)."""
    from_cols = list(block.from_columns) if isinstance(block.from_columns, list) else []
    if not from_cols:
        raise ValueError(f"TopK {block.name!r}: missing explicit from_columns for passthrough")

    # Check for res_columns mismatch (enforce for skip transforms)
    res_cols = list(source_block.res_columns) if isinstance(source_block.res_columns, list) else []
    if res_cols and from_cols != res_cols:
        raise ValueError(
            f"TopK block {block.name!r}: input_format={block.input_format!r} requires "
            f"res_columns to match from_columns; got res_columns={res_cols!r} "
            f"vs from_columns={from_cols!r}"
        )

    sdf = df[from_cols].copy()
    meta_out = _build_topk_output_block(
        name=block.name,
        columns=from_cols,
        from_cols=from_cols,
        res_cols=from_cols,
        block=block,
    )
    return sdf, meta_out


def _topk_transform_onehot(
    block: TopKBlock,
    df: pd.DataFrame,
    *,
    source_pattern: str | None,
    source_block: TopKBlock,
) -> tuple[pd.DataFrame, TopKBlock]:
    """TopK onehot transform: pivot multiple columns (mentions) into ranked TopK columns."""
    from_cols = list(block.from_columns) if isinstance(block.from_columns, list) else []
    na_vals = list(block.na_vals or [])

    # 1. Standardize input (mentions -> values)
    sdf = df[from_cols].astype("object").replace(na_vals, None)
    _check_topk_na_vals_after_replace(sdf, block_name=block.name)

    # 2. Pivot to values
    if source_pattern:
        regex = re.compile(source_pattern)
        agg_pos = block.agg_index
        agg_pos = agg_pos - 1 if agg_pos > 0 else agg_pos
        # Use capture group from regex as the value for each cell
        sdf.columns = [regex.match(c).groups()[agg_pos] for c in sdf.columns]  # type: ignore[union-attr]
        sdf = sdf.mask(~sdf.isna(), other=pd.Series(sdf.columns, index=sdf.columns), axis=1)
    else:
        # If no pattern, use column names (stripping prefix if present)
        if source_block.from_prefix:
            sdf.columns = [c.removeprefix(source_block.from_prefix) for c in sdf.columns]
        sdf = sdf.mask(~sdf.isna(), other=pd.Series(sdf.columns, index=sdf.columns), axis=1)

    # 3. Collapse left
    _throw_vals_left(sdf)

    # 4. Map to result column names
    res_cols = _resolve_topk_res_cols(block, source_block, source_pattern)
    sdf.columns = res_cols
    sdf = sdf.dropna(axis=1, how="all")

    # 5. Truncate to K
    kmax = block.k
    if isinstance(kmax, int) and sdf.shape[1] > kmax:
        if sdf.iloc[:, kmax:].notna().any().any():
            raise ValueError(
                f"TopK block {block.name!r}: truncation to k={kmax} would drop "
                f"non-NA values in columns {list(sdf.columns[kmax:])}"
            )
        sdf = sdf.iloc[:, :kmax]

    meta_out = _build_topk_output_block(
        name=block.name,
        columns=sdf.columns.tolist(),
        from_cols=from_cols,
        res_cols=res_cols,
        block=block,
    )
    return sdf, meta_out


def _resolve_topk_res_cols(block: TopKBlock, source: TopKBlock, pattern: str | None) -> list[str]:
    """Determine final result column names for a TopK sibling."""
    if isinstance(source.res_columns, list):
        return list(source.res_columns)

    if isinstance(source.res_columns, str) and pattern:
        regex = re.compile(pattern)
        from_cols = list(block.from_columns) if isinstance(block.from_columns, list) else []
        return [regex.match(c).expand(source.res_columns) for c in from_cols]  # type: ignore[union-attr]

    raise ValueError(f"TopK {block.name!r}: cannot resolve res_columns")


def _build_topk_output_block(
    *,
    name: str,
    columns: list[str],
    from_cols: list[str],
    res_cols: list[str],
    block: TopKBlock,
) -> TopKBlock:
    scale_dict = _block_scale_dict(block)
    return soft_validate(
        {
            "type": "topk",
            "name": name,
            "scale": scale_dict,
            "columns": list(columns),
            "from_columns": list(from_cols),
            "res_columns": list(res_cols),
            "agg_index": block.agg_index,
            "na_vals": list(block.na_vals or []),
            "k": block.k,
            "from_prefix": block.from_prefix,
            "input_format": block.input_format,
        },
        TopKBlock,
    )


def _maxdiff_apply_transform(
    block: MaxDiffBlock,
    df: pd.DataFrame,
    choice_sets: object,
    *,
    source_block: MaxDiffBlock,
) -> tuple[pd.DataFrame, MaxDiffBlock]:
    if block.input_format == "choice_sets":
        return _maxdiff_transform_choice_sets(block, df, choice_sets, source_block=source_block)
    if block.input_format == "resolved":
        return _maxdiff_transform_resolved(block, df)
    raise ValueError(f"unknown MaxDiff input_format: {block.input_format!r}")


def _maxdiff_transform_resolved(
    block: MaxDiffBlock,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, MaxDiffBlock]:
    # Roles arrive already resolved to concrete, index-aligned lists via
    # MaxDiffBlock.resolve_role_columns (regex roles matched + aligned by capture key,
    # explicit lists kept as-is). Here we only guard the explicit-list case, which the
    # validator does not length-check.
    if block.set_columns is None:
        raise ValueError(f"MaxDiffBlock {block.name!r}: set_columns is required for input_format='resolved'")
    best, worst, sets = list(block.best_columns), list(block.worst_columns), list(block.set_columns)
    if not (len(best) == len(worst) == len(sets)):
        raise ValueError(
            f"MaxDiff resolved lists must have equal length; got best={len(best)}, worst={len(worst)}, sets={len(sets)}"
        )
    cols = sorted(set(best) | set(worst) | set(sets))
    sdf = df[cols].copy()
    scale_dict = _block_scale_dict(block)
    out = soft_validate(
        {
            "type": "maxdiff",
            "name": block.name,
            "scale": scale_dict,
            "columns": {c: {} for c in cols},
            "best_columns": best,
            "worst_columns": worst,
            "set_columns": sets,
            "input_format": "resolved",
        },
        MaxDiffBlock,
    )
    return sdf, out


def _maxdiff_transform_choice_sets(
    block: MaxDiffBlock,
    df: pd.DataFrame,
    choice_sets: object,
    *,
    source_block: MaxDiffBlock,
) -> tuple[pd.DataFrame, MaxDiffBlock]:
    df = df.copy(deep=True)
    if not (
        isinstance(block.best_columns, list)
        and isinstance(block.worst_columns, list)
        and isinstance(block.set_columns, list)
    ):
        raise TypeError(
            f"_maxdiff_transform_choice_sets expects resolved role columns; got "
            f"best={type(block.best_columns).__name__}, "
            f"worst={type(block.worst_columns).__name__}, "
            f"set={type(block.set_columns).__name__}"
        )
    translate: dict[str, str] = (
        {str(k): str(v) for k, v in block.scale.translate.items()} if (block.scale and block.scale.translate) else {}
    )
    if not translate:
        raise ValueError(
            f"MaxDiffBlock {block.name!r}: scale.translate is required (maps 1-based "
            f"index strings to display names). Got empty translate."
        )
    topics: list[str] = [translate[k] for k in sorted(translate.keys(), key=int)]
    sets = choice_sets
    best_cols: Sequence[str] | str = block.best_columns
    worst_cols: Sequence[str] | str = block.worst_columns
    set_cols: Sequence[str] | str | None = block.set_columns
    # Parse setindex_column: can be None, str, or [str] or [str, dict]
    setindex_col_name: str | None = None
    setindex_col_meta: ColumnMeta | None = None
    if isinstance(block.setindex_column, str):
        setindex_col_name = block.setindex_column
    elif isinstance(block.setindex_column, list):
        _setindex_parts = list(block.setindex_column)
        setindex_col_name = str(_setindex_parts[0]) if _setindex_parts else None
        if _setindex_parts and len(_setindex_parts) > 1 and isinstance(_setindex_parts[1], dict):
            setindex_col_meta = soft_validate(_setindex_parts[1], ColumnMeta)
    if set_cols is None:
        raise ValueError("Maxdiff create blocks must define 'set_columns'.")
    best_cols = list(best_cols)
    worst_cols = list(worst_cols)
    set_cols = list(set_cols)

    def _maybe_json_load(value: str) -> list[object] | None:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else None
        except (TypeError, json.JSONDecodeError):
            return None

    def _tokens_from_value(value: object) -> list[str] | None:
        """Normalise one set cell into a list of string tokens (or None for NA).
        Accepts native lists, or a string encoding them as JSON or comma-separated."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            return [str(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            parsed = _maybe_json_load(stripped) if stripped.startswith("[") and stripped.endswith("]") else None
            if parsed is not None:
                return [str(item) for item in parsed]
            return [part.strip() for part in stripped.split(",") if part.strip()]
        raise ValueError(f"Unsupported maxdiff set specification value: {value}")

    def _tokens_to_topics(tokens: list[str] | None) -> list[str] | None:
        """Map a set cell's tokens to display-name topics: an integer token is a 1-based
        index into `topics`, anything else is a raw name run through `translate`. (The two
        agree on index tokens: `topics` is built from `translate` in index order.)"""
        if tokens is None:
            return None
        out: list[str] = []
        for tok in tokens:
            t = tok.strip()
            try:
                idx = int(t)
            except ValueError:
                out.append(translate.get(t, t))
                continue
            if idx < 1 or idx > len(topics):
                raise ValueError(f"Maxdiff set index {idx} is out of bounds for topics list of size {len(topics)}.")
            out.append(topics[idx - 1])
        return out

    ordered_cols = best_cols + worst_cols

    if setindex_col_name:
        df = df[ordered_cols + [setindex_col_name]]
        if sets is None:
            raise ValueError("Maxdiff definitions using 'setindex_column' must also define 'sets'.")
        topics_arr = np.array(["", *topics], dtype=object)  # "" at index 0: survey sets are 1-indexed
        sets_arr = np.asarray(sets, dtype=int)
        lsets = topics_arr[sets_arr]

        setindex = df[setindex_col_name].astype(np.int64).to_numpy() - 1
        selected_sets = lsets[setindex]
        df_setcols = pd.DataFrame(selected_sets.tolist(), columns=set_cols, index=df.index)
        df[set_cols] = df_setcols
    else:
        df = df[ordered_cols + set_cols]
        for col in set_cols:
            df[col] = [_tokens_to_topics(_tokens_from_value(v)) for v in df[col].tolist()]  # type: ignore[assignment]

    # Pre-translate already mapped index strings to topic names before the transform ran,
    # so df[col] already contains translated values — just cast to categorical.
    for col in best_cols + worst_cols:
        s = df[col]
        s = pd.Categorical(s, categories=topics)
        df[col] = s

    df = df.sort_index(axis=1)

    base_columns = sorted(best_cols + worst_cols + cast(list[str], set_cols))
    best_worst_col_meta = ColumnMeta(categories=topics)
    columns_spec: dict[str, ColumnMeta] = {col: best_worst_col_meta for col in base_columns}
    if setindex_col_name is not None:
        if setindex_col_meta is None:
            setindex_col_meta = ColumnMeta()
        if setindex_col_meta.categories is None:
            setindex_col_meta = setindex_col_meta.model_copy(update={"categories": topics})
        if setindex_col_meta.continuous is False:
            setindex_col_meta = setindex_col_meta.model_copy(update={"continuous": True})
        columns_spec = {setindex_col_name: setindex_col_meta} | columns_spec

    scale_dict = _block_scale_dict(block)
    scale_dict["categories"] = topics

    output_block = soft_validate(
        {
            "type": "maxdiff",
            "name": block.name,
            "scale": scale_dict,
            "columns": columns_spec,
            "best_columns": list(best_cols),
            "worst_columns": list(worst_cols),
            "set_columns": list(cast(list[str], set_cols)),
            "setindex_column": source_block.setindex_column,
        },
        MaxDiffBlock,
    )
    return df, output_block


def _onehot_apply_transform(
    block: OneHotBlock,
    df: pd.DataFrame,
    choices: list[str] | None,
) -> tuple[pd.DataFrame, OneHotBlock]:
    if block.input_format == "leftpacked":
        return _onehot_transform_leftpacked(block, df, choices)
    if block.input_format == "wide":
        return _onehot_transform_wide(block, df, choices)
    raise ValueError(f"unknown OneHot input_format: {block.input_format!r}")


def _onehot_transform_leftpacked(
    block: OneHotBlock,
    df: pd.DataFrame,
    choices: list[str] | None,
) -> tuple[pd.DataFrame, OneHotBlock]:
    from_cols = _match_columns(block, df)
    src = df[from_cols].astype("object")
    if block.na_vals:
        src = src.replace(block.na_vals, None)

    observed = [
        v for v in pd.unique(src.values.ravel("K")) if v is not None and not (isinstance(v, float) and pd.isna(v))
    ]

    if choices is not None:
        unknown = set(observed) - set(choices)
        if unknown:
            raise ValueError(f"OneHot block {block.name!r}: values {sorted(unknown)} not in choices")
        final_choices = list(choices)
    else:
        final_choices = sorted(observed)

    prefix = block.res_prefix or f"{block.name}_"
    out_cols = [f"{prefix}{c}" for c in final_choices]
    out_df = pd.DataFrame(
        {f"{prefix}{c}": src.eq(c).any(axis=1) for c in final_choices},
        index=df.index,
    )

    scale_dict = _block_scale_dict(block)
    out = soft_validate(
        {
            "type": "onehot",
            "name": block.name,
            "scale": scale_dict,
            "columns": {c: {} for c in out_cols},
            "from_columns": from_cols,
            "input_format": "leftpacked",
            "choices": final_choices,
            "res_prefix": block.res_prefix,
        },
        OneHotBlock,
    )
    return out_df, out


def _onehot_transform_wide(
    block: OneHotBlock,
    df: pd.DataFrame,
    choices: list[str] | None,
) -> tuple[pd.DataFrame, OneHotBlock]:
    from_cols = _match_columns(block, df)
    sdf = df[from_cols].copy()
    prefix = block.res_prefix or ""
    final_choices: list[str] | None
    if choices is not None:
        final_choices = list(choices)
    elif prefix:
        final_choices = [c.removeprefix(prefix) for c in from_cols]
    else:
        final_choices = None

    scale_dict = _block_scale_dict(block)
    out = soft_validate(
        {
            "type": "onehot",
            "name": block.name,
            "scale": scale_dict,
            "columns": {c: {} for c in from_cols},
            "from_columns": from_cols,
            "input_format": "wide",
            "choices": final_choices,
            "res_prefix": block.res_prefix,
        },
        OneHotBlock,
    )
    return sdf, out


def _demote_to_plain(block: ColumnBlockMeta) -> ColumnBlockMeta:
    """Demote a specialized block (TopKBlock / MaxDiffBlock / OneHotBlock) to a plain
    ColumnBlockMeta, preserving every field declared on ``ColumnBlockMeta`` and dropping
    subclass-specific ones. Using ``model_fields`` instead of a hand-enumerated list means
    new fields added to ``ColumnBlockMeta`` are carried over automatically. Input-only
    directives (from_columns, subgroup_labels) are cleared."""
    kwargs = {k: getattr(block, k) for k in ColumnBlockMeta.model_fields if k != "type"}
    # Clear input-only directives that are not part of the demoted plain block
    kwargs["from_columns"] = None
    kwargs["subgroup_labels"] = None
    return ColumnBlockMeta(**kwargs)


def _combine_first_preserving_order(*frames: pd.DataFrame) -> pd.DataFrame:
    """Like ``DataFrame.combine_first(other1).combine_first(other2)...`` but
    keeps the natural source-data column order instead of lex-sorting.

    ``DataFrame.combine_first`` lex-sorts the union of column names in its
    result. That ordering is wrong for our TopK/MaxDiff/OneHot block pipeline
    when ``from_columns`` is a regex with a single numeric capture group
    (e.g. ``vA10_M_(\\d+)`` matching ``vA10_M_1, ..., vA10_M_14, vA10_M_99``):
    after the onehot pivot + leftpack, the rename step assumes the columns are
    still in source order, so a lex-sorted ``vA10_M_1, vA10_M_10, vA10_M_11,
    ..., vA10_M_2, ...`` produces ``issue_top_1, issue_top_10, issue_top_11``
    instead of ``issue_top_1, issue_top_2, issue_top_3``.

    Restore the natural order: first frame's columns first (in their order),
    then any new columns from each subsequent frame (in their order), deduped.
    The underlying ``combine_first`` semantics (left-priority value coalescing
    plus index union) are preserved — only the column ordering changes.
    """
    if not frames:
        return pd.DataFrame()
    result = frames[0]
    for other in frames[1:]:
        result = result.combine_first(other)
    desired: list[str] = []
    seen: set[str] = set()
    for frame in frames:
        for c in frame.columns:
            if c not in seen:
                desired.append(c)
                seen.add(c)
    # Tolerate any columns combine_first invented that we didn't account for
    desired += [c for c in result.columns if c not in seen]
    return result[desired]
