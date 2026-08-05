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

import ast
import json
import re
from collections.abc import Iterable, Iterator
from copy import deepcopy
from typing import Any, TypeVar, cast

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

from salk_toolkit.io.core import _is_series_of_lists, expand_na_vals, expand_value_keys


def _throw_vals_left(df: pd.DataFrame) -> None:
    """Move all NaN values to the right in each row (in-place)."""
    # Helper fun to move inplace all nan values to right.
    df.iloc[:, :] = df.apply(lambda row: sorted(row, key=pd.isna), axis=1).to_list()


def _null_values(df: pd.DataFrame, cols: list[str], values: list[str]) -> pd.DataFrame:
    """Null the listed cell values, per column so list-valued cells (MaxDiff sets) pass through."""
    targets = expand_na_vals(list(values))
    out = df.copy()
    for c in cols:
        if not _is_series_of_lists(out[c]):
            out[c] = out[c].astype("object").replace(targets, None)
    return out


def _prepare_cells(
    block: ColumnBlockMeta, df: pd.DataFrame, cols: list[str], meta_not_asked: list[str] | None
) -> tuple[pd.DataFrame, pd.Series]:
    """Null not_asked then not_selected values, and derive the asked mask in between: a row was
    asked iff some source cell still held a value - a pick, or an explicit not-picked marker."""
    if not cols:
        return df, pd.Series(True, index=df.index)
    not_asked = block.not_asked if block.not_asked is not None else meta_not_asked
    df = _null_values(df, cols, not_asked) if not_asked else df
    asked = df[cols].notna().any(axis=1)

    not_selected = getattr(block, "not_selected", [])
    if not_selected:
        nulled = _null_values(df, cols, not_selected)
        matched = df[cols].notna() & nulled[cols].isna()
        if not matched.to_numpy().any():
            raise ValueError(f"Block {block.name!r}: not_selected={not_selected!r} matched no cell")
        no_match = [c for c in cols if not matched[c].any()]
        if no_match:
            warn(f"Block {block.name!r}: not_selected matched nothing in column(s) {no_match}")
        df = nulled
    return df, asked


def _map_cell(value: object, mapping: dict) -> object:
    """Map one cell through `mapping`: element-wise inside list cells, NA stays NA."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, np.ndarray)):
        return [mapping.get(x, x) for x in cast(Iterable[object], value)]
    return mapping.get(value, value)


def _apply_pre_transform_translate(block: ColumnBlockMeta, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Stage 3: map raw cell values through scale.translate before the transform runs."""
    if block.scale is None or not block.scale.translate:
        return df
    # Key expansion makes int64/float64 index cells (CSV round-trips) match string keys
    translate = cast("dict[object, object]", expand_value_keys(block.scale.translate))
    df = df.copy()
    for c in cols:
        df[c] = df[c].map(lambda v: _map_cell(v, translate))
    return df


def _apply_post_transform_translate(
    block: ColumnBlockMeta,
    sdf: pd.DataFrame,
    meta: ColumnBlockMeta,
) -> tuple[pd.DataFrame, ColumnBlockMeta]:
    """Stage 5: map output cells and the scale's categories through `scale.translate_after`.
    MaxDiff rejects translate_after at validation, so only topk/onehot reach this."""
    scale = meta.scale
    if scale is None or not scale.translate_after:
        return sdf, meta
    t = dict(scale.translate_after)

    # Categoricals go through object so a many-to-one map merges categories instead of raising
    for col in sdf.columns:
        sdf[col] = sdf[col].astype("object").map(lambda v: _map_cell(v, t))

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
    block: TopKBlock | MaxDiffBlock | OneHotBlock, df: pd.DataFrame, not_asked: list[str] | None = None
) -> Iterator[tuple[pd.DataFrame, ColumnBlockMeta]]:
    """Driver for specialized blocks: explode into siblings, then run the
    not_asked/not_selected -> pre-translate -> transform -> post-translate stages on each.
    `not_asked` is the meta-level default; a block-level not_asked overrides it."""
    siblings: list[ColumnBlockMeta]
    if isinstance(block, MaxDiffBlock) and block.from_columns is None:
        siblings = [_apply_role_resolution(block, block, df, raw_key="")]
    elif isinstance(block, OneHotBlock):
        siblings = [block]
    else:
        siblings = _subgroup_explode(block, df)

    if block.model_spec is not None and len(siblings) > 1:
        raise ValueError(
            f"Block {block.name!r}: model_spec on a block that explodes into {len(siblings)} "
            f"sibling blocks is ambiguous and not supported; the siblings get per-sibling defaults"
        )

    for sib in siblings:
        cols = sib.source_columns(df)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Block {sib.name!r}: declared column(s) {missing} are not in the data")
        df_t, asked = _prepare_cells(sib, df, cols, not_asked)
        df_t = _apply_pre_transform_translate(sib, df_t, sib.translate_columns(df))
        sdf, meta = _apply_transform(sib, df_t, source_block=block, asked=asked)
        sdf, meta = _apply_post_transform_translate(sib, sdf, meta)
        # Stamp the observation-model description onto the output block: an authored
        # model_spec wins, else the type default (ordinal_ranking for topk/maxdiff).
        spec = block.model_spec or meta.default_model_spec()
        if spec is not None:
            meta = meta.model_copy(update={"model_spec": spec})
        yield sdf, meta


def _apply_transform(
    block: ColumnBlockMeta,
    df: pd.DataFrame,
    *,
    source_block: ColumnBlockMeta,
    asked: pd.Series,
) -> tuple[pd.DataFrame, ColumnBlockMeta]:
    """Dispatch to the per-type transform."""
    if isinstance(block, TopKBlock):
        assert isinstance(source_block, TopKBlock)
        source_pattern = source_block.from_columns if isinstance(source_block.from_columns, str) else None
        return _topk_apply_transform(block, df, source_pattern=source_pattern, source_block=source_block)
    if isinstance(block, MaxDiffBlock):
        assert isinstance(source_block, MaxDiffBlock)
        # A design-keyed dict (2-level values, looked up by setindex string) passes through
        # whole; anything else may be sibling-keyed and goes through subgroup extraction.
        cs: object = source_block.choice_sets
        if not _is_design_keyed_sets(cs):
            cs = _get_subgroup_config(cs, block.name, source_block.name)
        return _maxdiff_apply_transform(block, df, cs, source_block=source_block)
    assert isinstance(block, OneHotBlock) and isinstance(source_block, OneHotBlock)
    return _onehot_apply_transform(block, df, source_block.choices, asked)


def _is_design_keyed_sets(value: object) -> bool:
    """True for a design-name-keyed choice_sets dict: values are per-question item lists
    (2 levels; scalars inside), vs sibling-keyed per-version tables (3 levels)."""
    if not isinstance(value, dict) or not value:
        return False
    v = next(iter(value.values()))
    return isinstance(v, list) and bool(v) and isinstance(v[0], list) and not any(isinstance(x, list) for x in v[0])


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


def _agg_pos(agg_index: int, n_groups: int | None = None) -> int:
    """0-based capture-group position of the item index (agg_index is 1-based; -1 = last)."""
    pos = agg_index - 1 if agg_index > 0 else agg_index
    return pos + n_groups if (n_groups is not None and pos < 0) else pos


def _match_columns(block: ColumnBlockMeta, df: pd.DataFrame) -> list[str]:
    """Stage 1: resolve `from_columns` to the concrete df columns it selects, raising if any
    declared column is absent or nothing matches. Uses the base resolver explicitly, so it stays
    correct for MaxDiffBlock (whose `source_columns` override means best/worst/set, not this)."""
    cols = ColumnBlockMeta.source_columns(block, df)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Block {block.name!r}: declared column(s) {missing} are not in the data")
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
        return [_apply_role_resolution(_narrow_sibling(block, matched_cols, label_suffix=""), block, df, raw_key="")]

    regex = re.compile(pattern)
    first = regex.match(matched_cols[0])
    assert first is not None
    n_groups = len(first.groups())

    # TOPK-specific: skip one group for sibling identity if aggregating.
    # Otherwise, use all groups.
    agg_pos = None
    if isinstance(block, TopKBlock):
        agg_pos = _agg_pos(block.agg_index, n_groups)
        if not (0 <= agg_pos < n_groups):
            raise ValueError(
                f"Block {block.name!r}: agg_index={block.agg_index} out of range for {n_groups} capture group(s)"
            )

    non_agg_positions = [i for i in range(n_groups) if i != agg_pos] if agg_pos is not None else list(range(n_groups))

    if not non_agg_positions:
        return [_apply_role_resolution(_narrow_sibling(block, matched_cols, label_suffix=""), block, df, raw_key="")]

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
        _apply_role_resolution(_narrow_sibling(block, cols, label_suffix=_label(key)), block, df, raw_key=key[0])
        for key, cols in sibling_cols.items()
    ]


def _apply_role_resolution(
    sib: ColumnBlockMeta, source: ColumnBlockMeta, df: pd.DataFrame, *, raw_key: str
) -> ColumnBlockMeta:
    """Resolve per-type column roles. Roles match on the raw capture value, not the display
    label a subgroup_labels mapping may have given the sibling."""
    updates = sib.resolve_role_columns(df, raw_key)
    for role, cols in updates.items():
        if not cols:
            raise ValueError(f"Block {sib.name!r}: {role} matched no columns (raw key {raw_key!r})")
    return sib.model_copy(update=updates) if updates else sib


def _topk_apply_transform(
    block: TopKBlock,
    df: pd.DataFrame,
    *,
    source_pattern: str | None,
    source_block: TopKBlock,
) -> tuple[pd.DataFrame, TopKBlock]:
    """Dispatch TopK transformation based on input_format."""
    if block.input_format in ("leftpacked", "ranked_leftpack"):
        return _topk_transform_passthrough(block, df, source_block=source_block)
    return _topk_transform_onehot(block, df, source_pattern=source_pattern, source_block=source_block)


def _check_k(sdf: pd.DataFrame, block: TopKBlock) -> None:
    """k is a data check, not a truncation: more picks than the question allowed is an error."""
    if sdf.shape[1] > block.k:
        raise ValueError(
            f"TopK block {block.name!r}: {sdf.shape[1]} picks in some row exceeds k={block.k} "
            f"(surplus slots {list(sdf.columns[block.k :])}); fix k or the data"
        )


def _topk_transform_passthrough(
    block: TopKBlock,
    df: pd.DataFrame,
    *,
    source_block: TopKBlock,
) -> tuple[pd.DataFrame, TopKBlock]:
    """TopK passthrough: the source columns are already the pick slots."""
    from_cols = list(block.from_columns)
    res_cols = list(source_block.res_columns) if isinstance(source_block.res_columns, list) else []
    if res_cols and from_cols != res_cols:
        raise ValueError(
            f"TopK block {block.name!r}: input_format={block.input_format!r} requires "
            f"res_columns to match from_columns; got res_columns={res_cols!r} "
            f"vs from_columns={from_cols!r}"
        )

    sdf = df[from_cols].copy()
    _check_k(sdf, block)
    if block.not_selected:  # nulled markers leave gaps mid-row
        _throw_vals_left(sdf)
    meta_out = _output_block(block, columns=from_cols, from_columns=from_cols, res_columns=from_cols)
    return sdf, meta_out


def _order_by_rank(sdf: pd.DataFrame, ranks: np.ndarray) -> pd.DataFrame:
    """Reorder each row's items by their rank cell (ascending); unranked items sort last."""
    order = np.argsort(np.where(np.isnan(ranks), np.inf, ranks), axis=1, kind="stable")
    ordered = np.take_along_axis(sdf.to_numpy(dtype=object), order, axis=1)
    return pd.DataFrame(ordered, index=sdf.index, columns=sdf.columns)


def _topk_transform_onehot(
    block: TopKBlock,
    df: pd.DataFrame,
    *,
    source_pattern: str | None,
    source_block: TopKBlock,
) -> tuple[pd.DataFrame, TopKBlock]:
    """TopK onehot: one source column per item; a non-null cell means the item was picked
    (`ranked_onehot`: the cell is its rank, `cell_values`: the cell is the item itself)."""
    from_cols = list(block.from_columns)
    sdf = df[from_cols].astype("object")

    ranked = block.input_format == "ranked_onehot"
    ranks = sdf.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) if ranked else None
    if ranks is not None and (bad := sdf.notna().to_numpy() & np.isnan(ranks)).any():
        raise ValueError(
            f"TopK block {block.name!r}: input_format='ranked_onehot' needs numeric rank cells; "
            f"got e.g. {sdf.to_numpy(dtype=object)[bad][:3].tolist()}"
        )

    # Cells identify the item by their column, unless they already carry the item value
    if not block.cell_values:
        if source_pattern:
            regex = re.compile(source_pattern)
            agg_pos = _agg_pos(block.agg_index)
            sdf.columns = [regex.match(c).groups()[agg_pos] for c in sdf.columns]  # type: ignore[union-attr]
        elif source_block.from_prefix:
            sdf.columns = [c.removeprefix(source_block.from_prefix) for c in sdf.columns]
        sdf = sdf.mask(~sdf.isna(), other=pd.Series(sdf.columns, index=sdf.columns), axis=1)

    sdf = _order_by_rank(sdf, ranks) if ranks is not None else sdf
    _throw_vals_left(sdf)

    # cell_values slots are named <prefix>1..n; otherwise each source column maps to a res column
    if block.cell_values:
        prefix = _cell_values_res_prefix(block, source_block, source_pattern, from_cols)
        res_cols = [f"{prefix}{i + 1}" for i in range(sdf.shape[1])]
    else:
        res_cols = _resolve_topk_res_cols(block, source_block, source_pattern)
    sdf.columns = res_cols
    sdf = sdf.dropna(axis=1, how="all")
    _check_k(sdf, block)

    meta_out = _output_block(block, columns=sdf.columns.tolist(), from_columns=from_cols, res_columns=res_cols)
    return sdf, meta_out


def _cell_values_res_prefix(block: TopKBlock, source: TopKBlock, pattern: str | None, from_cols: list[str]) -> str:
    """Slot-name prefix for cell_values mode: res_columns expanded against the first matched
    column (so subgroup backrefs like 'Q9_\\1b' resolve per sibling), or used verbatim."""
    res = source.res_columns
    if not isinstance(res, str):
        raise ValueError(f"TopK {block.name!r}: cell_values requires a string res_columns prefix/template")
    if pattern:
        m = re.compile(pattern).match(from_cols[0])
        assert m is not None, f"Column {from_cols[0]} should match regex {pattern}"
        return m.expand(res)
    return res


def _resolve_topk_res_cols(block: TopKBlock, source: TopKBlock, pattern: str | None) -> list[str]:
    """Determine final result column names for a TopK sibling."""
    if isinstance(source.res_columns, list):
        return list(source.res_columns)

    if isinstance(source.res_columns, str) and pattern:
        regex = re.compile(pattern)
        from_cols = list(block.from_columns) if isinstance(block.from_columns, list) else []
        return [regex.match(c).expand(source.res_columns) for c in from_cols]  # type: ignore[union-attr]

    raise ValueError(f"TopK {block.name!r}: cannot resolve res_columns")


BlockT = TypeVar("BlockT", bound=ColumnBlockMeta)


def _output_block(block: BlockT, **updates: object) -> BlockT:
    """Rebuild a block with its roles resolved to concrete lists. Carries every declared field
    over (new schema fields included) and clears the input-only subgroup directive."""
    spec = block.model_dump(mode="python") | {"scale": _block_scale_dict(block), "subgroup_labels": None, **updates}
    return soft_validate(spec, type(block))


def _maxdiff_apply_transform(
    block: MaxDiffBlock,
    df: pd.DataFrame,
    choice_sets: object,
    *,
    source_block: MaxDiffBlock,
) -> tuple[pd.DataFrame, MaxDiffBlock]:
    if block.input_format == "choice_sets":
        return _maxdiff_transform_choice_sets(block, df, choice_sets, source_block=source_block)
    return _maxdiff_transform_resolved(block, df)


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
    out = _output_block(block, columns={c: {} for c in cols}, best_columns=best, worst_columns=worst, set_columns=sets)
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
    # Topic universe: an index-keyed translate ("1" -> name), else scale.categories
    # (data already in display names; translate, if any, is then a plain name recode).
    index_keyed = bool(translate) and all(k.lstrip("-").isdigit() for k in translate)
    if index_keyed:
        topics: list[str] = [translate[k] for k in sorted(translate.keys(), key=int)]
    elif block.scale and isinstance(block.scale.categories, list):
        topics = [str(c) for c in block.scale.categories]
    else:
        raise ValueError(
            f"MaxDiffBlock {block.name!r}: needs an index-keyed scale.translate "
            f"(1-based index -> display name) or an explicit scale.categories topic list."
        )
    sets = choice_sets
    best_cols, worst_cols, set_cols = list(block.best_columns), list(block.worst_columns), list(block.set_columns)
    # Parse setindex_column: can be None, str, or [str] or [str, dict]
    setindex_col_name: str | None = None
    setindex_col_meta: ColumnMeta | None = None
    if isinstance(block.setindex_column, str):
        setindex_col_name = block.setindex_column
    elif isinstance(block.setindex_column, list):
        parts = list(block.setindex_column)
        setindex_col_name = str(parts[0]) if parts else None
        if len(parts) > 1 and isinstance(parts[1], dict):
            setindex_col_meta = soft_validate(parts[1], ColumnMeta)

    def _parse_list_literal(value: str) -> list[object] | None:
        """A set cell can arrive as a JSON list or a python-repr list (CSV round-trip)."""
        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(value)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
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
            parsed = _parse_list_literal(stripped) if stripped.startswith("[") and stripped.endswith("]") else None
            if parsed is not None:
                return [str(item) for item in parsed]
            return [part.strip() for part in stripped.split(",") if part.strip()]
        raise ValueError(f"Unsupported maxdiff set specification value: {value}")

    def _topic_of(token: object) -> str:
        """Resolve one set token to a display topic. With an index-keyed scale.translate the
        token IS a translate key; otherwise an integer token is a 1-based position in `topics`."""
        t = str(token)
        if index_keyed:
            if t in translate:
                return translate[t]
            if t in topics:
                return t  # already a display name
            raise ValueError(f"Maxdiff set token {t!r} is not a scale.translate key of block {block.name!r}")
        if t.strip().lstrip("-").isdigit():
            idx = int(t)
            if not 1 <= idx <= len(topics):
                raise ValueError(f"Maxdiff set index {idx} is out of bounds for topics list of size {len(topics)}.")
            return topics[idx - 1]
        return translate.get(t, t)

    def _tokens_to_topics(tokens: list[str] | None) -> list[str] | None:
        return [_topic_of(t) for t in tokens] if tokens is not None else None

    ordered_cols = best_cols + worst_cols

    setindex_designs: list[str] | None = None  # design-name-keyed sets (vs numeric version index)
    if setindex_col_name:
        df = df[ordered_cols + [setindex_col_name]]
        if sets is None:
            raise ValueError("Maxdiff definitions using 'setindex_column' must also define 'choice_sets'.")
        if isinstance(sets, dict):
            # Design-name strings in the setindex column, sets keyed by design name;
            # set entries are topic indices/keys or topic names, resolved like set cells.
            per_design = {str(k): [[_topic_of(x) for x in q] for q in cast(list, v)] for k, v in sets.items()}
            for dname, qsets in per_design.items():
                if len(qsets) != len(set_cols):
                    raise ValueError(f"Maxdiff design {dname!r} has {len(qsets)} sets for {len(set_cols)} questions")
                bad = {t for q in qsets for t in q} - set(topics)
                if bad:
                    raise ValueError(f"Maxdiff design {dname!r} contains unknown topics: {sorted(bad)}")
            setindex_designs = list(per_design)
            keys = df[setindex_col_name].astype(str)
            unknown_keys = sorted(set(keys) - set(setindex_designs))
            if unknown_keys:
                raise ValueError(f"Maxdiff setindex values not in choice_sets designs: {unknown_keys}")
            for qi, sc in enumerate(set_cols):
                df[sc] = [per_design[k][qi] for k in keys]
        else:
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

    base_columns = sorted(best_cols + worst_cols + set_cols)
    best_worst_col_meta = ColumnMeta(categories=topics)
    columns_spec: dict[str, ColumnMeta] = {col: best_worst_col_meta for col in base_columns}
    if setindex_col_name is not None:
        setindex_col_meta = setindex_col_meta or ColumnMeta()
        if setindex_designs is not None:  # design-name setindex: categorical over the design keys
            if setindex_col_meta.categories is None:
                setindex_col_meta = setindex_col_meta.model_copy(update={"categories": setindex_designs})
        else:
            # A numeric version index is continuous - which also keeps the scale's topic
            # categories off it; those belong on the best/worst columns
            setindex_col_meta = setindex_col_meta.model_copy(update={"continuous": True})
        columns_spec = {setindex_col_name: setindex_col_meta} | columns_spec

    output_block = _output_block(
        block,
        scale=_block_scale_dict(block) | {"categories": topics},
        columns=columns_spec,
        best_columns=best_cols,
        worst_columns=worst_cols,
        set_columns=set_cols,
        setindex_column=source_block.setindex_column,
        choice_sets=None,  # design table consumed: the sets are materialised into set_columns
    )
    return df, output_block


def _onehot_apply_transform(
    block: OneHotBlock,
    df: pd.DataFrame,
    choices: list[str] | None,
    asked: pd.Series,
) -> tuple[pd.DataFrame, OneHotBlock]:
    if block.input_format == "leftpacked":
        return _onehot_transform_leftpacked(block, df, choices, asked)
    return _onehot_transform_wide(block, df, choices, asked)


def _onehot_output(
    block: OneHotBlock, bool_df: pd.DataFrame, from_cols: list[str], final_choices: list[str], asked: pd.Series
) -> tuple[pd.DataFrame, OneHotBlock]:
    """Shared onehot output stage: code the boolean frame and build the output block. Rows that
    were never asked stay NA - an unasked question is not the same answer as picking nothing."""
    scale_dict = _block_scale_dict(block)
    if block.coding is not None:
        coding = list(block.coding)
        coded = np.where(bool_df.to_numpy(), coding[1], coding[0])
        bool_df = pd.DataFrame(coded, index=bool_df.index, columns=bool_df.columns).astype(
            pd.CategoricalDtype(categories=coding, ordered=True)
        )
        scale_dict["categories"] = coding
        scale_dict["ordered"] = True
        scale_dict.pop("translate", None)  # consumed for choice naming; must not re-map coded cells
    bool_df = bool_df.mask(~asked)
    out = _output_block(
        block,
        scale=scale_dict,
        columns={c: {} for c in bool_df.columns},
        from_columns=from_cols,
        choices=final_choices,
    )
    return bool_df, out


def _onehot_transform_leftpacked(
    block: OneHotBlock,
    df: pd.DataFrame,
    choices: list[str] | None,
    asked: pd.Series,
) -> tuple[pd.DataFrame, OneHotBlock]:
    from_cols = _match_columns(block, df)
    src = df[from_cols].astype("object")

    observed = [
        v for v in pd.unique(src.values.ravel("K")) if v is not None and not (isinstance(v, float) and pd.isna(v))
    ]

    if choices is not None:
        unknown = set(observed) - set(choices)
        if unknown:
            raise ValueError(f"OneHot block {block.name!r}: values {sorted(map(str, unknown))} not in choices")
        final_choices = list(choices)
    else:
        final_choices = sorted(observed)

    prefix = block.res_prefix if block.res_prefix is not None else f"{block.name}_"
    bool_df = pd.DataFrame(
        {f"{prefix}{c}": src.eq(c).any(axis=1) for c in final_choices},
        index=df.index,
    )
    return _onehot_output(block, bool_df, from_cols, final_choices, asked)


def _onehot_wide_truthy(s: pd.Series) -> pd.Series:
    """A wide cell is selected iff non-NA and, when numeric, nonzero."""
    if s.dtype == bool:
        return s.fillna(False)
    num = pd.to_numeric(s, errors="coerce")
    return pd.Series(np.where(num.notna(), num != 0, s.notna()), index=s.index)


def _onehot_transform_wide(
    block: OneHotBlock,
    df: pd.DataFrame,
    choices: list[str] | None,
    asked: pd.Series,
) -> tuple[pd.DataFrame, OneHotBlock]:
    """Wide: one source column per choice. Choice identity = first regex capture group of
    `from_columns` (or the bare column name), named through scale.translate. Choices the
    translate/choices universe expects but the data lacks become all-unselected (warn)."""
    pattern = block.from_columns if isinstance(block.from_columns, str) else None
    regex = re.compile(pattern) if pattern else None
    translate_raw = dict(block.scale.translate) if block.scale and block.scale.translate else {}
    translate = cast("dict[object, str]", expand_value_keys({str(k): str(v) for k, v in translate_raw.items()}))
    from_cols = _match_columns(block, df)

    def _key(c: str) -> str:
        m = regex.match(c) if regex and regex.groups else None
        return m.group(1) if m else c

    matched_names: dict[str, str] = {}  # choice name -> source column
    for col in from_cols:
        name = translate.get(_key(col), str(_key(col)))
        if name in matched_names:
            raise ValueError(
                f"OneHot block {block.name!r}: choice {name!r} matches both {matched_names[name]} and {col}"
            )
        matched_names[name] = col

    # Expected universe fixes order and surfaces missing columns; falls back to matched order
    if choices is not None:
        universe = list(choices)
    elif translate_raw:
        universe = list(dict.fromkeys(str(v) for v in translate_raw.values()))
    else:
        universe = list(matched_names)
    unknown = set(matched_names) - set(universe)
    if unknown:
        raise ValueError(f"OneHot block {block.name!r}: matched choices {sorted(unknown)} not in choices")
    missing = [c for c in universe if c not in matched_names]
    if missing:
        warn(f"OneHot block {block.name!r}: no source column for choice(s) {missing}; coding as all-unselected")

    src = df[list(matched_names.values())]
    prefix = block.res_prefix if block.res_prefix is not None else f"{block.name}_"
    unselected = pd.Series(False, index=df.index)
    bool_df = pd.DataFrame(
        {
            f"{prefix}{c}": _onehot_wide_truthy(src[matched_names[c]]) if c in matched_names else unselected
            for c in universe
        },
        index=df.index,
    )
    return _onehot_output(block, bool_df, list(matched_names.values()), universe, asked)


def _demote_to_plain(block: ColumnBlockMeta) -> ColumnBlockMeta:
    """Demote a specialized block (TopKBlock / MaxDiffBlock / OneHotBlock) to a plain
    ColumnBlockMeta, preserving every field declared on ``ColumnBlockMeta`` and dropping
    subclass-specific ones. Using ``model_fields`` instead of a hand-enumerated list means
    new fields added to ``ColumnBlockMeta`` are carried over automatically. Input-only
    directives (from_columns, subgroup_labels) are cleared."""
    kwargs = {k: getattr(block, k) for k in ColumnBlockMeta.model_fields if k != "type"}
    # Clear input-only directives that are not part of the demoted plain block
    # (model_spec belongs to the derived output block, not the raw-columns parent)
    kwargs["from_columns"] = None
    kwargs["subgroup_labels"] = None
    kwargs["model_spec"] = None
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
