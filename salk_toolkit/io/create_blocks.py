"""Create-block builders (topk/maxdiff) that expand grouped raw columns into derived blocks."""

import json
import re
from collections.abc import Iterator, Sequence
from copy import deepcopy
from typing import Callable, cast

import numpy as np
import pandas as pd

from salk_toolkit.utils import (
    warn,
)
from salk_toolkit.validation import (
    ColumnBlockMeta,
    ColumnMeta,
    MaxDiffBlock,
    TopKBlock,
    soft_validate,
)


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


def _create_topk_metas_and_dfs(
    df: pd.DataFrame,
    block_with_create: ColumnBlockMeta,
    **kwargs: dict[str, str],  # used by other create blocks
) -> tuple[list[pd.DataFrame], list[ColumnBlockMeta]]:
    """Create top K aggregations from DataFrame."""
    if block_with_create.create is None:
        raise ValueError("ColumnBlockMeta must have a create block")
    create = block_with_create.create
    if not isinstance(create, TopKBlock):
        raise ValueError("Expecting TopKBlock create block")
    name = f"{block_with_create.name}_{create.type}"
    from_columns = create.from_columns
    if from_columns is None:
        raise ValueError("TopKBlock must have from_columns specified")

    has_regex = isinstance(from_columns, str)
    has_list = isinstance(from_columns, list)

    if has_regex:
        return _create_topk_metas_and_dfs_regex(df, block_with_create, create, name)
    elif has_list:
        return _create_topk_metas_and_dfs_list(df, block_with_create, create, name)
    else:
        raise ValueError(f"from_columns must be either str (regex) or list, got {type(from_columns)}")


def _create_topk_metas_and_dfs_regex(
    df: pd.DataFrame,
    block_with_create: ColumnBlockMeta,
    create: TopKBlock,
    name: str,
) -> tuple[list[pd.DataFrame], list[ColumnBlockMeta]]:
    """Create top K aggregations from DataFrame using regex pattern matching."""
    from_columns = create.from_columns
    assert isinstance(from_columns, str), "from_columns must be str for regex mode"

    regex_from = re.compile(from_columns)
    from_cols = list(filter(lambda s: regex_from.match(s), df.columns))
    # agg_ind is index of the regex group that we want to aggregate over.
    # All other indeces are unique identifiers for each subgroup
    # e.g. ['A'] and ['B'] (that are also added to meta names)
    # Recall that regex group 0 is the whole match.
    # Note that re.Match.groups() does not include the whole match.
    # This means that we need to subtract 1 from agg_ind.
    agg_ind = create.agg_index
    agg_ind = agg_ind - 1 if agg_ind > 0 else agg_ind
    first_match = regex_from.match(from_cols[0])
    assert first_match is not None, f"Column {from_cols[0]} should match regex {regex_from.pattern}"
    n_groups = len(first_match.groups())
    has_subgroups = n_groups >= 2  # Multiple aggregations needed?
    regex_to = create.res_columns if isinstance(create.res_columns, str) else None
    kmax = create.k
    na_vals = create.na_vals if create.na_vals is not None else []

    if has_subgroups:
        # collect all subgroups, later aggregate each subgroup separately
        def _get_subgroup_id(column: str) -> tuple[str, ...]:
            """Discard the aggregation index and collect the rest as identifier."""
            match = regex_from.match(column)
            assert match is not None, f"Column {column} should match regex {regex_from.pattern}"
            subgroup_id = list(match.groups())
            subgroup_id.pop(agg_ind)
            return tuple(subgroup_id)

        subgroups_ids = dict.fromkeys(map(_get_subgroup_id, from_cols))
        subgroups = [
            [col for col in from_cols if _get_subgroup_id(col) == subgroup_id] for subgroup_id in subgroups_ids
        ]
    else:
        subgroups = [from_cols]
    topk_dfs, subgroup_metas = [], []

    # select group at agg_ind in col name to allow translate if spec-d in scale
    # e.g. {A_11: selected} |-> {11: selected}, later by using mask |-> {11: 11}
    # this fun is def-d in current fun, so agg_ind acts as global var
    def _get_regex_group_at_agg_ind(s: str) -> str:
        """Get regex group at aggregation index."""
        match = regex_from.match(s)
        assert match is not None, f"Column {s} should match regex {regex_from.pattern}"
        return match.groups()[agg_ind]

    for subgroup in subgroups:
        sdf = df[subgroup].astype("object").replace(na_vals, None)
        subgroup_block_name = name
        if has_subgroups:
            subgroup_block_name = name + "_" + "_".join(map(str, _get_subgroup_id(subgroup[0])))
        _check_topk_na_vals_after_replace(sdf, block_name=subgroup_block_name)

        def _expand_col(col: str) -> str:
            match = regex_from.match(col)
            assert match is not None, f"Column {col} should match regex {regex_from.pattern}"
            return match.expand(regex_to)  # type: ignore[call-overload]

        newcols = [
            # from_cols names map to res_cols names
            # note regex groups stay the same: e.g A_11 |-> A_R11
            _expand_col(col)
            for col in sdf.columns
        ]

        # Convert one-hot encoded columns into a list-of-selected format
        if "|" in from_columns:
            raise ValueError(sdf.columns)
        sdf.columns = sdf.columns.map(_get_regex_group_at_agg_ind)

        sdf = sdf.mask(
            ~sdf.isna(), other=pd.Series(sdf.columns, index=sdf.columns), axis=1
        )  # replace cell with column name where not NA
        _throw_vals_left(sdf)  # changes df in place, Nones go to rightmost side
        sdf.columns = newcols  # set column names per the regex_to template

        if "|" in from_columns:
            raise NotImplementedError("Regex symbol `|` is not supported yet.")

        sdf = sdf.dropna(axis=1, how="all")  # drop rightmost cols that are all NA
        # Handle kmax: if it's "max" or None, don't limit; otherwise limit to that number
        if kmax and kmax != "max" and isinstance(kmax, int):
            sdf = sdf.iloc[:, :kmax]  # up to kmax columns if spec-d
        sname = name
        if has_subgroups:
            sname += "_" + "_".join(map(str, _get_subgroup_id(subgroup[0])))
        meta_subgroup = {
            "name": sname,
            "scale": deepcopy(block_with_create.scale.model_dump(mode="python") if block_with_create.scale else {}),
            "columns": sdf.columns.tolist(),
        }
        if create.translate_after is not None:
            sdf = sdf.replace(create.translate_after)
        # Apply scale.translate to cell values (e.g. Lithuanian party names → English short codes)
        scale_translate = (
            dict(block_with_create.scale.translate)
            if block_with_create.scale and block_with_create.scale.translate
            else {}
        )
        if scale_translate:
            sdf = sdf.replace(scale_translate)
            effective_cats = list(dict.fromkeys(scale_translate.values()))
            for col in sdf.columns:
                sdf[col] = pd.Categorical(sdf[col], categories=effective_cats)
            meta_subgroup["scale"]["categories"] = effective_cats
        meta_subgroup = soft_validate(meta_subgroup, ColumnBlockMeta)
        topk_dfs.append(sdf)
        subgroup_metas.append(meta_subgroup)
    return topk_dfs, subgroup_metas  # note: each df has one meta for zip later


def _create_maxdiff_metas_and_dfs(
    df: pd.DataFrame,
    group: ColumnBlockMeta,
    topics: Sequence[str] | None = None,
    sets: Sequence[Sequence[int]] | None = None,
    **kwargs: dict[str, str],
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    """
    Create metas and dfs for maxdiff.

    Meta args:
        best_columns (Union[str,List[str]]): Regex template with groups to select df cols from OR just a list of cols.
        worst_columns (Union[str,List[str]]): Regex template with groups to select df cols from OR just a list of cols.
        setindex (int): The index of the column to use as the set index.
        topics (Optional[List[str]]): The topics used in maxdiffs. If `None`, then `constants` is used.
        sets (Optional[List[List[int]]]): The sets of indices per each version. If `None`, then `constants` is used.

    Args:
        df (pd.DataFrame): The dataframe to create metas and dfs for.
        group (dict): The create block in meta with meta args.
        topics (list): The topics used in maxdiffs. Is not none if constants have defined such a variable.
        sets (list): The sets of indices per each version. Is not none if constants have defined such a variable.
            E.g. sets[j][k] = k-th topics subset permutation in j-th version. 1:1 mapping between k and best_columns.

    Returns:
        Tuple[List[dict], List[pd.DataFrame]]: A tuple of 1-list of meta and df with added/translated topics sets.
    """
    df = df.copy(deep=True)
    create = group.create
    assert create is not None and create.type == "maxdiff", "Create block type must be 'maxdiff'"
    topics = create.topics or topics
    sets = create.sets or sets
    best_cols: Sequence[str] | str = create.best_columns
    worst_cols: Sequence[str] | str = create.worst_columns
    set_cols: Sequence[str] | str | None = create.set_columns
    # Parse setindex_column: can be None, str, or [str] or [str, dict]
    setindex_col_name: str | None = None
    setindex_col_meta: ColumnMeta | None = None
    if isinstance(create.setindex_column, str):
        setindex_col_name = create.setindex_column
    elif isinstance(create.setindex_column, list):
        items = list(create.setindex_column)
        setindex_col_name = str(items[0]) if items else None
        if items and len(items) > 1 and isinstance(items[1], dict):
            setindex_col_meta = soft_validate(items[1], ColumnMeta)
    best_is_str = isinstance(best_cols, str)
    set_is_str = isinstance(set_cols, str)
    if set_cols is None:
        raise ValueError("Maxdiff create blocks must define 'set_columns'.")
    if best_is_str != set_is_str:
        raise ValueError(
            "Create args best_cols and set_cols must be of the same type: "
            f"{type(best_cols)} != {type(set_cols)}. "
            f"Got {best_cols} and {set_cols}."
        )
    if best_is_str:
        best_cols_str = cast(str, best_cols)
        worst_cols_str = cast(str, worst_cols)
        best_template = re.compile(best_cols_str)
        worst_template = re.compile(worst_cols_str)
        best_cols = list(filter(lambda col: best_template.match(col), df.columns))
        worst_cols = list(filter(lambda col: worst_template.match(col), df.columns))
        set_cols_pattern = cast(str, set_cols)

        def _expand_set_col(col: str) -> str:
            match = best_template.match(col)
            if match is None:
                raise ValueError(f"Column {col} does not match best_cols pattern {best_cols_str}")
            return match.expand(set_cols_pattern)

        def _get_group_index(s: str) -> int:
            match = best_template.match(s)
            if match is None:
                return 0
            return int(match.group(1))

        set_cols = list(
            map(_expand_set_col, sorted(best_cols, key=_get_group_index))
        )  # order should be based on group indices. might the group not be numeric?
    else:
        best_cols = list(best_cols)
        worst_cols = list(worst_cols)
        set_cols = list(set_cols)

    # Extract translate dict from scale to apply to topic values in the data
    translate = dict(group.scale.translate) if group.scale is not None and group.scale.translate else {}

    def _maybe_json_load(value: str | None) -> list[object] | None:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else None
        except (TypeError, json.JSONDecodeError):
            return None

    def _is_int_like(token: object) -> bool:
        if isinstance(token, (int, np.integer)):
            return True
        if isinstance(token, str):
            stripped = token.strip()
            if not stripped:
                return False
            try:
                int(stripped)
                return True
            except ValueError:
                return False
        return False

    def _tokens_from_value(value: object) -> list[str] | float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return value
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            return [str(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            parsed = _maybe_json_load(stripped) if stripped.startswith("[") and stripped.endswith("]") else None
            if parsed is not None:
                return [str(item) for item in parsed] if isinstance(parsed, list) else None
            return [part.strip() for part in stripped.split(",") if part.strip()]
        raise ValueError(f"Unsupported maxdiff set specification value: {value}")

    def _convert_tokens_to_topics(tokens: list[str] | float | None) -> list[str] | None | float:
        if tokens is None or (isinstance(tokens, float) and pd.isna(tokens)):
            return tokens
        if not isinstance(tokens, list):
            tokens = [str(tokens)]
        if not tokens:
            return []
        if all(isinstance(token, str) for token in tokens) and not all(_is_int_like(token) for token in tokens):
            stripped = [token.strip() for token in tokens]
            return [translate.get(t, t) for t in stripped] if translate else stripped  # type: ignore[return-value]
        if all(_is_int_like(token) for token in tokens):
            if topics is None:
                raise ValueError("Explicit maxdiff set columns with indices require 'topics' to be defined.")
            converted = []
            for token in tokens:
                idx = int(token) if not isinstance(token, (int, np.integer)) else int(token)
                if idx < 1 or idx > len(topics):
                    raise ValueError(f"Maxdiff set index {idx} is out of bounds for topics list of size {len(topics)}.")
                converted.append(topics[idx - 1])
            return converted
        if all(isinstance(token, str) for token in tokens):
            return [token.strip() for token in tokens]
        raise ValueError(f"Unsupported token types in maxdiff set definition: {tokens}")

    ordered_cols = best_cols + worst_cols
    # Compute effective (translated) topics once, used for both setindex columns and best/worst categorical dtypes
    effective_topics: list[str] | None = None
    if topics is not None:
        effective_topics = [translate.get(t, t) for t in topics] if translate else list(topics)  # type: ignore[misc]

    if setindex_col_name:
        df = df[ordered_cols + [setindex_col_name]]
        if topics is None or sets is None or effective_topics is None:
            raise ValueError("Maxdiff definitions using 'setindex_column' must also define 'topics' and 'sets'.")
        topics_arr = np.array(
            ["", *effective_topics], dtype=object
        )  # "" at index 0: survey sets are 1-indexed (subject to change)
        sets_arr = np.asarray(sets, dtype=int)
        lsets = topics_arr[sets_arr]

        setindex = df[setindex_col_name].astype(np.int64).to_numpy() - 1
        selected_sets = lsets[setindex]
        df_setcols = pd.DataFrame(selected_sets.tolist(), columns=set_cols, index=df.index)
        df[set_cols] = df_setcols
    else:
        df = df[ordered_cols + set_cols]
        for col in set_cols:
            converted_values: list[list[str] | None] = []
            for value in df[col].tolist():
                tokens = _tokens_from_value(value)
                converted = _convert_tokens_to_topics(tokens)
                if converted is None or (isinstance(converted, float) and pd.isna(converted)):
                    converted_values.append(None)
                elif isinstance(converted, list):
                    converted_values.append(converted)
                else:
                    converted_values.append(None)
            df[col] = converted_values  # type: ignore[assignment]

    # Apply translate to best/worst scalar topic values, then cast to categorical with full translated topic list.
    # This ensures the column dtype carries only the translated categories, preventing pollution from partial
    # observed values being appended to Lithuanian originals by _fix_meta_categories later.
    for col in best_cols + worst_cols:
        s = df[col]
        if translate:
            s = s.map(lambda x, _t=translate: _t.get(x, x) if isinstance(x, str) else x)
        if effective_topics is not None:
            s = pd.Categorical(s, categories=effective_topics)
        df[col] = s

    generated_name = create.name or f"{group.name}_maxdiff"
    df = df.sort_index(axis=1)  # sort columns

    base_columns = sorted(best_cols + worst_cols + cast(list[str], set_cols))
    # Carry the full translated topic list as categories in the column meta so _fix_meta_categories
    # does not need to infer them from the (potentially partial) dtype.
    best_worst_col_meta = ColumnMeta(categories=effective_topics) if effective_topics is not None else ColumnMeta()
    columns_spec: dict[str, ColumnMeta] = {col: best_worst_col_meta for col in base_columns}
    if setindex_col_name is not None:
        if setindex_col_meta is None:
            setindex_col_meta = ColumnMeta()
        if effective_topics is not None and setindex_col_meta.categories is None:
            # Use translated topic names for the setindex categories
            setindex_col_meta = setindex_col_meta.model_copy(update={"categories": effective_topics})
        if setindex_col_meta.continuous is False:
            setindex_col_meta = setindex_col_meta.model_copy(update={"continuous": True})
        columns_spec = {setindex_col_name: setindex_col_meta} | columns_spec

    return [df], [
        {
            "name": generated_name,
            "scale": deepcopy(group.scale.model_dump(mode="python") if group.scale else {}),
            "columns": columns_spec,
        }
    ]


def _create_topk_metas_and_dfs_list(
    df: pd.DataFrame,
    block_with_create: ColumnBlockMeta,
    create: TopKBlock,
    name: str,
) -> tuple[list[pd.DataFrame], list[ColumnBlockMeta]]:
    """Create top K aggregations from DataFrame using explicit column lists."""
    from_columns = create.from_columns
    assert isinstance(from_columns, list), "from_columns must be list for list mode"

    if not isinstance(create.res_columns, list):
        raise ValueError("res_columns must be a list when from_columns is a list")

    if len(from_columns) != len(create.res_columns):
        raise ValueError(
            f"from_columns ({len(from_columns)}) and res_columns ({len(create.res_columns)}) must have the same length"
        )

    from_cols = from_columns
    res_cols = create.res_columns
    kmax = create.k
    na_vals = create.na_vals if create.na_vals is not None else []
    if create.from_prefix:
        without_prefix = {col: col.removeprefix(create.from_prefix) for col in from_cols}
        from_cols = [without_prefix.get(col, col) for col in from_cols]
        df.columns = [without_prefix.get(col, col) for col in df.columns]

    sdf = df[from_cols].replace(na_vals, None)
    _check_topk_na_vals_after_replace(sdf, block_name=name)

    sdf = sdf.mask(
        ~sdf.isna(), other=pd.Series(sdf.columns, index=sdf.columns), axis=1
    )  # replace cell with column name where not NA

    if create.translate_after is not None:
        sdf = sdf.replace(create.translate_after)
    # Apply scale.translate to cell values (e.g. Lithuanian party names → English short codes)
    scale_translate = (
        dict(block_with_create.scale.translate) if block_with_create.scale and block_with_create.scale.translate else {}
    )
    if scale_translate:
        sdf = sdf.replace(scale_translate)
    sdf.columns = res_cols

    _throw_vals_left(sdf)  # changes df in place, Nones go to rightmost side
    sdf = sdf.dropna(axis=1, how="all")  # drop rightmost cols that are all NA

    # Handle kmax: if it's "max" or None, don't limit; otherwise limit to that number
    if kmax and kmax != "max" and isinstance(kmax, int):
        sdf = sdf.iloc[:, :kmax]  # up to kmax columns if spec-d

    meta_subgroup = {
        "name": name,
        "scale": deepcopy(block_with_create.scale.model_dump(mode="python") if block_with_create.scale else {}),
        "columns": sdf.columns.tolist(),
    }
    meta_subgroup = soft_validate(meta_subgroup, ColumnBlockMeta)

    return [sdf], [meta_subgroup]


CreateBlockModel = TopKBlock | MaxDiffBlock

create_block_type_to_create_fn: dict[str, Callable] = {
    "topk": _create_topk_metas_and_dfs,  # type: ignore[assignment]
    "maxdiff": _create_maxdiff_metas_and_dfs,  # type: ignore[assignment]
}


def _create_new_columns_and_metas(
    df: pd.DataFrame, group: ColumnBlockMeta, **kwargs: dict[str, str]
) -> Iterator[tuple[pd.DataFrame, dict[str, object]]]:
    """Create new columns and metadata from a group definition."""
    if group.create is None:
        raise ValueError("Group must have a create block")
    dfs, metas = create_block_type_to_create_fn[group.create.type](df, group, **kwargs)
    return zip(dfs, metas)
