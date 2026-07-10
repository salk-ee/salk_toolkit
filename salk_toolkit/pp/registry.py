"""Plot registry: the ``@stk_plot`` decorator and plot metadata lookups."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, cast

import pandas as pd

import salk_toolkit.utils as utils
from salk_toolkit.validation import DF, ColumnMeta, GroupOrColumnMeta, PBase, soft_validate

from .common import AltairChart, FacetMeta, PlotInput


class PlotMeta(PBase):
    """Metadata registered for each plot function via ``@stk_plot``."""

    name: str
    data_format: Literal["longform", "raw"] = "longform"
    draws: bool = False
    continuous: bool = False
    n_facets: Optional[Tuple[int, int]] = None
    requires: List[Dict[str, Any]] = DF(list)
    requires_factor: bool = False
    no_question_facet: bool = False
    agg_fn: Optional[str] = None
    sample: Optional[int] = None
    group_sizes: bool = False
    sort_numeric_first_facet: bool = False
    no_faceting: bool = False
    factor_columns: int = 1
    aspect_ratio: Optional[float] = None
    as_is: bool = False
    priority: int = 0
    args: Dict[str, Any] = DF(dict)
    hidden: bool = False
    transform_fn: Optional[str] = None
    nonnegative: bool = False


registry: Dict[str, Callable[..., Any]] = {}
registry_meta: Dict[str, PlotMeta] = {}
_registry_bootstrapped = False

stk_plot_defaults = {"data_format": "longform"}


def _ensure_plot_registry_loaded() -> None:
    """Import the plots module lazily to populate the registry."""
    global _registry_bootstrapped
    if _registry_bootstrapped:
        return
    try:
        import salk_toolkit.plots  # noqa: F401
    except Exception as exc:  # pragma: no cover
        utils.warn(f"Plot registry bootstrap failed: {exc}")
    else:
        _registry_bootstrapped = True


def _ensure_plot_args_sync(func: Callable[..., Any], decorator_kwargs: Dict[str, Any]) -> None:
    """Verify that declared args (including pass-through requirements) match the function signature.

    Used as part of the decorator for registering a plot type with metadata.
    """

    args_dict = decorator_kwargs.get("args")
    declared_args = set(cast(dict[str, object], args_dict if args_dict is not None else {}).keys())
    requires = cast(list[dict[str, object]], decorator_kwargs.get("requires") or [])
    pass_args = {k for req in requires for k, v in cast(dict[str, object], req).items() if v == "pass"}

    expected = declared_args | pass_args
    if not expected:
        return

    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    extra_params = [p for p in params[1:] if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    seen = {p.name for p in extra_params}
    if seen != expected:
        raise ValueError(
            f"Plot '{func.__name__}' signature args {sorted(seen)} do not match declared args {sorted(expected)}"
        )


def stk_plot(plot_name: str, **r_kwargs: object) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Register a plotting function inside the global plot registry."""

    def _decorator(gfunc: Callable[..., object]) -> Callable[..., object]:
        _ensure_plot_args_sync(gfunc, r_kwargs)
        registry[plot_name] = gfunc
        meta_payload = {"name": plot_name, **stk_plot_defaults, **r_kwargs}
        registry_meta[plot_name] = PlotMeta.model_validate(meta_payload)

        return gfunc

    return _decorator


def _stk_deregister(plot_name: str) -> None:
    """Remove a plot from the registry (used in tests)."""

    del registry[plot_name]
    del registry_meta[plot_name]


def _get_plot_fn(plot_name: str) -> Callable[..., Any]:
    """Retrieve a registered plot function by name."""

    _ensure_plot_registry_loaded()
    return registry[plot_name]


# External legacy plot builder callable for Streamlit tools.
def get_plot_fn(plot_name: str) -> Callable[..., AltairChart]:
    """Return a legacy plot builder callable for Streamlit tools.

    Historically, `salk_internal_package` tools called `stk` plots directly via:
    `get_plot_fn("matrix")(**pparams)`, where `pparams` was a dict containing
    keys like `data`, `facets`, and `value_col`.

    The plot pipeline was refactored to pass a `PlotInput` object into plot
    functions instead. This helper restores the old calling convention by
    wrapping the registered plot function.

    Args:
        plot_name: Registry name of the plot to retrieve (e.g. "matrix", "density").

    Returns:
        Callable that accepts legacy `pparams` keyword arguments and returns an Altair chart.
    """

    plot_fn = _get_plot_fn(plot_name)
    sig = inspect.signature(plot_fn)
    params = list(sig.parameters.keys())
    first_param = params[0] if params else None
    plot_param_names = {p for p in params if p != first_param}

    def _facet(raw: object) -> FacetMeta:
        if isinstance(raw, FacetMeta):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("Facet definitions must be dicts or FacetMeta instances")
        col = raw.get("col")
        if not isinstance(col, str) or not col:
            raise ValueError("Facet dict must contain non-empty 'col'")
        meta_raw = raw.get("meta")
        return FacetMeta(
            col=col,
            ocol=cast(str, raw.get("ocol", col)),
            order=[str(x) for x in cast(Sequence[Any], raw.get("order", []))],
            colors=raw.get("colors"),
            neutrals=[str(x) for x in cast(Sequence[Any], raw.get("neutrals", []))],
            meta=soft_validate(meta_raw, ColumnMeta) if meta_raw is not None else ColumnMeta(),
        )

    def _legacy_callable(**pparams: object) -> AltairChart:
        # Split PlotInput-ish keys vs plot-function kwargs.
        plot_kwargs = {k: v for k, v in dict(pparams).items() if k in plot_param_names}

        data = pparams.get("data")
        if data is None:
            raise ValueError("Legacy plot call requires 'data'")
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)  # type: ignore[arg-type]

        facets_raw = pparams.get("facets", [])
        facets_list = [_facet(f) for f in cast(Sequence[Any], facets_raw)] if facets_raw else []

        value_col = pparams.get("value_col")
        if not isinstance(value_col, str) or not value_col:
            raise ValueError("Legacy plot call requires non-empty 'value_col'")

        # pyright: ignore[reportCallIssue] - legacy dict-based API intentionally coerces loosely typed inputs.
        pi = PlotInput(  # type: ignore[call-arg]
            data=data,
            col_meta=cast(Dict[str, GroupOrColumnMeta], pparams.get("col_meta") or {}),
            value_col=value_col,
            cat_col=cast(Optional[str], pparams.get("cat_col")),
            val_format=cast(str, pparams.get("val_format") or "%"),
            val_range=cast(Optional[Tuple[Optional[float], Optional[float]]], pparams.get("val_range")),
            filtered_size=float(cast(object, pparams.get("filtered_size") or 0.0)),
            facets=facets_list,
            tooltip=cast(List[Any], pparams.get("tooltip") or []),
            value_range=cast(Optional[Tuple[float, float]], pparams.get("value_range")),
            outer_colors=cast(Dict[str, Any], pparams.get("outer_colors") or {}),
            width=int(cast(object, pparams.get("width") or 800)),
            alt_properties=cast(Dict[str, Any], pparams.get("alt_properties") or {}),
            outer_factors=cast(List[str], pparams.get("outer_factors") or []),
            plot_args=cast(Dict[str, Any], pparams.get("plot_args") or {}),
        )

        return cast(AltairChart, plot_fn(pi, **plot_kwargs))

    return _legacy_callable


def get_plot_meta(plot_name: str) -> PlotMeta | None:
    """Return the registry metadata entry for ``plot_name``."""

    _ensure_plot_registry_loaded()
    if plot_name not in registry_meta:
        return None
    return registry_meta[plot_name].model_copy(deep=True)


def _get_all_plots() -> List[str]:
    """List registered plot names in alphabetical order."""

    _ensure_plot_registry_loaded()
    return sorted(list(registry.keys()))
