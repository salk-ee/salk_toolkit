"""Plot registry: the ``@stk_plot`` decorator and plot metadata lookups."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, cast


import salk_toolkit.utils as utils
from salk_toolkit.validation import DF, PBase

from .common import AltairChart


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
        registry_meta[plot_name] = PlotMeta.model_validate({"name": plot_name, **r_kwargs})

        return gfunc

    return _decorator


def _stk_deregister(plot_name: str) -> None:
    """Remove a plot from the registry (used in tests)."""

    del registry[plot_name]
    del registry_meta[plot_name]


def get_plot_fn(plot_name: str) -> Callable[..., AltairChart]:
    """Return the registered plot function.

    Plot functions take a ``PlotInput`` as their first argument, followed by
    plot-specific keyword arguments, e.g. ``get_plot_fn("matrix")(pi, log_colors=True)``.
    """

    _ensure_plot_registry_loaded()
    return registry[plot_name]


def get_plot_meta(plot_name: str) -> PlotMeta | None:
    """Return the registry metadata entry for ``plot_name``."""

    _ensure_plot_registry_loaded()
    if plot_name not in registry_meta:
        return None
    return registry_meta[plot_name].model_copy(deep=True)
