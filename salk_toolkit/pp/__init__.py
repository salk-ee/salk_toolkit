"""Plot Pipeline
----------------

This is the end-to-end plotting stack that used to live in `02_pp.ipynb`.
It maps annotated survey data to Altair charts by:

- discovering eligible plots via the registry metadata in `salk_toolkit.plots`
- lazily transforming data with Polars, including filters, melts, draws, and
  aggregation helpers such as `pp_transform_data` / `_wrangle_data`
- enriching metadata (`update_data_meta_with_pp_desc`, `impute_facet_dims`,
  `matching_plots`) so dashboards and CLI tools can reason about aliases,
  ordering, draws, and group scales
- building final Altair specs with `create_plot`, colour utilities, tooltips,
  and translation-ready labels
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# These are the only functions that should be exposed to the public
__all__ = [
    "e2e_plot",
    "matching_plots",
    "test_new_plot",
    "cont_transform_options",
    "create_plot",
    "get_plot_fn",
]
# `create_plot_payload` / `UnsupportedPayloadError` are also importable here, lazily via `__getattr__`

from .common import (
    AltairChart as AltairChart,
    FacetMeta as FacetMeta,
    PlotInput as PlotInput,
    special_columns as special_columns,
    _get_cat_num_vals as _get_cat_num_vals,
    _normalize_color_dict as _normalize_color_dict,
)
from .registry import (
    PlotMeta as PlotMeta,
    registry as registry,
    registry_meta as registry_meta,
    stk_plot as stk_plot,
    get_plot_fn as get_plot_fn,
    get_plot_meta as get_plot_meta,
    _ensure_plot_registry_loaded as _ensure_plot_registry_loaded,
    _stk_deregister as _stk_deregister,
)
from .meta import (
    _extract_column_meta_cached as _extract_column_meta_cached,
    _update_data_meta_with_pp_desc as _update_data_meta_with_pp_desc,
)
from .matching import (
    matching_plots as matching_plots,
    impute_facet_dims as impute_facet_dims,
    impute_factor_cols as impute_factor_cols,
    priority_weights as priority_weights,
    _calculate_priority as _calculate_priority,
    _inner_outer_facets as _inner_outer_facets,
    _remove_from_inner_facets as _remove_from_inner_facets,
)
from .transforms import (
    cont_transform_options as cont_transform_options,
    custom_row_transforms as custom_row_transforms,
    _transform_cont as _transform_cont,
)
from .filters import (
    _discretize_continuous as _discretize_continuous,
    _ensure_ldf_categories as _ensure_ldf_categories,
    _pl_quantiles as _pl_quantiles,
    _pp_filter_data as _pp_filter_data,
    _pp_filter_data_lz as _pp_filter_data_lz,
)
from .wrangle import (
    pp_transform_data as pp_transform_data,
    _wrangle_data as _wrangle_data,
)
from .plotting import (
    create_plot as create_plot,
    e2e_plot as e2e_plot,
    publish_spec as publish_spec,
    test_new_plot as test_new_plot,
)

# Lazy re-export from `salk_toolkit.payload` for back-compat (e.g. dms-plots-api imports these from pp)
if TYPE_CHECKING:
    from salk_toolkit.payload import (
        UnsupportedPayloadError as UnsupportedPayloadError,
        create_plot_payload as create_plot_payload,
    )


def __getattr__(name: str) -> object:
    if name in ("create_plot_payload", "UnsupportedPayloadError"):
        import salk_toolkit.payload as _payload

        return getattr(_payload, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
