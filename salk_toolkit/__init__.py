"""salk_toolkit public API proxy.

Historically this module re-exported everything via star-imports.  We now load
the submodules explicitly and forward the symbols listed in their ``__all__`` to
keep backwards compatibility while satisfying lint rules.
"""

from __future__ import annotations

from typing import Dict

from . import io as _io
from . import pp as _pp
from . import utils as _utils
from .election_models import simulate_election, simulate_election_e2e

__all__ = [  # pyright: ignore[reportUnsupportedDunderAll]
    *getattr(_utils, "__all__", []),
    *getattr(_io, "__all__", []),
    *getattr(_pp, "__all__", []),
    "simulate_election",
    "simulate_election_e2e",
]


def _export(module: object, namespace: Dict[str, object]) -> None:
    """Export all symbols from a module's __all__ into the given namespace.

    Args:
        module: Module object to export from.
        namespace: Dictionary (typically globals()) to populate with exported symbols.
    """
    for name in getattr(module, "__all__", []):
        namespace[name] = getattr(module, name)


_export(_utils, globals())
_export(_io, globals())
_export(_pp, globals())
