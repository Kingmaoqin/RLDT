"""Utilities for importing pandas in environments without working pyarrow."""
from __future__ import annotations

import builtins
import os
import sys
from contextlib import contextmanager
from types import ModuleType
from typing import Iterator


_PANDAS_MODULE_NAME = "pandas"


def _should_demote_pyarrow(exc: AttributeError) -> bool:
    """Return True if the AttributeError stems from an incompatible pyarrow build."""
    message = str(exc)
    return "_ARRAY_API" in message or "pyarrow" in message


@contextmanager
def _pyarrow_as_optional() -> Iterator[None]:
    """Temporarily wrap ``__import__`` so pyarrow build errors look like ImportError."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):  # type: ignore[override]
        try:
            return original_import(name, *args, **kwargs)
        except AttributeError as exc:
            if name.split(".")[0] == "pyarrow" and _should_demote_pyarrow(exc):
                raise ImportError(str(exc)) from None
            raise

    builtins.__import__ = guarded_import
    try:
        yield
    finally:
        builtins.__import__ = original_import


def get_pandas() -> ModuleType:
    """Import pandas while gracefully skipping broken pyarrow wheels."""
    if _PANDAS_MODULE_NAME in sys.modules:
        return sys.modules[_PANDAS_MODULE_NAME]

    # Make sure pandas does not attempt to switch to the Arrow backend automatically.
    os.environ.setdefault("PANDAS_USE_PYARROW_BACKEND", "0")
    os.environ.setdefault("PANDAS_USE_PYARROW_EXTENSION_ARRAY", "0")

    with _pyarrow_as_optional():
        import pandas as pd  # type: ignore import

    try:
        pd.options.mode.dtype_backend = "numpy_nullable"
    except Exception:
        pass

    return pd


__all__ = ["get_pandas"]
