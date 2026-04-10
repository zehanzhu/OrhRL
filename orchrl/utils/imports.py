from __future__ import annotations

from importlib import import_module
from typing import Any, Callable


def import_callable(import_path: str) -> Callable[..., Any]:
    if ":" in import_path:
        module_name, attr_name = import_path.split(":", 1)
    else:
        module_name, _, attr_name = import_path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"invalid callable import path: {import_path}")
    module = import_module(module_name)
    func = getattr(module, attr_name)
    if not callable(func):
        raise TypeError(f"imported object is not callable: {import_path}")
    return func
