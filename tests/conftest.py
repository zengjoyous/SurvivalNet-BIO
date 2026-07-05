"""Test configuration helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = ROOT / "scripts" / "_bootstrap.py"


def _load_bootstrap_module():
    spec = importlib.util.spec_from_file_location("_bootstrap", BOOTSTRAP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load bootstrap helper from {BOOTSTRAP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("_bootstrap", _load_bootstrap_module())
