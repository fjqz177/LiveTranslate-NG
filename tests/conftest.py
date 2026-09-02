"""Test-wide fixtures.

paths.py computes its module-level constants (DATA_ROOT / SETTINGS_FILE / LOG_DIR)
at import time. test_frozen_paths.py and test_settings.py reload that module under
a monkeypatched runtime state; the reloaded constants OUTLIVE the monkeypatch
teardown, so a later test that imports ``livetranslate.core.paths`` sees a
stale/polluted view (data root pointing at a fake path, or platformdirs) and the
suite becomes order-dependent — a real flake under ``pytest -k <subset>``,
``pytest-xdist``, or random ordering.

This autouse fixture reloads ``core.paths`` back to the *dev* default after
every test, guaranteeing order-independence regardless of which test ran last
(ARCH M3).
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys

import pytest

import livetranslate.core.paths as _paths

# Env vars that reroute the data root at import time (see core/paths.py).
_PATH_ENV = (
    "LIVETRANSLATE_PORTABLE_DIR",
    "LIVETRANSLATE_PORTABLE",
    "LIVETRANSLATE_PLATFORM_DIRS",
)


@pytest.fixture(autouse=True)
def _restore_paths_to_dev_default():
    """Reload core.paths to the dev default after every test so the reload side
    effects of test_frozen_paths/test_settings cannot bleed into later tests.

    Snapshot the prior process state and restore it, so a host/CI-set
    LIVETRANSLATE_PORTABLE_DIR (the documented smoke-gate override) or the
    PLATFORM_DIRS=1 escape hatch is NOT destroyed for the rest of the run —
    unconditionally popping them would make a ``pytest -k`` subset order-
    dependent, the inverse of this fixture's stated goal."""
    prev_env = {var: os.environ.get(var) for var in _PATH_ENV}
    had_frozen = hasattr(sys, "frozen")
    prev_frozen = getattr(sys, "frozen", None)
    yield
    for var, val in prev_env.items():
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val
    if had_frozen:
        sys.frozen = prev_frozen
    else:
        with contextlib.suppress(AttributeError):
            delattr(sys, "frozen")
    importlib.invalidate_caches()
    importlib.reload(_paths)
