"""pyappify entry point (LiveTranslate-NG).

The launcher runs this with the managed Python from
``data/apps/LiveTranslate/<...>``. ``sys.path`` differs from a normal source run,
so this shim:

* points ``LIVETRANSLATE_PORTABLE_DIR`` at the persistent ``lt_data`` area — a
  *sibling* of the git checkout. ``update_working_from_repo`` rebuilds
  ``working/`` without touching ``lt_data``, so data survives every git-tag
  update (``paths.py`` already honours this env var as its top-priority root);
* makes the ``src/`` layout importable;
* and hands control to the real entry — ``__main__.py`` is the sole
  import-order owner (apply_cache_env before torch, Windows torch-before-Qt).

The ``--smoke`` flag still works here (it is handled inside ``__main__.py``) so
the CI smoke gate can drive both a source run and the packaged launcher.
"""

import os
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# Live data lives beside the git checkout: <app>/lt_data — NOT inside working/,
# so a git-tag update (which rebuilds working/) never clears settings/models.
os.environ.setdefault("LIVETRANSLATE_PORTABLE_DIR", str(_here.parent / "lt_data"))

sys.path.insert(0, os.path.join(str(_here), "src"))

from livetranslate.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
