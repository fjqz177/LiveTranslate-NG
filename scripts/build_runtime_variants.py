"""Generate runtime variant requirements (SelfServe P0-A7).

The frozen bundle ships pinned variant requirements so the embedded uv can
install engines on any hardware without re-resolving:

    runtime/requirements/cpu.txt    PyTorch CPU wheels
    runtime/requirements/cu126.txt  PyTorch cu126 wheels (the pyproject default)

Boundary: each file is `uv export --no-dev` output for the engine extras
(engine-funasr + engine-whisper) — the engine venv gets those extras and
their transitive closure, and nothing else. The dev toolchain
([dependency-groups] dev: mypy/ruff/pytest/pre-commit/pyinstaller/pillow/
silero-vad/onnx) is excluded via --no-dev so the embedded uv never installs
build tooling into an end-user engine venv (M-VENV). The base closure from
[project.dependencies] is still structurally carried by uv export — that is
harmless redundancy (the worker subprocess inherits the frozen base env), not
a removed set: engines never re-install it and the export is NOT claimed to
drop it.

Per-variant export contract (M-VENV review round 2):
- cu126 (index_url is None): export with `--frozen` — the committed uv.lock
  already holds the cu126 resolution (pyproject's default pytorch index), so
  a frozen export is deterministic and never mutates the lock.
- cpu (index_url set): export WITHOUT `--frozen` — the lock has no resolution
  for the `--index pytorch=.../cpu` override, and a frozen export would
  silently fall back to the lock's cu126 wheels (torch==...+cu126), making
  cpu.txt incoherent with its own install wheelhouse. The live export with
  the cpu-index override yields the real +cpu wheels. Because that export
  re-resolves, it can rewrite uv.lock in place; the script snapshots uv.lock
  first and restores it afterwards (with a WARNING) so the committed lock is
  never drifted by a regeneration.

Each file is `uv export` output: every package pinned with hashes, plus
per-package `--index-url` lines where uv resolved from the pytorch index.
The engine manager installs with
`uv pip install -r runtime/requirements/<variant>.txt --python <venv python>`
(mirror switching rewrites the index-url lines). Variants that fail to
resolve are skipped — the manager only offers variants whose file exists.
CI regenerates these on every release and diffs them for drift.

Usage:  uv run python scripts/build_runtime_variants.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ_DIR = ROOT / "runtime" / "requirements"

# Keep in lockstep with livetranslate.core.uv_runner.VARIANTS (M-VENV enforces
# the agreement via tests/test_runtime_requirements.py::test_variant_name_sets_are_consistent).
# A variant only exists here if torch publishes that wheel; no higher-CUDA wheel
# is available, so only cpu + the default CUDA slot are exported.
VARIANTS = [
    # (name, pytorch index override; None = pyproject default, cu126)
    ("cpu", "https://download.pytorch.org/whl/cpu"),
    ("cu126", None),
]


def build_variant(name: str, index_url: str | None) -> bool:
    out = REQ_DIR / f"{name}.txt"
    # Per-variant export contract: the cu126 resolution lives in the committed
    # lock (pyproject default pytorch index) so its export is frozen and never
    # mutates the lock. The cpu override (--index pytorch=.../cpu) has NO
    # resolution in the lock — a frozen export would silently export the
    # lock's cu126 wheels (torch==...+cu126) via pyproject's [tool.uv.sources]
    # torch → pytorch index, leaving cpu.txt incoherent with its own install
    # wheelhouse. So the cpu variant exports live with its index override to
    # get the real +cpu wheels.
    frozen = index_url is None
    lock_path = ROOT / "uv.lock"
    lock_before = lock_path.read_bytes() if (not frozen and lock_path.exists()) else None
    cmd = [
        "uv",
        "export",
        "--extra",
        "engine-funasr",
        "--extra",
        "engine-whisper",
        "--format",
        "requirements-txt",
        # The engine venv must contain engine deps ONLY. Without this, uv
        # emits "-e ." for the workspace project itself, and the install
        # resolves it against the calling process's CWD — which is
        # C:\Windows\system32 for Explorer-launched apps ("...does not
        # appear to be a Python project", measured on a user machine).
        "--no-emit-project",
        # [tool.uv] default-groups=["dev"] would otherwise leak the whole dev
        # toolchain (mypy/ruff/pytest/pre-commit/pyinstaller/pillow/
        # silero-vad/onnx) into the shipped engine-venv requirements. The
        # engine venv gets its extras + transitive closure ONLY (M-VENV).
        "--no-dev",
        # Relative output path: the `uv export` header comment echoes the -o
        # value verbatim, and an absolute path would bake the generator's
        # machine-local location into the committed requirements (releases are
        # diffed by raw `git diff --exit-code`, so D:\... style paths break the
        # CI drift gate for zero real drift). cwd is ROOT, so a relative path
        # writes to the same file.
        "-o",
        str(out.relative_to(ROOT)),
    ]
    if frozen:
        # cu126: the locked resolution is exactly the shipped content — frozen
        # export keeps it deterministic. A non-frozen export would re-resolve
        # and may rewrite uv.lock (silently bumping versions, e.g. the measured
        # cuda-pathfinder 1.6.0 -> 1.8.0 rewrite).
        cmd.append("--frozen")
    if index_url:
        cmd += ["--index", f"pytorch={index_url}"]
    cmd += ["--index-strategy", "unsafe-best-match"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace")
    # The live cpu export re-resolves and can rewrite uv.lock in place (it
    # bumps whatever newer versions the index offers). The committed lock must
    # stay byte-identical — restore the snapshot and make the mutation loud.
    if lock_before is not None and (
        not lock_path.exists() or lock_path.read_bytes() != lock_before
    ):
        lock_path.write_bytes(lock_before)
        print(f"[warn] {name}: uv export rewrote uv.lock; restored the committed lock")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        print(f"[skip] {name}: export failed ({'; '.join(tail)})")
        out.unlink(missing_ok=True)
        return False
    text = out.read_text(encoding="utf-8")
    if re.search(r"^\s*-e\s", text, flags=re.MULTILINE):
        print(f"[skip] {name}: export leaked an editable entry (-e .)")
        out.unlink(missing_ok=True)
        return False
    lines = text.count("\n")
    print(f"[ok]   {name}: {out.relative_to(ROOT)} ({lines} lines)")
    return True


def main() -> int:
    # Local packaging is offline-first: if the variant requirements are already
    # on disk (they are committed), reuse them instead of re-exporting. The cpu
    # variant needs a live export against download.pytorch.org/whl/cpu, which
    # stalls on a slow network. CI (release.yml drift gate) forces a regenerate
    # with --force and diffs the output for drift.
    force = "--force" in sys.argv
    REQ_DIR.mkdir(parents=True, exist_ok=True)
    if not force:
        existing = [REQ_DIR / f"{name}.txt" for name, _ in VARIANTS]
        if all(path.exists() for path in existing):
            print(
                "[skip] variant requirements already present (cpu.txt, cu126.txt) — "
                "reusing them for an offline local package. If you changed project "
                "deps, regenerate with: "
                "uv run python scripts/build_runtime_variants.py --force"
            )
            return 0
    ok = True
    for name, index_url in VARIANTS:
        ok = build_variant(name, index_url) and ok
    if not ok:
        print(
            "WARNING: one or more variants were skipped. The engine manager "
            "only offers variants whose requirements file exists."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
