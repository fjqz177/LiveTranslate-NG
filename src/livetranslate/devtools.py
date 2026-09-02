"""One-command dev workflows via uv (no extra tools, no justfile).

    uv run livetranslate-pr                # full quality gate (authoritative)
    uv run livetranslate-pr --smoke        # gate + frozen-style --smoke
    uv run livetranslate-pr --git-audit    # gate + .gitignore self-check
    uv run livetranslate-check             # alias of livetranslate-pr (back-compat)

Everything else is plain uv:

    uv sync --extra engine-funasr --extra engine-whisper   # setup (NVIDIA; CPU variant in README)
    uv run livetranslate                                   # dev run
    uv run livetranslate --smoke                           # dev smoke (offscreen, auto-quit)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# src/livetranslate/devtools.py -> repository root
ROOT = Path(__file__).resolve().parents[2]

# Quality gate steps, in commit order. Every step carries its own runner so it
# works whether or not the caller has an activated venv: `uv run <tool>` for
# CLI tools, plain `uv lock --check` for the lockfile consistency check.
_GATE: list[list[str]] = [
    ["uv", "lock", "--check"],
    ["uv", "run", "ruff", "check", "."],
    ["uv", "run", "ruff", "format", "--check", "."],
    ["uv", "run", "mypy", "src"],
    ["uv", "run", "pytest", "tests/"],
]


def _run(argv: list[str]) -> int:
    print(f"$ {' '.join(argv)}")
    try:
        proc = subprocess.run(argv, cwd=ROOT)
    except FileNotFoundError as exc:
        print(f"FAILED: {argv[0]} not found ({exc})")
        return 127
    return proc.returncode


def _git_audit() -> int:
    """Stage-consistency check: anything about to be committed that .gitignore
    was meant to ignore (data/, coverage, runtime noise, ...) is reported and
    fails the gate. Catches the "runtime data snuck into git add -A" class of
    mistake before it lands, instead of after.
    """
    print("$ git audit")
    proc = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    if proc.returncode != 0:
        print("  git status failed — skipped.")
        return 0
    offenders: list[str] = []
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        # "XY path" porcelain; take the path (after the two status chars + space).
        path = line[3:].strip().strip('"')
        if not path:
            continue
        # Ignored-but-staged? git check-ignore reports paths matching .gitignore.
        check = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT, capture_output=True)
        if check.returncode == 0:
            offenders.append(path)
    if offenders:
        print("  Detected files that should be gitignored but are tracked/staged:")
        for p in offenders:
            print(f"    - {p}")
        print("  Fix: add them to .gitignore or `git rm --cached` (they are tracked).")
        return 1
    print("  OK: no gitignore-missed files.")
    return 0


def gate_main() -> int:
    """Authoritative quality gate — the single entry point for humans, agents
    and CI. Fail-fast on the first red step; returns the step's exit code.
    """
    steps = list(_GATE)
    if "--smoke" in sys.argv:
        steps.append(["uv", "run", "livetranslate", "--smoke"])
    if "--git-audit" in sys.argv:
        code = _git_audit()
        if code != 0:
            return code
    for step in steps:
        code = _run(step)
        if code != 0:
            print(f"\nFAILED: {' '.join(step)} (exit {code})")
            return code
    print("\nAll checks passed.")
    return 0


# Back-compat alias: the original name for the gate entry point (kept so a
# CI/blob calling `livetranslate-check` still means exactly the same gate).
check_main = gate_main


if __name__ == "__main__":
    sys.exit(gate_main())
