"""devtools gate surface tests (workflow discipline Phase W1).

Assert the authoritative gate shape without running the heavy tools: the
compile-time list must cover the four quality gates plus the lockfile check,
each invoked through `uv run` (or `uv lock`) so an agent without an activated
venv still executes them.
"""

from livetranslate.devtools import _GATE, check_main, gate_main


def test_gate_contains_all_steps():
    steps = " ".join(" ".join(s) for s in _GATE)
    for tool in ("lock --check", "ruff check", "ruff format --check", "mypy", "pytest"):
        assert tool in steps, f"gate missing step: {tool}"


def test_gate_steps_use_uv_runner():
    # Every step must be runnable regardless of venv activation.
    for step in _GATE:
        assert step[0] == "uv", f"step not uv-runnable: {step}"
        # lock check is bare `uv lock`; the rest go through `uv run <tool>`.
        if "lock" in step[1]:
            assert step[1] == "lock", f"lock step shape: {step}"
        else:
            assert step[1] == "run", f"expected uv run prefix: {step}"


def test_check_main_is_gate_alias():
    assert check_main is gate_main


def test_gate_order_fail_fast():
    # ruff (lint) precedes pytest; a code error surfaces before the slow suite.
    joined = [" ".join(s) for s in _GATE]
    assert joined.index("uv run ruff check .") < joined.index("uv run pytest tests/")
