"""Doc-vs-code drift guard (M-GUARD, architecture-hardening Milestone 1).

ONE consolidated guard for the honest-delivery invariants that a plain doc
edit keeps breaking. It asserts over the source tree + pyproject (vendor
excluded) that:

1. Every module / package that CLAUDE.md section 8 lists as part of the
   source map actually exists on disk (the map must not document ghosts).
2. No cu128 token remains (the Windows-only convergence ships only cpu /
   cu126; cu128 has no torch wheel).
3. No live mlx-whisper / whisper-cpp / engine-mlx / engine-whispercpp
   engine reference remains (darwin-only dead stubs deleted by the
   Windows-only convergence).
4. No reference resolves to a module / script deleted by M-WINONLY or the
   full-install model (permission_registry / asr.integrity /
   permission_backends / package_macos / package_linux / engine_runtime /
   uv_runner / dependency_dialog / build_runtime_variants).

It deliberately does NOT assert "no sys.platform anywhere": the surviving
uses are intentional (the __main__.py torch gate, the devtools.py
Windows-only package guard, the vad_engine.py platform param, the
diagnostics.py OS display).

Zero third-party dependencies (stdlib only); modeled on
tests/test_architecture.py. The section-8 parser and the token scanners are
unit-tested so a regex / AST blind spot cannot silently disable the guard.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
PACKAGE = SRC / "livetranslate"
VENDOR = PACKAGE / "asr" / "vendor"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"

# Tokens removed by the Windows-only convergence. Kept as tuples so the
# scanners read as a policy table, not ad-hoc literals.
CUDA_FORBIDDEN = ("cu128",)
MLX_FORBIDDEN = ("mlx-whisper", "whisper-cpp", "engine-mlx", "engine-whispercpp")
# Modules / scripts deleted by M-WINONLY or the full-install model
# (runtime engine variant install path); any surviving reference is stale.
DELETED_MODULES = (
    "permission_registry",
    "asr.integrity",
    "permission_backends",
    "package_macos",
    "package_linux",
    # Full-install model (2026-09-01): engines ship with the app (pyappify), so
    # the engine_venv / embedded-uv / variant-requirements path is gone.
    "engine_runtime",
    "uv_runner",
    "dependency_dialog",
    "build_runtime_variants",
)


# ── Source enumeration ────────────────────────────────────────────────────


def _src_files() -> list[Path]:
    """Every .py under src/ (both packages), vendored code excluded."""
    return [
        path for path in sorted(SRC.rglob("*.py")) if not (VENDOR in path.parents or path == VENDOR)
    ]


def _scanned_sources() -> list[tuple[str, str]]:
    """(label, text) for the surface guarded by the token scanners."""
    out = [
        (path.relative_to(SRC).as_posix(), path.read_text(encoding="utf-8"))
        for path in _src_files()
    ]
    out.append(("pyproject.toml", PYPROJECT.read_text(encoding="utf-8")))
    return out


# ── CLAUDE.md §8 parser ───────────────────────────────────────────────────

_SECTION8_RE = re.compile(r"^## 8\.\s", re.MULTILINE)


def _section8_text(claude_md: str) -> str:
    """The §8 block: from '## 8.' up to the next top-level '## ' heading."""
    m = _SECTION8_RE.search(claude_md)
    assert m, "CLAUDE.md must contain a '## 8.' section"
    nxt = re.search(r"^## \d", claude_md[m.end() :], re.MULTILINE)
    end = m.end() + nxt.start() if nxt else len(claude_md)
    return claude_md[m.end() : end]


def _expand_braces(token: str) -> list[str]:
    """Expand one brace-group: 'vad/{x,y}.py' -> ['vad/x.py','vad/y.py']."""
    m = re.search(r"\{([^{}]*)\}", token)
    if not m:
        return [token]
    out: list[str] = []
    for item in m.group(1).split(","):
        out.extend(_expand_braces(token[: m.start()] + item + token[m.end() :]))
    return out


def _section8_refs(section: str) -> tuple[set[str], set[str]]:
    """(file_refs, dir_refs) as src-relative paths for the §8 module map.

    Reads the top-level layer bullets ('-**`core/`**') and the per-layer
    sub-bullets. A backticked token is a module reference when it ends in
    '.py' (file), ends in '/' (package dir), or is a path containing '/'
    (package path). Identifiers / env vars / classes / json filenames are
    not refs and are ignored. The trailing '顶层资源' summary line is not a
    bullet and is skipped."""
    file_refs: set[str] = set()
    dir_refs: set[str] = set()
    current: str | None = None  # src-relative directory prefix, e.g. 'livetranslate/core'

    for raw in section.splitlines():
        stripped = raw.strip()
        if not stripped.startswith("-"):
            continue  # only bullets; skips the '顶层资源' summary paragraph
        mb = re.match(r"-\s*\*\*`([^`]+)`\*\*", stripped)
        if mb:
            token = mb.group(1)
            if token.endswith(".py"):
                # top-level file bullet: livetranslate/__main__.py, app.py, ...
                current = None
                file_refs.add(f"livetranslate/{token}")
            elif token == "livetranslate_server" or token.startswith("livetranslate_server/"):
                current = "livetranslate_server"
                dir_refs.add("livetranslate_server")
            else:
                current = f"livetranslate/{token.rstrip('/')}"
                dir_refs.add(current)
            body = stripped[mb.end() :]  # the layer bullet's own token stays out
        else:
            body = stripped
        for token in re.findall(r"`([^`]+)`", body):
            for variant in _expand_braces(token):
                if not current:
                    continue
                if variant.endswith(".py"):
                    file_refs.add(f"{current}/{variant}")
                elif variant.endswith("/"):
                    dir_refs.add(f"{current}/{variant.rstrip('/')}")
                elif "/" in variant:
                    dir_refs.add(f"{current}/{variant}")
                # else: identifier / env var / class / json — not a module ref.
    return file_refs, dir_refs


# ── Import-symbol collector (for the deleted-module guard) ────────────────


def _module_of(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_of(path: Path) -> str:
    module = _module_of(path)
    if path.name == "__init__.py":
        return module
    return module.rsplit(".", 1)[0]


def _resolve_relative(path: Path, level: int, module: str | None) -> str:
    parts = _package_of(path).split(".")
    up = level - 1
    if up > 0:
        parts = parts[:-up] if up < len(parts) else []
    base = ".".join(parts)
    return f"{base}.{module}" if module else base


def _import_symbols(path: Path):
    """Yield every imported symbol as a dotted name. Unlike a bare module
    scan this also resolves 'from x import y' names, so a name-based import
    of a deleted submodule ('from livetranslate.platform import
    permission_registry') cannot slip past the guard."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            # level > 0: relative import ('from .x import y'); else the module
            # name is already absolute and must NOT be re-prefixed.
            base = _resolve_relative(path, node.level, node.module) if node.level else node.module
            yield base
            for alias in node.names:
                if alias.name != "*":
                    yield f"{base}.{alias.name}"


# ── Guards ────────────────────────────────────────────────────────────────


def test_claude_md_section8_module_map_exists():
    """Every module/package §8 lists must actually exist on disk."""
    text = (PROJECT_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    file_refs, dir_refs = _section8_refs(_section8_text(text))
    assert file_refs, "§8 parser found no module refs (section moved or parser broke)"
    missing_files = [r for r in sorted(file_refs) if not (SRC / r).is_file()]
    missing_dirs = [r for r in sorted(dir_refs) if not (SRC / r).is_dir()]
    assert not missing_files, f"CLAUDE.md §8 lists non-existent modules: {missing_files}"
    assert not missing_dirs, f"CLAUDE.md §8 lists non-existent packages: {missing_dirs}"


def test_no_cu128_token_in_source():
    offenders = [
        label for label, text in _scanned_sources() if any(t in text for t in CUDA_FORBIDDEN)
    ]
    assert not offenders, (
        "cu128 token survived (Windows-only VARIANTS = cpu/cu126; cu128 has no torch wheel): "
        f"{offenders}"
    )


def test_no_live_mlx_or_whisper_cpp_engine_reference():
    offenders = [
        label for label, text in _scanned_sources() if any(t in text for t in MLX_FORBIDDEN)
    ]
    assert not offenders, (
        "mlx-whisper / whisper-cpp engine reference survived "
        "(darwin-only dead stub, deleted by Windows-only convergence): "
        f"{offenders}"
    )


def test_no_reference_to_deleted_windows_only_module():
    offenders: list[tuple[str, list[str]]] = []
    for path in _src_files():
        text = path.read_text(encoding="utf-8")
        text_hits = [d for d in DELETED_MODULES if d in text]
        symbol_hits = [d for d in DELETED_MODULES if any(d in s for s in _import_symbols(path))]
        hits = sorted(set(text_hits) | set(symbol_hits))
        if hits:
            offenders.append((path.relative_to(SRC).as_posix(), hits))
    py = PYPROJECT.read_text(encoding="utf-8")
    if hits := [d for d in DELETED_MODULES if d in py]:
        offenders.append(("pyproject.toml", hits))
    assert not offenders, (
        "reference to a module/script deleted by the Windows-only convergence survived: "
        f"{offenders}"
    )


# ── Helper unit tests (blind-spot guards) ─────────────────────────────────


def test_docs_drift_section8_parser_handles_braces_and_summary():
    section = (
        "## 8.\n"
        "- **`__main__.py`**: entry.\n"
        "- **`core/`**:\n"
        "  - `a.py` A; `vad/{x,y}.py` VAD; `ENV_VAR` `SomeClass` not module.\n"
        "- **`audio/`**:\n"
        "  - `backend.py`; `backends/`: wasapi, null.\n"
        "- **`platform/`**:\n"
        "  - `{hotkey,system}_backends/win32.py`; `window.py`.\n"
        "- **`ui/`**: `panel/tabs/`; `panel.py`; `Foo`/`BAR` skip.\n"
        "top resources: `config.yaml`, `assets/icons/`, `i18n/{zh,en}.yaml`.\n"
    )
    file_refs, dir_refs = _section8_refs(section)
    assert "livetranslate/__main__.py" in file_refs
    assert "livetranslate/core/a.py" in file_refs
    assert "livetranslate/core/vad/x.py" in file_refs
    assert "livetranslate/core/vad/y.py" in file_refs
    assert "livetranslate/audio/backend.py" in file_refs
    assert "livetranslate/platform/hotkey_backends/win32.py" in file_refs
    assert "livetranslate/platform/system_backends/win32.py" in file_refs
    assert "livetranslate/ui/panel/tabs" in dir_refs
    assert "livetranslate/ui/panel.py" in file_refs
    # identifiers / env vars / classes are not module refs
    assert not any(
        "SomeClass" in r or "ENV_VAR" in r or "Foo" in r or "BAR" in r for r in file_refs
    )
    # the summary line's json/assets paths must not leak in as refs
    assert not any(r.startswith("livetranslate/ui/assets") for r in dir_refs)
    assert not any("config.yaml" in r or "zh.yaml" in r for r in file_refs)
    assert file_refs, "parser produced no file refs"
    assert dir_refs, "parser produced no dir refs"


def test_docs_drift_section8_parser_separates_server_layer():
    section = "## 8.\n- **`livetranslate_server/`**: `__main__.py` (FastAPI).\n"
    file_refs, _dir_refs = _section8_refs(section)
    assert "livetranslate_server/__main__.py" in file_refs
    # sanity: the server module is not mis-prefixed under livetranslate/
    assert not any(r.startswith("livetranslate/livetranslate_server") for r in file_refs)


def test_docs_drift_deleted_module_scan_is_sound():
    """The symbol collector must see a name-based import of a deleted
    submodule, not just a dotted module path — otherwise the deleted-module
    guard silently no-ops on the realistic re-introduction vector."""
    src = textwrap.dedent(
        """
        from livetranslate.platform import permission_registry
        from livetranslate.asr.integrity import IntegrityReport
        import livetranslate.platform.permission_backends as pb
        import os
        """
    )
    path = SRC / "livetranslate" / "_synthetic.py"
    path.write_text(src, encoding="utf-8")
    try:
        symbols = list(_import_symbols(path))
    finally:
        path.unlink(missing_ok=True)
    assert "livetranslate.platform.permission_registry" in symbols
    assert "livetranslate.asr.integrity" in symbols
    assert "livetranslate.platform.permission_backends" in symbols
    assert "os" in symbols  # stdlib stdlib import is a legit, non-deleted symbol


def test_docs_drift_token_scanners_are_sound():
    assert any(t in "download.pytorch.org/whl/cu128" for t in CUDA_FORBIDDEN)
    assert not any(t in "download.pytorch.org/whl/cu126" for t in CUDA_FORBIDDEN)
    assert any(t in "engine-whispercpp extra" for t in MLX_FORBIDDEN)
    assert any(t in "from livetranslate import package_macos" for t in DELETED_MODULES)
    assert not any(t in "from livetranslate import package_windows" for t in DELETED_MODULES)
