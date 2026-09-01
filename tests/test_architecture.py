"""Architecture guard: enforce the package layering rules from
docs/development/ARCHITECTURE_PLAN.md so dependency direction cannot
silently rot.

Rules are encoded as import-path allow/deny checks over an AST scan of
every module in livetranslate/ (vendored code excluded) plus the root
entry shim. Relative imports are resolved to dotted names so they cannot
sneak past the layer rules, and the second package (livetranslate_server)
is scanned for isolation. Zero third-party dependencies.
"""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT_ROOT / "src" / "livetranslate"
SERVER = PROJECT_ROOT / "src" / "livetranslate_server"
VENDOR = PACKAGE / "asr" / "vendor"


def _python_files(directory: Path):
    for path in sorted(directory.rglob("*.py")):
        if VENDOR in path.parents or path == VENDOR:
            continue
        yield path


def _module_of(path: Path) -> str:
    """Dotted module name of a source file ('livetranslate.core.paths')."""
    parts = list(path.relative_to(PACKAGE.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_of(path: Path) -> str:
    """Dotted package containing this module. For __init__.py files the
    module *is* the package ('from . import x' targets it directly)."""
    module = _module_of(path)
    if path.name == "__init__.py":
        return module
    return module.rsplit(".", 1)[0]


def _resolve_relative(path: Path, level: int, module: str | None) -> str:
    """Resolve 'from ..pkg import x' to a dotted absolute name."""
    parts = _package_of(path).split(".")
    up = level - 1
    if up > 0:
        parts = parts[:-up] if up < len(parts) else []
    base = ".".join(parts)
    return f"{base}.{module}" if module else base


def _imports_of(path: Path):
    """Yield every imported module name as a dotted absolute name
    (import x.y / from x.y import ... / relative imports resolved)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                yield _resolve_relative(path, node.level, node.module)
            elif node.module:
                yield node.module


def _relative(path: Path) -> str:
    # Package-relative ("livetranslate/core/...") so layer prefixes below
    # don't depend on the src/ layout.
    return path.relative_to(PACKAGE.parent).as_posix()


def _has_import(path: Path, prefix: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for name in _imports_of(path))


# ── Root whitelist ───────────────────────────────────────────────────────

# The pyappify distribution shim (pyappify.yml main_script: "main.py") is the
# one legitimate root module: it bootstraps LIVETRANSLATE_PORTABLE_DIR and
# inserts src/ into sys.path before importing livetranslate. Fixed-name
# whitelist so the guard keeps its teeth against any other stray .py.
_ROOT_ENTRY_SHIMS = {"main.py"}


def test_root_is_module_free():
    # The only entry point is src/livetranslate/__main__.py; the repository
    # root must contain no importable modules at all.
    root_modules = {p.name for p in PROJECT_ROOT.glob("*.py")}
    extra = root_modules - _ROOT_ENTRY_SHIMS
    assert not extra, (
        "the repository root must contain no .py modules except the "
        f"pyappify shim ({', '.join(sorted(_ROOT_ENTRY_SHIMS))}); "
        f"found: {sorted(extra)}"
    )


def test_entry_shim_owns_the_import_order():
    shim = PACKAGE / "__main__.py"
    lines = shim.read_text(encoding="utf-8").splitlines()
    cache_env_line = next(i for i, line in enumerate(lines) if "apply_cache_env()" in line)
    torch_line = next(i for i, line in enumerate(lines) if line.strip().startswith("import torch"))
    assert cache_env_line < torch_line, "apply_cache_env() must run before import torch"
    qt_imports = [n for n in _imports_of(shim) if n.startswith("PyQt6")]
    assert not qt_imports, "the shim must not import Qt; livetranslate.app owns the Qt imports"


def test_entry_shim_freeze_support_runs_first():
    """freeze_support() must run before ANY livetranslate import (CLAUDE.md
    §17): without it the frozen ASR worker child re-executes the whole app,
    trips the single-instance gate and exits 0, leaving the client with a
    bare ASRWorkerExited instead of the real engine error."""
    shim = PACKAGE / "__main__.py"
    lines = shim.read_text(encoding="utf-8").splitlines()
    freeze_lines = [i for i, line in enumerate(lines) if "freeze_support()" in line]
    assert freeze_lines, "the shim must call multiprocessing.freeze_support()"
    app_import_lines = [
        i
        for i, line in enumerate(lines)
        if line.startswith(("from livetranslate.", "import livetranslate."))
    ]
    assert app_import_lines, "sanity: the shim imports the app somewhere"
    assert freeze_lines[0] < app_import_lines[0], (
        "freeze_support() must run before the first livetranslate import"
    )


def test_package_init_files_are_empty():
    empty = [
        _relative(path)
        for path in _python_files(PACKAGE)
        if path.name == "__init__.py" and path.read_text(encoding="utf-8").strip()
    ]
    assert not empty, f"__init__.py files must stay empty (no import cascades): {empty}"


# ── Layer rules ──────────────────────────────────────────────────────────

# PyQt6 must never leak below the UI layer (and the composition root).
QT_FORBIDDEN_PREFIXES = (
    "livetranslate/core",
    "livetranslate/modeling",
    "livetranslate/audio",
    "livetranslate/asr",
)


def test_no_qt_below_the_ui_layer():
    offenders = [
        _relative(p)
        for p in _python_files(PACKAGE)
        if _relative(p).startswith(QT_FORBIDDEN_PREFIXES) and _has_import(p, "PyQt6")
    ]
    assert not offenders, f"PyQt6 must not be imported below livetranslate.ui: {offenders}"


def test_layers_do_not_reach_up():
    # Dotted import prefixes (ARCH-1): _has_import compares dotted names,
    # slash-separated values silently never matched and made this rule dead.
    rules = {
        "livetranslate/core": ("livetranslate.ui", "livetranslate.audio", "livetranslate.asr"),
        "livetranslate/modeling": ("livetranslate.ui", "livetranslate.audio", "livetranslate.asr"),
        "livetranslate/audio": ("livetranslate.ui", "livetranslate.asr", "livetranslate.modeling"),
        "livetranslate/asr": ("livetranslate.ui",),
    }
    offenders = [
        f"{_relative(p)} -> {prefix}"
        for layer, forbidden in rules.items()
        for p in _python_files(PACKAGE)
        if _relative(p).startswith(layer)
        for prefix in forbidden
        if _has_import(p, prefix)
    ]
    assert not offenders, f"upward cross-layer imports: {offenders}"


def test_platform_layer_stays_below_the_ui():
    """platform/ must never reach up into ui/ or the engine backends."""
    offenders = [
        f"{_relative(p)} -> {prefix}"
        for p in _python_files(PACKAGE / "platform")
        for prefix in ("livetranslate.ui", "livetranslate.asr.engines")
        if _has_import(p, prefix)
    ]
    assert not offenders, f"platform layer may not import ui/engines: {offenders}"


def test_only_window_module_may_import_qt():
    """Qt is allowed in platform/window.py only (native window tweaks)."""
    offenders = []
    for p in _python_files(PACKAGE / "platform"):
        if p.name == "window.py":
            continue
        if _has_import(p, "PyQt6"):
            offenders.append(_relative(p))
    assert not offenders, f"only platform/window.py may import Qt: {offenders}"


def test_engines_stay_behind_the_contract():
    """asr/engines may only depend on the ASR layer itself plus the shared
    pure modules (i18n/paths/modeling/vendor) — a whitelist, so anything
    new (e.g. a stray Qt or audio-capture import) fails loudly instead of
    silently widening the contract (ARCH-2: audio.fbank was migrated into
    asr.fbank to keep this rule honest)."""
    allowed = (
        "livetranslate.asr",
        "livetranslate.core.i18n",
        "livetranslate.core.paths",
        # pure-stdlib cross-cutting helper (redact_text/redact_dict) the engines
        # call for log-time path/credential redaction — same tier as core.i18n /
        # core.paths (ARCH-2 contract, updated for a6c0553).
        "livetranslate.core.privacy",
        "livetranslate.modeling",
    )
    offenders = [
        f"{_relative(p)} -> {name}"
        for p in _python_files(PACKAGE / "asr" / "engines")
        for name in _imports_of(p)
        if name.startswith("livetranslate") and not name.startswith(allowed)
    ]
    assert not offenders, f"engine backends must stay behind the contract: {offenders}"


def test_gui_process_never_imports_engine_backends():
    """Only the ASR worker may import asr/engines (GUI must not load models)."""
    offenders = []
    for p in _python_files(PACKAGE):
        rel = _relative(p)
        if rel.startswith("livetranslate/asr/") or not _has_import(p, "livetranslate.asr.engines"):
            continue
        offenders.append(rel)
    assert not offenders, f"only the ASR worker may import engine backends: {offenders}"


def test_engine_venv_injection_lives_only_in_the_worker():
    """Engine-venv sys.path injection (SelfServe P1-B3) stays inside asr/
    (the worker process space; vendored funasr_nano has its own legacy
    insertion). GUI layers may compute/pass pythonpaths but never mutate
    sys.path to load engines."""
    offenders = [
        _relative(p)
        for p in _python_files(PACKAGE)
        if not _relative(p).startswith("livetranslate/asr/")
        and _has_import(p, "sys")
        and "sys.path.insert" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"sys.path.insert must stay inside asr/ (worker space): {offenders}"


def test_composition_root_is_not_imported_by_the_package():
    offenders = [
        _relative(p)
        for p in _python_files(PACKAGE)
        if p.name not in ("app.py", "__main__.py") and _has_import(p, "livetranslate.app")
    ]
    assert not offenders, f"livetranslate.app is the composition root: {offenders}"


# ── Blind-spot guards (ARCH-3) ───────────────────────────────────────────


# Sole exemption: asr/worker.py's env-gated test seam
# (LIVETRANSLATE_TEST_ENGINE_FACTORY) — production code never sets it.
_DYNAMIC_IMPORT_EXEMPT = {"livetranslate/asr/worker.py"}


def test_no_dynamic_import_module():
    """importlib.import_module bypasses every rule in this file. The package
    has no legitimate use for it (availability probes use find_spec); any
    new occurrence needs a deliberate guard extension, not a silent pass."""
    offenders = [
        str(p.relative_to(PROJECT_ROOT))
        for directory in (PACKAGE, SERVER)
        for p in _python_files(directory)
        if _relative(p) not in _DYNAMIC_IMPORT_EXEMPT
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8")))
        if isinstance(node, ast.Attribute) and node.attr == "import_module"
    ]
    assert not offenders, f"importlib.import_module is forbidden (AST blind spot): {offenders}"


def test_server_package_is_isolated():
    """The remote ASR server is a standalone wheel: it must never import
    the GUI package (and the GUI never imports it)."""
    server_offenders = [
        f"{p.relative_to(PROJECT_ROOT)} -> {name}"
        for p in _python_files(SERVER)
        for name in _imports_of(p)
        if name == "livetranslate" or name.startswith("livetranslate.")
    ]
    assert not server_offenders, f"livetranslate_server must stay standalone: {server_offenders}"
    gui_offenders = [
        _relative(p) for p in _python_files(PACKAGE) if _has_import(p, "livetranslate_server")
    ]
    assert not gui_offenders, f"the GUI must not import livetranslate_server: {gui_offenders}"


def test_relative_import_resolution_is_sound():
    """Meta-guard: the relative-import resolver must actually see through
    'from ..x import y' (the historical AST blind spot), otherwise the
    layer rules above silently pass again."""
    assert _resolve_relative(PACKAGE / "asr" / "engines" / "whisper.py", 1, None) == (
        "livetranslate.asr.engines"
    )
    assert _resolve_relative(PACKAGE / "asr" / "engines" / "whisper.py", 2, "protocol") == (
        "livetranslate.asr.protocol"
    )
    assert _resolve_relative(PACKAGE / "asr" / "__init__.py", 1, "worker") == (
        "livetranslate.asr.worker"
    )
