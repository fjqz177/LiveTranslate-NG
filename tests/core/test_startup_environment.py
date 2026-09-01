import pytest

from livetranslate.core.paths import PROJECT_ROOT


def _read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8").lower()


def test_pyproject_pins_python_below_3_13():
    # 3.13+ rejected: no ctranslate2 cp313 wheels (#15), strict SSL breaks torch.hub (#20)
    p = _read("pyproject.toml")
    assert "requires-python" in p
    assert "<3.13" in p


def test_torch_is_routed_to_the_named_pytorch_index():
    # PyPI's Windows torch wheels are CPU-only; CUDA variants must come from
    # the PyTorch index declared in [tool.uv.sources].
    p = _read("pyproject.toml")
    assert 'name = "pytorch"' in p
    assert "[tool.uv.sources]" in p
    assert 'index = "pytorch"' in p


def test_src_layout_and_entry_point():
    # src layout + console-script entry; the only entry module is
    # src/livetranslate/__main__.py (root must stay module-free).
    assert (PROJECT_ROOT / "src" / "livetranslate" / "__main__.py").exists()
    assert (PROJECT_ROOT / "src" / "livetranslate" / "app.py").exists()
    p = _read("pyproject.toml")
    assert 'pythonpath = ["src"]' in p


def test_legacy_launcher_scripts_are_gone():
    # The bat/ps1 launcher family was replaced by the installed package
    # (uv run livetranslate); a rewrite of update.bat etc. must not resurrect
    # the git-pull update model.
    for name in (
        "start.bat",
        "update.bat",
        "install.bat",
        "install.ps1",
        "build_release.ps1",
    ):
        assert not (PROJECT_ROOT / name).exists(), name
    assert not (PROJECT_ROOT / "release_templates").exists()


def test_server_has_a_console_script():
    server = _read("src/livetranslate_server/pyproject.toml")
    assert "[project.scripts]" in server
    assert 'livetranslate-server = "livetranslate_server.__main__:main"' in server


@pytest.mark.skipif(
    not (PROJECT_ROOT / "README.md").exists() or not (PROJECT_ROOT / "README_en.md").exists(),
    reason="README not present yet (docs deferred)",
)
def test_readmes_reference_the_new_workflow():
    # The user-facing quick-start must describe the installed-package workflow,
    # not the deleted bat launchers. README.md is the Chinese-primary one,
    # README_en.md its English mirror.
    for name in ("README.md", "README_en.md"):
        text = _read(name)
        assert "uv run livetranslate" in text
