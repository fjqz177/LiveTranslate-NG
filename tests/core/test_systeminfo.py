"""Tests for torch-free accelerator detection (mocked probes)."""

from livetranslate.core import systeminfo
from livetranslate.platform.system import AcceleratorInfo


class TestDetectAccelerator:
    def test_windows_without_cuda(self, monkeypatch):
        monkeypatch.setattr(systeminfo, "_cuda_windows", lambda: (False, ""))
        info = systeminfo.detect_accelerator()
        assert info.kind == "cpu"

    def test_windows_with_cuda(self, monkeypatch):
        monkeypatch.setattr(systeminfo, "_cuda_windows", lambda: (True, "RTX 4090"))
        info = systeminfo.detect_accelerator()
        assert info.kind == "cuda"
        assert info.device_name == "RTX 4090"


class TestAcceleratorInfoDisplay:
    def test_display_names(self):
        assert AcceleratorInfo("cuda", "RTX 4090").display == "RTX 4090"
        assert AcceleratorInfo("cuda").display == "NVIDIA GPU (CUDA)"
        assert AcceleratorInfo("cpu").display == "CPU"


def test_detection_never_raises_without_torch():
    # Real probe on this machine: must return something without torch being
    # imported (the test environment has torch, but this module never uses it).
    info = systeminfo.detect_accelerator()
    assert info.kind in ("cuda", "cpu")
