"""Variant recommendation tests (SelfServe P1-B4): driver-version mapping.

No real nvidia-smi runs: the probe functions are mocked to return canned
driver versions.
"""

from __future__ import annotations

import pytest

from livetranslate.core import systeminfo


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("566.14", (566, 14)),
        ("570.86.10", (570, 86, 10)),
        ("garbage", ()),
        ("", ()),
    ],
)
def test_parse_driver_version(raw: str, expected: tuple[int, ...]):
    assert systeminfo._parse_driver_version(raw) == expected


@pytest.mark.parametrize(
    "driver,expected",
    [
        ((570, 86), "cu126"),
        ((566, 14), "cu126"),
        ((555, 42), "cpu"),
        ((560, 0), "cu126"),
        ((), "cpu"),
    ],
)
def test_variant_for_driver(driver: tuple[int, ...], expected: str):
    assert systeminfo._variant_for_driver(driver) == expected


def test_detect_variant_cpu_without_gpu(monkeypatch):
    monkeypatch.setattr(systeminfo, "_driver_version", lambda: ())
    assert systeminfo.detect_variant() == "cpu"


def test_detect_variant_cu126(monkeypatch):
    monkeypatch.setattr(systeminfo, "_driver_version", lambda: (566, 14))
    assert systeminfo.detect_variant() == "cu126"
