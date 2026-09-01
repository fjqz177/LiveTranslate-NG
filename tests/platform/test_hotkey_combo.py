"""Tests for the platform-independent hotkey combo model."""

import pytest

from livetranslate.platform.hotkeys import HotkeyCombo


class TestParse:
    @pytest.mark.parametrize(
        ("text", "key", "mods"),
        [
            ("Ctrl+Alt+P", "P", frozenset({"ctrl", "alt"})),
            ("ctrl+alt+p", "P", frozenset({"ctrl", "alt"})),
            ("F9", "F9", frozenset()),
            ("Shift+SPACE", "SPACE", frozenset({"shift"})),
            ("Super+H", "H", frozenset({"super"})),
            ("Cmd+Shift+S", "S", frozenset({"super", "shift"})),
        ],
    )
    def test_parse_valid(self, text, key, mods):
        combo = HotkeyCombo.parse(text)
        assert combo.key == key
        assert combo.mods == mods

    @pytest.mark.parametrize(
        "text",
        ["", "+P", "Ctrl+", "Ctrl++P", "Ctrl+Unknown", "Ctrl+P+Q", "Ctrl+Enter+Tab"],
    )
    def test_parse_invalid(self, text):
        with pytest.raises(ValueError):
            HotkeyCombo.parse(text)


class TestFormat:
    def test_canonical_order_is_stable(self):
        combo = HotkeyCombo.parse("Shift+Alt+P")
        assert str(combo) == "Alt+Shift+P"

    def test_modifier_order_follows_ctrl_alt_shift_super(self):
        combo = HotkeyCombo(key="P", mods=frozenset({"super", "alt", "shift", "ctrl"}))
        assert str(combo) == "Ctrl+Alt+Shift+Super+P"

    def test_roundtrip(self):
        for text in ("Ctrl+Alt+P", "Ctrl+Alt+H", "Ctrl+Alt+S", "Ctrl+Alt+C", "F9"):
            assert str(HotkeyCombo.parse(text)) == text


class TestEquality:
    def test_frozen_dataclass_hashes(self):
        a = HotkeyCombo.parse("Ctrl+Alt+P")
        b = HotkeyCombo(key="P", mods=frozenset({"ctrl", "alt"}))
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1
