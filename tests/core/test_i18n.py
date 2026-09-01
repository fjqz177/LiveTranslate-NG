"""i18n guards: zh/en stay same-source (§3.6 discipline) and the
canonical error/status table is complete in both languages.
"""

from pathlib import Path

import yaml

from livetranslate.core.i18n import get_lang, set_lang, t

ROOT = Path(__file__).resolve().parents[2]

SECTION_36_KEYS = [
    "err_401",
    "err_429",
    "err_conn_refused",
    "err_timeout",
    "err_proxy",
    "err_stream_interrupt",
    "err_download_failed",
    "err_download_stalled",
    "err_download_verify",
    "err_disk_space",
    "err_settings_corrupt",
    "err_save_disk_full",
    "err_hotkey_conflict",
    "err_no_audio_device",
    "err_device_disconnected",
    "err_win_mic_blocked",
    "err_linux_no_monitor",
    "err_engine_failed",
    "err_engine_missing_deps",
    "err_wayland_hotkeys",
    "err_gnome_tray",
    "err_mac_mic",
    "err_mac_screen",
    "err_mac_denied",
    "err_blackhole",
    "err_update_failed",
    "mem_long_run_hint",
    "single_instance_msg",
]


def test_zh_en_key_parity():
    zh = yaml.safe_load((ROOT / "i18n" / "zh.yaml").read_text(encoding="utf-8"))
    en = yaml.safe_load((ROOT / "i18n" / "en.yaml").read_text(encoding="utf-8"))
    assert set(zh) == set(en)


def test_section_36_copy_complete_in_both_languages():
    original = get_lang()
    try:
        for lang in ("zh", "en"):
            set_lang(lang)
            for key in SECTION_36_KEYS:
                assert t(key) != key, f"missing or untranslated: {key} in {lang}"
    finally:
        set_lang(original)
