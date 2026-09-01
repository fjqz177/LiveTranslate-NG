import locale
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]  # PyYAML ships no type stubs here

from livetranslate.core.paths import PROJECT_ROOT

_strings: dict[str, Any] = {}
_lang: str = "en"
_dir: Path = PROJECT_ROOT / "i18n"


def _detect_system_lang() -> str:
    """Return 'zh' if system locale is Chinese, else 'en'."""
    try:
        lang_code = locale.getdefaultlocale()[0] or ""
        if lang_code.startswith("zh"):
            return "zh"
    except Exception:
        pass
    return "en"


def set_lang(lang: str) -> None:
    global _lang, _strings
    _lang = lang
    f = _dir / f"{lang}.yaml"
    if not f.exists():
        f = _dir / "en.yaml"
    _strings = yaml.safe_load(f.read_text("utf-8")) or {}


def get_lang() -> str:
    return _lang


def resolve_ui_lang(value: str | None) -> str:
    """Map a stored ui_lang preference ('system'/'zh'/'en'/None) to a
    concrete language code understood by set_lang()."""
    if value in ("zh", "en"):
        return value
    return _detect_system_lang()


def t(key: str) -> str:
    return cast("str", _strings.get(key, key))


# Detect system language on import
set_lang(_detect_system_lang())

# Shared language list: (code, native_name)
LANGUAGES = [
    ("auto", None),  # display name comes from t("asr_lang_auto")
    ("ja", "日本語"),
    ("en", "English"),
    ("zh", "中文"),
    ("ko", "한국어"),
    ("fr", "Français"),
    ("de", "Deutsch"),
    ("es", "Español"),
    ("ru", "Русский"),
    ("pt", "Português"),
    ("it", "Italiano"),
    ("nl", "Nederlands"),
    ("pl", "Polski"),
    ("tr", "Türkçe"),
    ("ar", "العربية"),
    ("th", "ไทย"),
    ("vi", "Tiếng Việt"),
    ("id", "Bahasa Indonesia"),
    ("ms", "Bahasa Melayu"),
    ("hi", "हिन्दी"),
    ("uk", "Українська"),
    ("cs", "Čeština"),
    ("ro", "Română"),
    ("el", "Ελληνικά"),
    ("hu", "Magyar"),
    ("sv", "Svenska"),
    ("da", "Dansk"),
    ("fi", "Suomi"),
    ("no", "Norsk"),
    ("he", "עברית"),
]

# Common languages shown directly in tray menu (no submenu)
COMMON_LANG_CODES = {"auto", "ja", "en", "zh", "ko", "fr", "de", "es", "ru"}

# English display names for ASR/translation language codes. Moved here from
# translator.py so ASR backends can resolve language names without depending
# on the translation module.
LANGUAGE_DISPLAY = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "hi": "Hindi",
    "uk": "Ukrainian",
    "cs": "Czech",
    "ro": "Romanian",
    "el": "Greek",
    "hu": "Hungarian",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "he": "Hebrew",
}
