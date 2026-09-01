"""Settings and monitoring panel.

ControlPanel owns the SettingsStore lifecycle, the per-page widgets and the
cross-page signals. Navigation follows plan §3.2: a 200px left list plus a
stacked content area with seven pages (常规/翻译/识别/字幕/数据与存储/诊断/
关于); the benchmark moved into an independent tool window opened from the
recognition page. Shared state is reached from pages via
TabBase.settings / TabBase.config, and collected back here through each
page's collect() when settings are applied.

Layout rules (plan §3.5, applied after the redesign):
- every tall page scrolls instead of pushing the window past the screen;
- each page opens with a title + one-line hint so content is self-explaining;
- the nav list switches the stack (``_on_nav_changed`` wires both the page
  switch and the per-page side effects — a missing switch here once froze
  the whole panel on page 0).
"""

import logging

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from livetranslate.core.i18n import t
from livetranslate.core.privacy import redact_settings
from livetranslate.core.settings import SETTINGS_FILE
from livetranslate.modeling.manager import (
    DEFAULT_FUNASR_MODEL,
    normalize_funasr_model_key,
)
from livetranslate.ui.panel._chrome import DEFAULT_THEME, THEME_MODES, apply_app_theme
from livetranslate.ui.panel._tab_base import make_scroll_page, page_header
from livetranslate.ui.panel.tabs.about_tab import AboutTab
from livetranslate.ui.panel.tabs.benchmark_dialog import BenchmarkDialog
from livetranslate.ui.panel.tabs.cache_tab import CacheTab
from livetranslate.ui.panel.tabs.diagnostics_tab import DiagnosticsTab
from livetranslate.ui.panel.tabs.general_tab import GeneralTab
from livetranslate.ui.panel.tabs.hotkeys_group import HotkeysGroup
from livetranslate.ui.panel.tabs.style_tab import StyleTab
from livetranslate.ui.panel.tabs.subtitle_tab import SubtitleTab
from livetranslate.ui.panel.tabs.translation_tab import TranslationTab
from livetranslate.ui.panel.tabs.vad_tab import VadTab
from livetranslate.ui.settings_bridge import SettingsStore

log = logging.getLogger("LiveTranslate.Panel")

# Theme chrome (light/dark QSS + safety-net palettes) lives in
# ui/panel/_chrome.py — one source of truth for both modes.
PANEL_PALETTE = None  # removed: palettes come from _chrome.build_palette


class ControlPanel(QWidget):
    """Settings and monitoring panel."""

    settings_changed = pyqtSignal(dict)
    model_changed = pyqtSignal(dict)
    models_list_changed = pyqtSignal(list, int)
    subtitle_settings_changed = pyqtSignal(dict)
    asr_language_changed = pyqtSignal(str)
    hotkeys_changed = pyqtSignal(dict)
    _bench_result = pyqtSignal(str)
    _cache_result = pyqtSignal(list)
    reset_positions = pyqtSignal()

    def __init__(
        self, config, saved_settings=None, recommended_engine=None, recommended_device=None
    ):
        super().__init__()
        self._config = config
        self._app = None  # attached by app.py; feeds the diagnostics page
        self.setWindowTitle(t("window_control_panel"))
        self.setMinimumSize(760, 560)
        self.resize(920, 660)
        # Plain QWidgets don't paint stylesheet backgrounds by default;
        # the panel root needs it so the window chrome area matches the
        # themed surfaces instead of the platform default.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._theme_mode = DEFAULT_THEME

        # SettingsStore is the single source of truth; _current_settings is
        # a panel-local working copy (draft): tabs mutate it in place and it
        # is committed atomically on save. The store dict is never handed out.
        self._store = SettingsStore(SETTINGS_FILE, parent=self)
        if saved_settings is not None:
            saved = self._store.seed(saved_settings)
        else:
            saved = self._store.load()
        if saved:
            self._current_settings = dict(saved)  # draft copy
        else:
            tc = config["translation"]
            # UI-3: first-run fallbacks come from the composition root's
            # recommend_engine() (platform + detected accelerator) — never a
            # hardcoded cuda that fails on machines without an NVIDIA GPU.
            self._current_settings = {
                "vad_mode": "silero",
                "vad_threshold": config["asr"]["vad_threshold"],
                "energy_threshold": 0.02,
                "min_speech_duration": config["asr"]["min_speech_duration"],
                "max_speech_duration": config["asr"]["max_speech_duration"],
                "silence_mode": "auto",
                "silence_duration": 0.8,
                "asr_language": config["asr"].get("language", "auto"),
                "asr_engine": recommended_engine or "sensevoice-onnx",
                "funasr_model": config["asr"].get("funasr_model", DEFAULT_FUNASR_MODEL),
                "asr_device": recommended_device or "cpu",
                "sensevoice_pad_seconds": config["asr"].get("sensevoice_pad_seconds", 0.5),
                "whisper_pad_seconds": config["asr"].get("whisper_pad_seconds", 0.5),
                "models": (
                    [
                        {
                            "name": f"{tc['model']}",
                            "api_base": tc["api_base"],
                            "api_key": tc["api_key"],
                            "model": tc["model"],
                        }
                    ]
                    if tc.get("api_key")
                    else []
                ),
                "active_model": 0,
                "hub": "ms",
            }

        if "models" not in self._current_settings and config["translation"].get("api_key"):
            tc = config["translation"]
            self._current_settings["models"] = [
                {
                    "name": f"{tc['model']}",
                    "api_base": tc["api_base"],
                    "api_key": tc["api_key"],
                    "model": tc["model"],
                }
            ]
            self._current_settings["active_model"] = 0

        self._current_settings.setdefault(
            "funasr_model",
            config["asr"].get("funasr_model", DEFAULT_FUNASR_MODEL),
        )
        self._current_settings["funasr_model"] = normalize_funasr_model_key(
            self._current_settings.get("funasr_model")
        )
        self._current_settings.setdefault(
            "sensevoice_pad_seconds",
            config["asr"].get("sensevoice_pad_seconds", 0.5),
        )
        self._current_settings.setdefault(
            "whisper_pad_seconds",
            config["asr"].get("whisper_pad_seconds", 0.5),
        )
        from livetranslate.ui.hotkeys import DEFAULT_HOTKEYS

        self._current_settings.setdefault("hotkeys", dict(DEFAULT_HOTKEYS))

        # Appearance: dark by default, overridable from the 常规 page.
        mode = str(self._current_settings.get("theme", DEFAULT_THEME))
        self._theme_mode = mode if mode in THEME_MODES else DEFAULT_THEME
        self._current_settings.setdefault("theme", self._theme_mode)

        # Give the store the finalized draft (a copy) so snapshot/export on
        # a first-run panel reports the defaults before the first commit. The
        # draft itself stays a separate panel-local working copy (never the
        # store's internal dict).
        self._store.seed(dict(self._current_settings))

        # Chrome (QSS + safety-net palette) — applied once the settings
        # are final, so a saved theme is honoured from the first frame.
        self._apply_chrome()

        # -- Pages -----------------------------------------------------------
        self._general_tab = GeneralTab(self)
        self._translation_tab = TranslationTab(self)
        self._vad_tab = VadTab(self)
        self._hotkeys_group = HotkeysGroup(self)  # §3.2/§3.7: 设置 → 识别
        self._recognition_page = self._build_recognition_page()
        self._subtitle_page = self._build_subtitle_page()
        self._cache_tab = CacheTab(self)
        self._diagnostics_tab = DiagnosticsTab(self)
        self._about_tab = AboutTab(self)

        self._pages = [
            (t("nav_general"), self._general_tab),
            (t("nav_translation"), self._translation_tab),
            (t("nav_recognition"), self._recognition_page),
            (t("nav_subtitles"), self._subtitle_page),
            (t("nav_data"), self._cache_tab),
            (t("nav_diagnostics"), self._diagnostics_tab),
            (t("nav_about"), self._about_tab),
        ]
        self._cache_index = next(
            i for i, (label, _) in enumerate(self._pages) if label == t("nav_data")
        )
        self._diagnostics_index = next(
            i for i, (label, _) in enumerate(self._pages) if label == t("nav_diagnostics")
        )

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Left column: brand, caption and the 200px page nav.
        nav_column = QVBoxLayout()
        nav_column.setContentsMargins(0, 14, 0, 10)
        nav_column.setSpacing(0)
        brand = QLabel("LiveTranslate")
        brand.setObjectName("navBrand")
        nav_column.addWidget(brand)
        caption = QLabel(t("settings"))
        caption.setObjectName("navCaption")
        nav_column.addWidget(caption)

        self._nav = QListWidget()
        self._nav.setObjectName("panelNav")
        self._nav.setFixedWidth(200)
        self._nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for label, _page in self._pages:
            item = QListWidgetItem(label)
            item.setSizeHint(QSize(0, 38))
            self._nav.addItem(item)
        self._nav.currentRowChanged.connect(self._on_nav_changed)
        nav_column.addWidget(self._nav, 1)

        outer.addLayout(nav_column)

        self._stack = QStackedWidget()
        self._stack.setObjectName("panelPages")
        self._stack.setMaximumWidth(960)  # §3.5.4: content column capped+centered
        for _label, page in self._pages:
            self._stack.addWidget(page)

        content_box = QHBoxLayout()
        content_box.setContentsMargins(16, 16, 16, 16)
        content_box.addStretch(1)
        content_box.addWidget(self._stack)
        content_box.addStretch(1)
        outer.addLayout(content_box, 1)

        # Initial row selection — only after the stack exists, because
        # _on_nav_changed switches the stack page.
        self._nav.setCurrentRow(0)

        self._cache_result.connect(self._cache_tab.on_result)

        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self._do_auto_save)

    def _build_recognition_page(self) -> QWidget:
        """识别 page: header (title + benchmark) over a scrollable column of
        the VAD/ASR tab and the global hotkeys group."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.addWidget(page_header(t("nav_recognition"), t("page_recognition_hint")), 1)
        bench_btn = QPushButton(t("btn_open_benchmark"))
        bench_btn.clicked.connect(self._open_benchmark)
        header_row.addWidget(bench_btn, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header_row)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 4, 0, 0)
        content_layout.setSpacing(12)
        content_layout.addWidget(self._vad_tab)
        content_layout.addWidget(self._hotkeys_group)
        layout.addWidget(make_scroll_page(content), 1)
        return page

    def _build_subtitle_page(self) -> QWidget:
        """字幕 page: style presets + OBS subtitle window settings, scrolling."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(page_header(t("nav_subtitles"), t("page_subtitles_hint")))

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 4, 0, 0)
        content_layout.setSpacing(12)
        self._style_tab = StyleTab(self)
        self._subtitle_tab = SubtitleTab(self)
        content_layout.addWidget(self._style_tab)
        content_layout.addWidget(self._subtitle_tab)
        content_layout.addStretch(1)
        layout.addWidget(make_scroll_page(content), 1)
        return page

    def _open_benchmark(self):
        dlg = BenchmarkDialog(self, parent=self)
        dlg.exec()

    # ── Appearance (light/dark chrome) ──

    def set_theme_mode(self, mode: str) -> None:
        """Switch the panel chrome, persist the choice, repolish live."""
        if mode not in THEME_MODES or mode == self._theme_mode:
            return
        self._theme_mode = mode
        self._current_settings["theme"] = mode
        self._apply_chrome()
        self._auto_save()

    def _apply_chrome(self) -> None:
        """Apply the active theme app-wide (QSS + safety-net palette).

        The theme must live at the application level: popups, menus and
        dialogs resolve their palette from the app palette, so a
        panel-only stylesheet left them on the platform light defaults.
        """
        apply_app_theme(self._theme_mode)

    # ── Shared save / apply machinery ──

    def _auto_save(self):
        self._save_timer.start()

    def _do_auto_save(self):
        self._apply_settings()

    def apply_settings(self):
        """Public alias used by the app shell's deferred init."""
        self._apply_settings()

    def _apply_settings(self):
        self._general_tab.collect()
        self._vad_tab.collect()
        self._translation_tab.collect()
        self._style_tab.collect()
        self._cache_tab.collect()
        safe = redact_settings(self._current_settings, exclude=("models", "system_prompt"))
        log.info(f"Settings applied: {safe}")
        # Commit the draft into the store (replace-merge + atomic save), then
        # hand out a fresh snapshot — never the live internal dict.
        self._store.update(self._current_settings)
        self.settings_changed.emit(self._store.snapshot())

    def get_settings(self):
        return dict(self._current_settings)

    def _on_nav_changed(self, row):
        # Wire the nav to the page stack first: without this the panel
        # stayed frozen on page 0 no matter what was clicked.
        if row >= 0:
            self._stack.setCurrentIndex(row)
        if row == self._cache_index:
            self._cache_tab.refresh()
        elif row == self._diagnostics_index:
            self._diagnostics_tab.ensure_built(self._app)

    def attach_app(self, app_ref):
        """Give the panel a live app reference for the diagnostics page."""
        self._app = app_ref
        self._diagnostics_tab.ensure_built(app_ref)

    # --- Public settings API (replaces cross-module dict writes) ---

    @property
    def store(self) -> SettingsStore:
        return self._store

    def update_settings(self, items: dict):
        """Merge + persist several settings keys through the store, keeping
        the panel draft in sync too (dual merge)."""
        self._store.update(items)
        self._current_settings = {**self._current_settings, **items}

    def commit_now(self):
        """Commit the current draft (panel-local working copy) to the store
        atomically."""
        self._store.update(self._current_settings)

    def refresh_model_list(self):
        """Re-render the translation tab's model list (called from the tray)."""
        self._translation_tab.refresh_model_list()

    def set_hotkey_combo(self, name: str, combo: str):
        """Revert a hotkey binding after an OS-level conflict (shell path)."""
        self._hotkeys_group.set_combo(name, combo)

    def set_asr_language(self, code: str):
        """Update the ASR language setting and sync the panel combo."""
        self._store.update({"asr_language": code})
        self._current_settings["asr_language"] = code
        self._vad_tab.sync_asr_language(code)

    def set_subtitle_click_through(self, checked: bool):
        """Update the subtitle click-through setting and sync the settings
        tab checkbox (the tray action calls this instead of reaching into
        private widgets)."""
        sm = dict(self._current_settings.get("subtitle_mode") or {})
        sm["click_through"] = bool(checked)
        self._store.update({"subtitle_mode": sm})
        self._current_settings["subtitle_mode"] = sm
        self._subtitle_tab.sync_click_through(checked)

    def update_subtitle_settings(self, s):
        self._current_settings["subtitle_mode"] = s
        self._subtitle_tab.update_settings(s)

    def get_active_model(self) -> dict | None:
        models = self._current_settings.get("models", [])
        idx = self._current_settings.get("active_model", 0)
        if 0 <= idx < len(models):
            return models[idx]
        return None

    def has_saved_settings(self) -> bool:
        return SETTINGS_FILE.exists()
