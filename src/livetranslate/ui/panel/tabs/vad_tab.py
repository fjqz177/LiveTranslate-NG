"""VAD / ASR tab: engine selection, devices, VAD thresholds and timing.

M-SPLIT (2026-08-31): the engine-orchestration methods now live in
``vad_engine._EngineRuntimeMixin`` and the whisper-model-download methods in
``vad_whisper._WhisperDownloadMixin``; this module composes them into
``VadTab``.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from livetranslate.asr.registry import (
    ENGINE_REGISTRY,
    GUI_ENGINE_ORDER,
    engine_display_key,
    engine_id_for_type,
)
from livetranslate.core.i18n import LANGUAGES, t
from livetranslate.modeling.manager import (
    DEFAULT_FUNASR_MODEL,
    funasr_model_options,
    funasr_supports_padding,
    normalize_asr_engine_selection,
    normalize_funasr_model_key,
)
from livetranslate.ui.panel._tab_base import TabBase
from livetranslate.ui.panel.tabs.vad_engine import _EngineRuntimeMixin
from livetranslate.ui.panel.tabs.vad_whisper import _WhisperDownloadMixin


class VadTab(TabBase, _EngineRuntimeMixin, _WhisperDownloadMixin):
    def __init__(self, panel):
        super().__init__(panel)
        layout = QVBoxLayout(self)
        s = self.settings

        asr_group = QGroupBox(t("group_asr_engine"))
        asr_layout = QGridLayout(asr_group)
        asr_layout.setColumnStretch(0, 1)
        asr_layout.setColumnMinimumWidth(1, 180)

        self._asr_engine = QComboBox()
        self._asr_engine.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        # M-MATRIX: build the dropdown from the single-source registry
        # (GUI_ENGINE_ORDER / ENGINE_REGISTRY) — userData is the registry id,
        # label is the localized engine_display_* key (t("engine_display_" + id),
        # with a tier badge on the recommended entry). No index<->engine map
        # lives here (selection, persist and restore all key on the id, not the
        # index; spec.display_name stays for diagnostics display).
        for engine_id in GUI_ENGINE_ORDER:
            spec = ENGINE_REGISTRY[engine_id]
            label = t(engine_display_key(spec.id))
            if spec.tier == "recommended":
                label += f"  ({t('engine_tier_recommended')})"
            self._asr_engine.addItem(label, spec.id)
        # Restore the persisted worker-frontier engine_type back to its registry
        # id. engine_id_for_type returns None for unknown/legacy/mismatched
        # values; degrade to the first item instead of crashing (Task 6 gate).
        engine_type, _ = normalize_asr_engine_selection(s.get("asr_engine"))
        engine_id = engine_id_for_type(engine_type)
        engine_idx = self._asr_engine.findData(engine_id) if engine_id is not None else -1
        self._asr_engine.setCurrentIndex(engine_idx if engine_idx >= 0 else 0)
        asr_layout.addWidget(QLabel(t("label_engine")), 0, 0)
        asr_layout.addWidget(self._asr_engine, 0, 1)
        self._asr_engine.currentIndexChanged.connect(self.auto_save)

        # Engine availability + one-click dependency install (§3.2/ADR-006).
        # The status row is placed inside the engine group, right under the
        # engine picker it describes (not floating above the group).
        # Engine runtime card (SelfServe P1-B5): hardware/variant state,
        # install/switch/remove actions and the dependency mirror picker.
        # Frozen builds install variants via the embedded uv; dev builds keep
        # the uv sync flow. The old disabled-button placeholder is gone.
        self._engine_status_label = QLabel()
        self._engine_status_label.setObjectName("hintLabel")
        self._engine_status_label.setWordWrap(True)
        self._engine_status_row = QWidget()
        status_row = QHBoxLayout(self._engine_status_row)
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.addWidget(self._engine_status_label, 1)
        self._refresh_engine_status()
        self._asr_engine.currentIndexChanged.connect(lambda _i: self._refresh_engine_status())

        self._asr_lang = QComboBox()
        for code, native in LANGUAGES:
            label = t("asr_lang_auto") if code == "auto" else native
            self._asr_lang.addItem(f"{code} - {label}", code)
        lang = s.get("asr_language", self.config["asr"].get("language", "auto"))
        idx = self._asr_lang.findData(lang)
        if idx >= 0:
            self._asr_lang.setCurrentIndex(idx)
        asr_layout.addWidget(QLabel(t("label_language_hint")), 1, 0)
        asr_layout.addWidget(self._asr_lang, 1, 1)
        self._asr_lang.currentIndexChanged.connect(self.auto_save)
        self._asr_lang.currentIndexChanged.connect(self._on_asr_lang_combo_changed)

        self._asr_device = QComboBox()
        devices = ["cuda", "cpu"]
        try:
            import torch

            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                devices.insert(i, f"cuda:{i} ({name})")
            if torch.cuda.device_count() > 0:
                devices = [d for d in devices if d != "cuda"]
        except Exception:
            pass
        self._asr_device.addItems(devices)
        saved_dev = s.get("asr_device", self.config["asr"].get("device", "cuda"))
        for i in range(self._asr_device.count()):
            if self._asr_device.itemText(i).startswith(saved_dev):
                self._asr_device.setCurrentIndex(i)
                break
        asr_layout.addWidget(QLabel(t("label_device")), 2, 0)
        asr_layout.addWidget(self._asr_device, 2, 1)
        self._asr_device.currentIndexChanged.connect(self.auto_save)

        self._funasr_model_label = QLabel(t("label_funasr_model"))
        self._funasr_model_combo = QComboBox()
        for key, display_name in funasr_model_options():
            self._funasr_model_combo.addItem(display_name, key)
        saved_funasr_model = normalize_funasr_model_key(s.get("funasr_model", DEFAULT_FUNASR_MODEL))
        funasr_idx = self._funasr_model_combo.findData(saved_funasr_model)
        if funasr_idx >= 0:
            self._funasr_model_combo.setCurrentIndex(funasr_idx)
        self._funasr_model_combo.currentIndexChanged.connect(self._on_funasr_model_changed)
        asr_layout.addWidget(self._funasr_model_label, 3, 0)
        asr_layout.addWidget(self._funasr_model_combo, 3, 1)

        self._whisper_pad_label = QLabel(t("label_whisper_padding"))
        self._whisper_pad_seconds = QDoubleSpinBox()
        self._whisper_pad_seconds.setRange(0.0, 5.0)
        self._whisper_pad_seconds.setDecimals(2)
        self._whisper_pad_seconds.setSingleStep(0.1)
        try:
            whisper_pad_seconds = float(s.get("whisper_pad_seconds", 0.5))
        except (TypeError, ValueError):
            whisper_pad_seconds = 0.5
        self._whisper_pad_seconds.setValue(whisper_pad_seconds)
        self._whisper_pad_seconds.setSuffix(" s")
        self._whisper_pad_seconds.setSpecialValueText(t("whisper_padding_off"))
        self._whisper_pad_seconds.setToolTip(t("whisper_padding_tooltip"))
        asr_layout.addWidget(self._whisper_pad_label, 4, 0)
        asr_layout.addWidget(self._whisper_pad_seconds, 4, 1)
        self._whisper_pad_seconds.valueChanged.connect(self.auto_save)

        self._sensevoice_pad_label = QLabel(t("label_sensevoice_padding"))
        self._sensevoice_pad_seconds = QDoubleSpinBox()
        self._sensevoice_pad_seconds.setRange(0.0, 5.0)
        self._sensevoice_pad_seconds.setDecimals(2)
        self._sensevoice_pad_seconds.setSingleStep(0.1)
        try:
            sensevoice_pad_seconds = float(s.get("sensevoice_pad_seconds", 0.5))
        except (TypeError, ValueError):
            sensevoice_pad_seconds = 0.5
        self._sensevoice_pad_seconds.setValue(sensevoice_pad_seconds)
        self._sensevoice_pad_seconds.setSuffix(" s")
        self._sensevoice_pad_seconds.setSpecialValueText(t("sensevoice_padding_off"))
        self._sensevoice_pad_seconds.setToolTip(t("sensevoice_padding_tooltip"))
        asr_layout.addWidget(self._sensevoice_pad_label, 5, 0)
        asr_layout.addWidget(self._sensevoice_pad_seconds, 5, 1)
        self._sensevoice_pad_seconds.valueChanged.connect(self.auto_save)

        self._audio_device = QComboBox()
        self._audio_device.addItem(t("audio_disabled"))
        self._audio_device.addItem(t("system_default"))
        try:
            from livetranslate.audio.registry import list_outputs

            for dev in list_outputs():
                self._audio_device.addItem(dev.name)
        except Exception:
            pass
        saved_audio = s.get("audio_device")
        if saved_audio == "__disabled__":
            self._audio_device.setCurrentIndex(0)
        elif saved_audio:
            idx = self._audio_device.findText(saved_audio)
            if idx >= 0:
                self._audio_device.setCurrentIndex(idx)
        else:
            self._audio_device.setCurrentIndex(1)  # system default
        asr_layout.addWidget(QLabel(t("label_audio")), 6, 0)
        asr_layout.addWidget(self._audio_device, 6, 1)
        self._audio_device.currentIndexChanged.connect(self.auto_save)

        self._mic_device = QComboBox()
        self._mic_device.addItem(t("mic_disabled"))
        self._mic_device.addItem(t("system_default"))
        try:
            from livetranslate.audio.registry import list_inputs

            for dev in list_inputs():
                self._mic_device.addItem(dev.name)
        except Exception:
            pass
        saved_mic = s.get("mic_device")
        if saved_mic:
            if saved_mic in ("__default__", "default"):
                self._mic_device.setCurrentIndex(1)
            else:
                idx = self._mic_device.findText(saved_mic)
                if idx >= 0:
                    self._mic_device.setCurrentIndex(idx)
        asr_layout.addWidget(QLabel(t("label_mic")), 7, 0)
        asr_layout.addWidget(self._mic_device, 7, 1)
        self._mic_device.currentIndexChanged.connect(self.auto_save)

        # Engine availability status lives inside the engine group, right
        # under the engine picker it describes.
        asr_layout.addWidget(self._engine_status_row, 8, 0, 1, 2)

        layout.addWidget(asr_group)

        # Download source (model hub + pypi/torch mirror + proxy) moved from
        # the old first-run wizard into the recognition page; collapsed by
        # default to keep the page clean.
        layout.addWidget(self._build_download_source_group())

        # Whisper model download — only visible when engine is Whisper
        self._whisper_group = QGroupBox(t("group_download_whisper"))
        whisper_layout = QHBoxLayout(self._whisper_group)
        self._whisper_size_combo = QComboBox()
        saved_size = s.get("whisper_model_size", self.config["asr"].get("model_size", "medium"))
        self._populate_whisper_models(saved_size)
        self._whisper_size_combo.currentIndexChanged.connect(self._on_whisper_size_changed)
        whisper_layout.addWidget(self._whisper_size_combo)
        self._whisper_status = QLabel("")
        self._whisper_status.setProperty("status", "none")
        whisper_layout.addWidget(self._whisper_status, 1)
        self._whisper_dl_btn = QPushButton(t("btn_download_whisper"))
        self._whisper_dl_btn.clicked.connect(self._download_whisper)
        whisper_layout.addWidget(self._whisper_dl_btn)
        layout.addWidget(self._whisper_group)
        self._asr_engine.currentIndexChanged.connect(self._on_engine_changed_whisper_vis)
        self._update_whisper_size_label()

        # Remote ASR server URL + optional token — only visible when the
        # engine is Remote Whisper. SEC-5: the token rides X-ASR-Token.
        self._remote_group = QGroupBox(t("group_remote_asr"))
        remote_layout = QVBoxLayout(self._remote_group)
        remote_row = QHBoxLayout()
        remote_row.addWidget(QLabel(t("label_remote_url")))
        self._remote_url_edit = QLineEdit(s.get("remote_asr_url", "http://127.0.0.1:8765"))
        self._remote_url_edit.setPlaceholderText("http://127.0.0.1:8765")
        self._remote_url_edit.editingFinished.connect(self.auto_save)
        remote_row.addWidget(self._remote_url_edit, 1)
        remote_layout.addLayout(remote_row)
        token_row = QHBoxLayout()
        token_row.addWidget(QLabel(t("label_remote_token")))
        self._remote_token_edit = QLineEdit(s.get("remote_asr_token", ""))
        self._remote_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._remote_token_edit.setToolTip(t("remote_token_hint"))
        self._remote_token_edit.editingFinished.connect(self.auto_save)
        token_row.addWidget(self._remote_token_edit, 1)
        remote_layout.addLayout(token_row)
        layout.addWidget(self._remote_group)
        # First visibility pass runs AFTER every group is built (whisper_group,
        # funasr widgets, pads, remote_group) so the method sets all of them by
        # engine_type in one place. A QGroupBox defaults to visible, so this
        # call is what hides the remote group for a non-remote engine — the
        # earlier placement (before _remote_group existed) silently skipped it
        # via the hasattr guard and left Remote visible for every engine.
        self._on_engine_changed_whisper_vis(self._asr_engine.currentIndex())

        mode_group = QGroupBox(t("group_vad_mode"))
        mode_layout = QVBoxLayout(mode_group)
        self._vad_mode = QComboBox()
        self._vad_mode.addItems([t("vad_silero"), t("vad_energy"), t("vad_disabled")])
        mode_map = {"silero": 0, "energy": 1, "disabled": 2}
        self._vad_mode.setCurrentIndex(mode_map.get(s.get("vad_mode", "energy"), 1))
        self._vad_mode.currentIndexChanged.connect(self._on_vad_mode_changed)
        self._vad_mode.currentIndexChanged.connect(self.auto_save)
        mode_layout.addWidget(self._vad_mode)
        layout.addWidget(mode_group)

        silero_group = QGroupBox(t("group_silero_threshold"))
        silero_layout = QGridLayout(silero_group)
        self._vad_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._vad_threshold_slider.setRange(0, 100)
        vad_pct = int(s.get("vad_threshold", 0.5) * 100)
        self._vad_threshold_slider.setValue(vad_pct)
        self._vad_threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self._vad_threshold_slider.sliderReleased.connect(self.auto_save)
        self._vad_threshold_label = QLabel(f"{vad_pct}%")
        self._vad_threshold_label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        silero_layout.addWidget(QLabel(t("label_threshold")), 0, 0)
        silero_layout.addWidget(self._vad_threshold_slider, 0, 1)
        silero_layout.addWidget(self._vad_threshold_label, 0, 2)
        layout.addWidget(silero_group)

        energy_group = QGroupBox(t("group_energy_threshold"))
        energy_layout = QGridLayout(energy_group)
        self._energy_slider = QSlider(Qt.Orientation.Horizontal)
        self._energy_slider.setRange(1, 100)
        energy_pm = int(s.get("energy_threshold", 0.03) * 1000)
        self._energy_slider.setValue(energy_pm)
        self._energy_slider.valueChanged.connect(self._on_energy_changed)
        self._energy_slider.sliderReleased.connect(self.auto_save)
        self._energy_label = QLabel(f"{energy_pm}‰")
        self._energy_label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        energy_layout.addWidget(QLabel(t("label_threshold")), 0, 0)
        energy_layout.addWidget(self._energy_slider, 0, 1)
        energy_layout.addWidget(self._energy_label, 0, 2)
        layout.addWidget(energy_group)

        timing_group = QGroupBox(t("group_timing"))
        timing_layout = QGridLayout(timing_group)
        timing_layout.setColumnStretch(0, 1)
        timing_layout.setColumnMinimumWidth(1, 180)
        self._min_speech = QDoubleSpinBox()
        self._min_speech.setRange(0.1, 5.0)
        self._min_speech.setSingleStep(0.1)
        self._min_speech.setValue(s.get("min_speech_duration", 2.0))
        self._min_speech.setSuffix(" s")
        self._min_speech.valueChanged.connect(self._on_timing_changed)
        self._min_speech.valueChanged.connect(self.auto_save)
        self._max_speech = QDoubleSpinBox()
        self._max_speech.setRange(2.0, 30.0)
        self._max_speech.setSingleStep(1.0)
        self._max_speech.setValue(s.get("max_speech_duration", 6.0))
        self._max_speech.setSuffix(" s")
        self._max_speech.valueChanged.connect(self._on_timing_changed)
        self._max_speech.valueChanged.connect(self.auto_save)
        self._silence_mode = QComboBox()
        self._silence_mode.addItems([t("silence_auto"), t("silence_fixed")])
        saved_smode = s.get("silence_mode", "auto")
        self._silence_mode.setCurrentIndex(0 if saved_smode == "auto" else 1)
        self._silence_mode.currentIndexChanged.connect(self._on_silence_mode_changed)
        self._silence_mode.currentIndexChanged.connect(self._on_timing_changed)
        self._silence_mode.currentIndexChanged.connect(self.auto_save)

        self._silence_duration = QDoubleSpinBox()
        self._silence_duration.setRange(0.1, 3.0)
        self._silence_duration.setSingleStep(0.1)
        self._silence_duration.setValue(s.get("silence_duration", 0.8))
        self._silence_duration.setSuffix(" s")
        self._silence_duration.setEnabled(saved_smode != "auto")
        # Disabled in Auto mode: explain why on hover (Qt hides tooltips on
        # disabled widgets unless WA_AlwaysShowToolTips is set).
        self._silence_duration.setToolTip(t("silence_dur_disabled_tooltip"))
        self._silence_duration.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips, True)
        self._silence_duration.valueChanged.connect(self._on_timing_changed)
        self._silence_duration.valueChanged.connect(self.auto_save)

        timing_layout.addWidget(QLabel(t("label_min_speech")), 0, 0)
        timing_layout.addWidget(self._min_speech, 0, 1)
        timing_layout.addWidget(QLabel(t("label_max_speech")), 1, 0)
        timing_layout.addWidget(self._max_speech, 1, 1)
        timing_layout.addWidget(QLabel(t("label_silence")), 2, 0)
        timing_layout.addWidget(self._silence_mode, 2, 1)
        timing_layout.addWidget(QLabel(t("label_silence_dur")), 3, 0)
        timing_layout.addWidget(self._silence_duration, 3, 1)

        self._incremental_asr_cb = QCheckBox(t("label_incremental_asr"))
        self._incremental_asr_cb.setToolTip(t("incremental_asr_tooltip"))
        self._incremental_asr_cb.setChecked(s.get("incremental_asr", False))
        self._incremental_asr_cb.toggled.connect(self._on_timing_changed)
        self._incremental_asr_cb.toggled.connect(self.auto_save)
        timing_layout.addWidget(self._incremental_asr_cb, 4, 0)

        self._interim_interval_spin = QDoubleSpinBox()
        self._interim_interval_spin.setRange(1.0, 10.0)
        self._interim_interval_spin.setSingleStep(0.5)
        self._interim_interval_spin.setValue(s.get("interim_interval", 2.0))
        self._interim_interval_spin.setSuffix(" s")
        self._interim_interval_spin.setEnabled(s.get("incremental_asr", False))
        # Disabled until Incremental ASR is checked: explain why on hover.
        self._interim_interval_spin.setToolTip(t("interim_interval_disabled_tooltip"))
        self._interim_interval_spin.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips, True)
        self._interim_interval_spin.valueChanged.connect(self._on_timing_changed)
        self._interim_interval_spin.valueChanged.connect(self.auto_save)
        self._incremental_asr_cb.toggled.connect(self._on_incremental_toggled)
        timing_layout.addWidget(QLabel(t("label_interim_interval")), 5, 0)
        timing_layout.addWidget(self._interim_interval_spin, 5, 1)

        layout.addWidget(timing_group)

        layout.addStretch()

    def _selected_engine_id(self) -> str:
        """The registry id of the currently selected engine (combo userData).

        Guards a None userData (shouldn't happen — the combo always has an
        item) by degrading to the first GUI_ENGINE_ORDER entry.
        """
        value = self._asr_engine.currentData()
        return str(value) if value is not None else GUI_ENGINE_ORDER[0]

    def _build_download_source_group(self) -> QWidget:
        """Collapsed "download source" group: model hub + pypi/torch mirror +
        the proxy used for model & engine downloads.

        Relocated from the old first-run wizard's advanced group into the
        recognition page — env-fill now lives in Settings. A checkable group
        box with a hidden body keeps the default page clean: the options only
        appear once the user ticks the group header.
        """
        group = QGroupBox(t("group_download_source"))
        group.setCheckable(True)
        group.setChecked(False)  # collapsed: ordinary users never need these
        body = QWidget()
        form = QFormLayout(body)
        form.setContentsMargins(0, 0, 0, 0)
        outer = QVBoxLayout(group)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(body)
        group.toggled.connect(lambda checked: body.setVisible(checked))
        body.setVisible(False)

        s = self.settings
        # Widgets set their initial index BEFORE the change-signal connect
        # (the existing vad_tab convention): setCurrentIndex fires
        # currentIndexChanged, and calling auto_save() during __init__ would
        # run before ControlPanel._save_timer exists.
        self._hub_combo = QComboBox()
        self._hub_combo.addItems([t("hub_modelscope"), t("hub_huggingface")])
        saved_hub = s.get("hub", "ms")
        self._hub_combo.setCurrentIndex(0 if saved_hub == "ms" else 1)
        self._hub_combo.currentIndexChanged.connect(self.auto_save)
        form.addRow(t("label_hub"), self._hub_combo)

        self._proxy_mode = QComboBox()
        self._proxy_mode.addItems([t("proxy_none"), t("proxy_system"), t("proxy_custom")])
        saved_proxy = s.get("download_proxy", "system")
        self._proxy_mode.setCurrentIndex({"none": 0, "system": 1, "custom": 2}.get(saved_proxy, 1))
        self._proxy_mode.currentIndexChanged.connect(self._on_proxy_changed)
        self._proxy_url = QLineEdit()
        self._proxy_url.setPlaceholderText("http://127.0.0.1:7890")
        self._proxy_url.editingFinished.connect(self._on_proxy_changed)
        self._proxy_url.setEnabled(self._proxy_mode.currentIndex() == 2)
        form.addRow(t("label_proxy"), self._proxy_mode)
        form.addRow(t("label_proxy_url"), self._proxy_url)
        self.settings["download_proxy"] = self._download_proxy_value()

        return group

    def _on_proxy_changed(self) -> None:
        """Enable the custom URL only in custom mode; persist downloads proxy."""
        self._proxy_url.setEnabled(self._proxy_mode.currentIndex() == 2)
        self.settings["download_proxy"] = self._download_proxy_value()
        self.auto_save()

    def _download_proxy_value(self) -> str:
        index = self._proxy_mode.currentIndex()
        if index == 1:
            return "system"
        if index == 2:
            return self._proxy_url.text().strip() or "system"
        return "none"

    def collect(self):
        """Write this tab's widget state into the shared settings dict."""
        self.settings["asr_language"] = self.get_asr_lang_code()
        # M-MATRIX: persist the worker-frontier engine_type for the selected
        # registry id (ENGINE_REGISTRY[id].engine_type) — the same vocabulary
        # the app layer stores and the registry reverse-looks-up on restore.
        self.settings["asr_engine"] = ENGINE_REGISTRY[self._selected_engine_id()].engine_type
        self.settings["funasr_model"] = self._selected_funasr_model()
        url = self._remote_url_edit.text().strip()
        if url:
            self.settings["remote_asr_url"] = url
        self.settings["remote_asr_token"] = self._remote_token_edit.text().strip()
        self.settings["whisper_model_size"] = self._selected_whisper_model()
        dev_text = self._asr_device.currentText()
        self.settings["asr_device"] = dev_text.split(" (")[0]
        audio_idx = self._audio_device.currentIndex()
        if audio_idx == 0:
            self.settings["audio_device"] = "__disabled__"
        elif audio_idx == 1:
            self.settings["audio_device"] = None
        else:
            self.settings["audio_device"] = self._audio_device.currentText()
        mic_idx = self._mic_device.currentIndex()
        if mic_idx == 0:
            self.settings["mic_device"] = None
        elif mic_idx == 1:
            self.settings["mic_device"] = "__default__"
        else:
            self.settings["mic_device"] = self._mic_device.currentText()
        self.settings["hub"] = "ms" if self._hub_combo.currentIndex() == 0 else "hf"
        self.settings["download_proxy"] = self._download_proxy_value()
        self.settings["sensevoice_pad_seconds"] = round(self._sensevoice_pad_seconds.value(), 2)
        self.settings["whisper_pad_seconds"] = round(self._whisper_pad_seconds.value(), 2)
        self._on_timing_changed()

    def get_asr_lang_code(self) -> str:
        """Get the language code from the ASR language combo (stored as userData)."""
        return self._asr_lang.currentData() or "auto"

    def sync_asr_language(self, code: str):
        """Sync the combo with a language code coming from outside the panel."""
        idx = self._asr_lang.findData(code)
        if idx >= 0:
            self._asr_lang.blockSignals(True)
            self._asr_lang.setCurrentIndex(idx)
            self._asr_lang.blockSignals(False)

    def _on_asr_lang_combo_changed(self, _index):
        self.panel.asr_language_changed.emit(self._asr_lang.currentData() or "auto")

    def _on_engine_changed_whisper_vis(self, _index=0):
        # M-MATRIX: keyed on the worker-frontier engine_type (from the selected
        # registry id), never on the combo index — the dropdown now derives from
        # GUI_ENGINE_ORDER and indices are no longer an engine vocabulary.
        engine_type = ENGINE_REGISTRY[self._selected_engine_id()].engine_type
        self._whisper_group.setVisible(engine_type == "whisper")
        is_funasr = engine_type == "funasr"
        if hasattr(self, "_funasr_model_combo"):
            self._funasr_model_label.setVisible(is_funasr)
            self._funasr_model_combo.setVisible(is_funasr)
        if hasattr(self, "_whisper_pad_seconds"):
            is_whisper = engine_type == "whisper"
            self._whisper_pad_label.setVisible(is_whisper)
            self._whisper_pad_seconds.setVisible(is_whisper)
        if hasattr(self, "_sensevoice_pad_seconds"):
            if engine_type == "funasr":
                show_sensevoice_pad = funasr_supports_padding(self._selected_funasr_model())
            elif engine_type == "sensevoice-onnx":
                show_sensevoice_pad = True
            else:
                show_sensevoice_pad = False
            self._sensevoice_pad_label.setVisible(show_sensevoice_pad)
            self._sensevoice_pad_seconds.setVisible(show_sensevoice_pad)
        if hasattr(self, "_remote_group"):
            self._remote_group.setVisible(engine_type == "remote-whisper")

    def _selected_funasr_model(self) -> str:
        value = self._funasr_model_combo.currentData()
        return normalize_funasr_model_key(str(value) if value else None)

    def _on_funasr_model_changed(self):
        self.settings["funasr_model"] = self._selected_funasr_model()
        self._on_engine_changed_whisper_vis(self._asr_engine.currentIndex())
        self.auto_save()

    def _selected_whisper_model(self) -> str:
        value = self._whisper_size_combo.currentData()
        return str(value) if value else self._whisper_size_combo.currentText()

    # ── VAD / timing slots ──

    def _on_silence_mode_changed(self, index):
        self._silence_duration.setEnabled(index == 1)
        # Auto mode: keep the "why disabled" tooltip; Fixed mode: clear it.
        self._silence_duration.setToolTip("" if index == 1 else t("silence_dur_disabled_tooltip"))

    def _on_incremental_toggled(self, checked):
        self._interim_interval_spin.setEnabled(checked)
        self._interim_interval_spin.setToolTip(
            "" if checked else t("interim_interval_disabled_tooltip")
        )

    def _on_vad_mode_changed(self, index):
        modes = ["silero", "energy", "disabled"]
        self.settings["vad_mode"] = modes[index]

    def _on_threshold_changed(self, value):
        val = value / 100.0
        self.settings["vad_threshold"] = val
        self._vad_threshold_label.setText(f"{value}%")
        if not self._vad_threshold_slider.isSliderDown():
            self.auto_save()

    def _on_energy_changed(self, value):
        val = value / 1000.0
        self.settings["energy_threshold"] = val
        self._energy_label.setText(f"{value}‰")
        if not self._energy_slider.isSliderDown():
            self.auto_save()

    def _on_timing_changed(self):
        self.settings["min_speech_duration"] = round(self._min_speech.value(), 2)
        self.settings["max_speech_duration"] = round(self._max_speech.value(), 2)
        self.settings["silence_mode"] = (
            "auto" if self._silence_mode.currentIndex() == 0 else "fixed"
        )
        self.settings["silence_duration"] = round(self._silence_duration.value(), 2)
        self.settings["incremental_asr"] = self._incremental_asr_cb.isChecked()
        self.settings["interim_interval"] = round(self._interim_interval_spin.value(), 2)
