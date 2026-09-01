"""
LiveTranslate application composition root.

Startup sequencing, MainApp wiring and main() live here. The entry shim
(main.py at the repo root) owns the import-order constraints —
apply_cache_env() before torch, torch before PyQt6 — and then calls
main() from this module.
"""

import logging
import os
import signal
import sys
import threading
import types
from datetime import datetime
from typing import Any

import yaml

# torch is present in the base install (engine deps ship with the package).
# The entry shim imports torch before PyQt6 on Windows (DLL order); the
# memory diagnostics below degrade gracefully in split installs.
try:
    import torch
except ImportError:  # unusual split install; keep diagnostics graceful
    torch = None

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox, QSystemTrayIcon

from livetranslate.asr.controller import AsrController
from livetranslate.audio.registry import create_audio_backend
from livetranslate.audio.vad.processor import VADProcessor
from livetranslate.core.i18n import resolve_ui_lang, set_lang, t
from livetranslate.core.paths import LOG_DIR, PROJECT_ROOT, transcripts_dir
from livetranslate.core.pipeline import Pipeline
from livetranslate.core.settings import load_user_settings
from livetranslate.core.transcript_writer import TranscriptWriter
from livetranslate.core.translator import Translator
from livetranslate.modeling.manager import (
    DEFAULT_FUNASR_MODEL,
    get_missing_models,
    migrate_funasr_settings,
    normalize_funasr_model_key,
)
from livetranslate.platform.registry import create_system_integration
from livetranslate.ui.app_services.settings_applier import apply_settings
from livetranslate.ui.app_services.worker_config import build_worker_config
from livetranslate.ui.app_shell import build_tray_shell
from livetranslate.ui.dialogs import ModelDownloadDialog, _ModelLoadDialog
from livetranslate.ui.icons import create_app_icon
from livetranslate.ui.log_window import LogWindow
from livetranslate.ui.memory_monitor import MemoryMonitor
from livetranslate.ui.overlay import SubtitleOverlay
from livetranslate.ui.panel.panel import ControlPanel
from livetranslate.ui.single_instance import WAKE_FAILED, acquire_single_instance
from livetranslate.ui.subtitle_window import SubtitleWindow


def setup_logging():
    log_dir = LOG_DIR
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f"livetrans_{datetime.now():%Y%m%d_%H%M%S}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(fmt)

    handlers: list[logging.Handler] = [file_handler]
    # Windowed frozen builds have sys.stdout = None; a StreamHandler bound
    # to None crashes on every emit ("'NoneType' object has no attribute
    # 'write'" spam). The file handler (and the Qt log window) still get
    # every record.
    if sys.stdout is not None:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(fmt)
        handlers.append(console_handler)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    for noisy in (
        "httpcore",
        "httpx",
        "openai",
        "filelock",
        "huggingface_hub",
        "funasr",
        "modelscope",
        "onnxruntime",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info(f"Log file: {log_file}")

    # FunASR/ModelScope spam the root logger; suppress after our own init log
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("LiveTranslate").setLevel(logging.DEBUG)

    _logger = logging.getLogger("LiveTranslate")

    def _excepthook(exc_type, exc_value, exc_tb):
        UNCAUGHT_ERRORS.append(f"{exc_type.__name__}: {exc_value}")
        _logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    def _thread_excepthook(args):
        UNCAUGHT_ERRORS.append(f"{args.exc_type.__name__}: {args.exc_value}")
        _logger.critical(
            f"Uncaught exception in thread {args.thread}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_excepthook

    return _logger


log = logging.getLogger("LiveTranslate")

# Uncaught thread exceptions land here too (smoke mode fails on them).
UNCAUGHT_ERRORS: list[str] = []


def load_config():
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class LiveTranslateApp:
    def __init__(self, config):
        self._config = config
        # Created after AsrController (see below) to avoid a construction-order
        # cycle; the type is declared up front so readers never hit None.
        self._pipeline: Pipeline | None = None

        self._audio = create_audio_backend()
        # AUD-3: capture death after the restart budget (PipeWire) escalates
        # to a user-visible banner — detect → degrade → guide, never silent.
        if hasattr(self._audio, "set_fatal_error_cb"):
            self._audio.set_fatal_error_cb(self._on_capture_fatal_error)
        self._audio_cfg = {
            "device": config["audio"].get("device"),
            "sample_rate": config["audio"]["sample_rate"],
            "chunk_ms": int(config["audio"]["chunk_duration"] * 1000),
        }
        self._vad = VADProcessor(
            sample_rate=config["audio"]["sample_rate"],
            threshold=config["asr"]["vad_threshold"],
            min_speech_duration=config["asr"]["min_speech_duration"],
            max_speech_duration=config["asr"]["max_speech_duration"],
            chunk_duration=config["audio"]["chunk_duration"],
        )
        # The initial translator comes from config.yaml whose api_key may be
        # empty (the user's real model config arrives via the deferred init,
        # which replaces this translator before the pipeline starts).
        translator = Translator(
            api_base=config["translation"]["api_base"],
            api_key=config["translation"].get("api_key") or "placeholder",
            model=config["translation"]["model"],
            target_language=config["translation"]["target_language"],
            max_tokens=config["translation"]["max_tokens"],
            temperature=config["translation"]["temperature"],
            streaming=config["translation"]["streaming"],
            system_prompt=config["translation"].get("system_prompt"),
        )
        translator.set_context_turns(config["translation"].get("context_window", 0))
        self._transcript = TranscriptWriter(transcripts_dir())

        # All ASR worker lifecycle state lives in the controller now. The
        # running flag is resolved lazily through the pipeline (which is
        # created below) to avoid a construction-order cycle.
        self._asr_ctl = AsrController(
            initial_device=config["asr"]["device"],
            initial_whisper_model_size=config["asr"]["model_size"],
            initial_funasr_model_key=normalize_funasr_model_key(
                config["asr"].get("funasr_model", DEFAULT_FUNASR_MODEL)
            ),
            status_cb=self._update_asr_status_label,
            is_running_cb=lambda: self._pipeline is not None and self._pipeline.running,
            release_memory_cb=self._release_memory_caches,
            mem_cb=self._log_mem_after_asr,
        )

        # Capture/VAD/ASR/translation orchestration lives in Pipeline.
        self._pipeline = Pipeline(
            config,
            self._vad,
            self._audio,
            self._asr_ctl,
            translator,
            self._transcript,
            start_hook=self._on_pipeline_started,
            stop_hook=self._on_pipeline_stopping,
        )

        self._overlay = None
        self._subwin = None
        self._panel = None

        # Recent classified translate errors (diagnostics network card, §3.4)
        from collections import deque

        self._recent_errors: deque[str] = deque(maxlen=3)
        self._pipeline.set_error_reporter(self._recent_errors.append)
        self._smoke = "--smoke" in sys.argv

        # Memory diagnostic state: the monitor owns process-RSS baseline,
        # warn-once threshold and the periodic timer; the collaborator reads
        # (ASR worker pid, VAD buffer, overlay msgs, counters, GPU) stay in the
        # composition root via the lazy _memory_collab_snapshot closure. The
        # AsrController is built before the Pipeline, so those reads must be
        # deferred to snapshot time, never at construction.
        self._mem_monitor = MemoryMonitor(self._memory_collab_snapshot)

    def set_overlay(self, overlay: SubtitleOverlay):
        self._overlay = overlay
        self._pipeline.set_overlay(overlay)

    def set_subtitle_window(self, subwin: SubtitleWindow):
        self._subwin = subwin
        self._pipeline.set_subtitle_window(subwin)

    def set_panel(self, panel: ControlPanel):
        self._panel = panel
        self._pipeline.set_panel(panel)
        panel.settings_changed.connect(self._on_settings_changed)
        panel.model_changed.connect(self._on_model_changed)
        panel.models_list_changed.connect(self._on_models_list_changed)

    # --- Public surface used by the app shell ---

    @property
    def asr_controller(self):
        return self._asr_ctl

    def get_settings(self) -> dict:
        """Current settings snapshot (panel store). The app shell's
        empty-state guide reads this; without it the guide always
        reported "not configured" even with models set up."""
        return self._panel.get_settings() if self._panel else {}

    def on_model_changed(self, model_config: dict):
        self._on_model_changed(model_config)

    def on_target_language_changed(self, lang: str):
        self._on_target_language_changed(lang)

    def _update_asr_status_label(self, text: str):
        """Callback used by AsrController to push status into the overlay."""
        if self._overlay:
            self._overlay.update_asr_device(text)

    def _on_capture_fatal_error(self, message: str) -> None:
        """Audio capture gave up after its restart budget (e.g. parec keeps
        dying on Linux). Surface it in the overlay banner (thread-safe via
        the overlay's signal bridge) — never a silent stall."""
        log.error(f"Capture fatal: {message}")
        if self._overlay:
            self._overlay.show_error(message)

    def _on_models_list_changed(self, models: list, active_idx: int):
        if self._overlay:
            self._overlay.set_models(models, active_idx)

    def _on_settings_changed(self, settings):
        # M-COMPOSE: the routing decision lives in ui/app_services; the
        # composition root just hands over its live collaborators. The ASR
        # switcher is bridged to the `switcher.switch(...)` callable contract
        # via a tiny adapter (SimpleNamespace).
        apply_settings(
            settings,
            vad=self._vad,
            asr_ctl=self._asr_ctl,
            pipeline=self._pipeline,
            audio=self._audio,
            overlay=self._overlay,
            subwin=self._subwin,
            transcript=self._transcript,
            config=self._config,
            switcher=types.SimpleNamespace(switch=self._switch_asr_engine),
        )

    def _on_target_language_changed(self, lang: str):
        self._pipeline.set_target_language(lang)
        log.info(f"Target language: {lang}")
        if self._panel:
            self._panel.update_settings({"target_language": lang})

    def _on_model_changed(self, model_config: dict):
        log.info(f"Switching translator: {model_config['name']} ({model_config['model']})")
        prompt = None
        if self._panel:
            prompt = self._panel.get_settings().get("system_prompt")
        if not prompt:
            prompt = self._config["translation"].get("system_prompt")
        timeout = 10
        if self._panel:
            timeout = self._panel.get_settings().get("timeout", 10)
        try:
            translator = Translator(
                api_base=model_config["api_base"],
                api_key=model_config["api_key"],
                model=model_config["model"],
                target_language=self._pipeline.target_language,
                max_tokens=self._config["translation"]["max_tokens"],
                temperature=self._config["translation"]["temperature"],
                streaming=model_config.get("streaming", True),
                system_prompt=prompt,
                proxy=model_config.get("proxy", "none"),
                no_system_role=model_config.get("no_system_role", False),
                no_think=model_config.get("no_think", True),
                json_response=model_config.get("json_response", False),
                # CORE-8 migration: models that had json_response before the
                # strict/portable split keep their strict json_schema mode;
                # new models default to the portable json_object mode.
                json_schema_mode=model_config.get(
                    "json_schema_mode", model_config.get("json_response", False)
                ),
                timeout=timeout,
                overrides=model_config.get("overrides"),
                extra_body=model_config.get("extra_body"),
            )
        except Exception as e:
            # Degrade instead of crashing (e.g. the shipped default model has
            # no API key until the wizard fills it in): keep the previous
            # translator and surface the copy in the banner.
            log.warning(f"Translator init failed for {model_config.get('name')}: {e}")
            self._recent_errors.append(t("err_401") if "401" in str(e) else str(e))
            if self._overlay:
                self._overlay.show_error(str(e))
            return
        context_turns = model_config.get(
            "context_turns", self._config["translation"].get("context_window", 0)
        )
        translator.set_context_turns(context_turns)
        self._pipeline.set_translator(translator)
        self._pipeline.set_prices(
            model_config.get("input_price", 0), model_config.get("output_price", 0)
        )

    def _switch_asr_engine(self, engine_type: str):
        ctl = self._asr_ctl
        settings = self._panel.get_settings() if self._panel else {}
        # M-COMPOSE: the pure engine-normalize / model-choice / signature /
        # worker_config / target_state build lives in ui/app_services; the
        # composition root keeps the Qt modal switch (dialog + QTimer poll +
        # rollback) skeleton below.
        plan = build_worker_config(self._config, settings, ctl, engine_type)

        # M-MATRIX: sensevoice-onnx has no model-download flow (the ONNX export
        # and the community artifact both land in models/sensevoice/). If the
        # artifact is absent, degrade with a visible message instead of letting
        # the worker load raise a FileNotFoundError and forcing a rollback.
        if plan.sensevoice_missing:
            message = t("engine_sensevoice_onnx_missing")
            log.warning(f"SenseVoice ONNX model missing: {message}")
            self._recent_errors.append(message)
            if self._overlay:
                self._overlay.show_error(message)
            ctl.refresh_ready()
            return

        if plan.already_ready:
            return
        ctl.refresh_ready()

        log.info(f"Switching ASR worker: {ctl.type} -> {plan.engine_type}")
        # M-COMPOSE: reset the real interim-ASR state in the Pipeline (the
        # old block assigned dead app attributes that no code ever reads).
        self._pipeline.reset_interim()

        parent = self._panel if self._panel and self._panel.isVisible() else self._overlay

        if not plan.cached:
            missing = get_missing_models(plan.engine_type, plan.cache_model_key, plan.hub)
            missing = [m for m in missing if m["type"] != "silero-vad"]
            if missing and not self._smoke:
                dlg = ModelDownloadDialog(
                    missing, hub=plan.hub, proxy=plan.download_proxy, parent=parent
                )
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    log.info(f"Download cancelled/failed: {plan.engine_type}")
                    ctl.refresh_ready()
                    return

        old_asr, old_state = ctl.detach_current()

        dlg = _ModelLoadDialog(t("loading_model").format(name=plan.display_name), parent=parent)

        new_asr = [None]
        restored_asr = [None]
        load_error = [None]
        restore_error = [None]

        def _load():
            if old_asr is not None:
                log.info(f"Stopping old ASR worker before switch: pid={old_asr.pid}")
                old_asr.shutdown()
                self._release_memory_caches()
            try:
                new_asr[0] = ctl.load_engine_client(plan.worker_config)
            except Exception as e:
                load_error[0] = str(e)
                # A remote server that is simply down is an expected, user-actionable
                # condition, not a bug, so skip the noisy traceback for it.
                expected = isinstance(e, ConnectionError)
                log.error(f"Failed to load ASR worker: {e}", exc_info=not expected)
                if old_state and old_state.get("config"):
                    try:
                        log.info("Restoring previous ASR worker after switch failure")
                        restored_asr[0] = ctl.load_engine_client(old_state["config"])
                    except Exception as restore_exc:
                        restore_error[0] = str(restore_exc)
                        log.error(
                            f"Failed to restore previous ASR worker: {restore_exc}",
                            exc_info=True,
                        )

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

        poll_timer = QTimer()

        def _check():
            if not thread.is_alive():
                poll_timer.stop()
                dlg.accept()

        poll_timer.setInterval(100)
        poll_timer.timeout.connect(_check)
        poll_timer.start()

        dlg.exec()
        poll_timer.stop()

        if new_asr[0] is not None:
            ctl.activate(new_asr[0], plan.target_state)
            self._update_asr_status_label(
                f"{plan.display_name} [{plan.target_state['device_label']}]"
            )
            log.info(f"ASR worker ready: {plan.engine_type} on {plan.device}")
            return

        if restored_asr[0] is not None:
            ctl.activate(restored_asr[0], old_state)
            restored_name = old_state.get("display_name") or old_state.get("type")
            self._update_asr_status_label(
                f"{restored_name} [{old_state.get('device_label', old_state['device'])}]"
            )
            QMessageBox.warning(
                parent,
                t("error_title"),
                t("error_load_asr").format(
                    error=(f"{load_error[0] or 'unknown error'}\n{t('asr_restore_succeeded')}")
                ),
            )
            log.info(
                f"Previous ASR worker restored: "
                f"{old_state.get('type')} on {old_state.get('device')}"
            )
            return

        error = load_error[0] or "unknown error"
        if restore_error[0]:
            error = f"{error}\n{t('asr_restore_failed').format(error=restore_error[0])}"
        QMessageBox.warning(
            parent,
            t("error_title"),
            t("error_load_asr").format(error=error),
        )

        self._update_asr_status_label("ASR unavailable")

    def _memory_collab_snapshot(self) -> dict[str, Any]:
        """Collaborator reads for the memory snapshot: ASR worker RSS, GPU
        alloc/reserved, overlay message count, VAD buffer and the pipeline
        counters. The gate snapshot collection stays in the composition root;
        the monitor owns process RSS + threshold/timer formatting."""
        worker_rss_mb = 0.0
        client = self._asr_ctl.client
        if client is not None and client.pid is not None:
            try:
                import psutil

                worker_rss_mb = psutil.Process(client.pid).memory_info().rss / 1024 / 1024
            except Exception:
                worker_rss_mb = 0.0
        gpu_alloc_mb = 0.0
        gpu_reserved_mb = 0.0
        try:
            if torch is not None and torch.cuda.is_available():
                gpu_alloc_mb = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
            elif torch is not None and hasattr(torch, "mps") and torch.backends.mps.is_available():
                # Apple Silicon: MPS has no reserved/allocated split — report
                # the driver allocation as "reserved" and the active as "alloc".
                gpu_alloc_mb = torch.mps.current_allocated_memory() / 1024 / 1024
                gpu_reserved_mb = torch.mps.driver_allocated_memory() / 1024 / 1024
        except Exception:
            pass
        msgs = self._overlay.message_count() if self._overlay else 0
        vad_buf = self._pipeline.vad.buffered_samples() if self._pipeline else 0
        return {
            "worker_rss": worker_rss_mb,
            "gpu_alloc": gpu_alloc_mb,
            "gpu_reserved": gpu_reserved_mb,
            "msgs": msgs,
            "vad_buf": vad_buf,
            "asr_count": self._pipeline.asr_count if self._pipeline else 0,
            "translate_count": self._pipeline.translate_count if self._pipeline else 0,
        }

    def _log_mem_after_asr(self, kind: str, audio_seconds: float, asr_ms: float):
        self._mem_monitor.log_after_asr(kind, audio_seconds, asr_ms)

    def _release_memory_caches(self):
        self._mem_monitor.release_caches()

    def set_memory_warning_callback(self, callback):
        self._mem_monitor.set_warning_callback(callback)

    # --- Lifecycle hooks invoked by the Pipeline ---

    def _on_pipeline_started(self):
        self._mem_monitor.on_pipeline_started()

    def _on_pipeline_stopping(self):
        self._mem_monitor.on_pipeline_stopping()

    def start(self):
        self._audio.start(
            device_id=self._audio_cfg["device"],
            mic_id=None,
            sample_rate=self._audio_cfg["sample_rate"],
            chunk_ms=self._audio_cfg["chunk_ms"],
        )
        self._pipeline.start()

    def stop(self):
        self._pipeline.stop()
        self._audio.stop()

    def pause(self):
        self._pipeline.pause()

    def resume(self):
        self._pipeline.resume()


def main():
    # CI/frozen-build smoke mode (L4 打包冒烟, §7.4): the real startup path
    # runs offscreen with a fresh portable dir (env set in __main__), skips
    # interactive dialogs and quits automatically. Never shown to users.
    _SMOKE = "--smoke" in sys.argv

    setup_logging()
    log.info("LiveTranslate starting...")
    config = load_config()
    config.setdefault("asr", {})
    config["asr"].setdefault("asr_engine", "funasr")
    config["asr"].setdefault("funasr_model", DEFAULT_FUNASR_MODEL)
    saved = load_user_settings()
    migrate_funasr_settings(saved)

    # Log actual effective config
    _asr_eng = (saved or {}).get("asr_engine", config["asr"].get("asr_engine", "funasr"))
    _funasr_model = (saved or {}).get(
        "funasr_model", config["asr"].get("funasr_model", DEFAULT_FUNASR_MODEL)
    )
    _active_idx = (saved or {}).get("active_model", 0)
    _models = (saved or {}).get("models", [])
    if 0 <= _active_idx < len(_models):
        _m = _models[_active_idx]
        _model_info = f"{_m.get('name', '?')} ({_m.get('model', '?')})"
    else:
        _model_info = f"{config['translation']['model']} (default)"
    if _asr_eng == "funasr":
        log.info(f"Config loaded: ASR={_asr_eng}/{_funasr_model}, Translator={_model_info}")
    else:
        log.info(f"Config loaded: ASR={_asr_eng}, Translator={_model_info}")

    # Apply UI language before creating any widgets
    if saved:
        set_lang(resolve_ui_lang(saved.get("ui_lang")))

    os.environ["QT_LOGGING_RULES"] = "qt.text.font.db=false;qt.qpa.fonts.warning=false"
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    # Pin a guaranteed TrueType UI font to avoid DirectWrite failures on the
    # legacy "MS Sans Serif" bitmap font Windows may resolve as the default
    if "Segoe UI" in QFontDatabase.families():
        app.setFont(QFont("Segoe UI", 9))
    _app_icon = create_app_icon()
    app.setWindowIcon(_app_icon)

    # Apply the saved UI theme before any window exists so the first-run
    # wizard, dialogs and popups all resolve the same chrome (the panel
    # switches it live afterwards via set_theme_mode).
    from livetranslate.ui.panel._chrome import DEFAULT_THEME, THEME_MODES, apply_app_theme

    theme = (saved or {}).get("theme")
    apply_app_theme(theme if theme in THEME_MODES else DEFAULT_THEME)

    # Single-instance gate (§3.7): a second launch wakes the running
    # instance (overlay + tray hint) and exits instead of double-capturing.
    _system = create_system_integration()
    _is_primary, _wake = acquire_single_instance("livetranslate", _system)
    if not _is_primary:
        if _wake is WAKE_FAILED:
            # A live lock holder without a responsive wake channel: explain
            # the failure instead of exiting with a bare log line.
            QMessageBox.warning(
                None,
                t("single_instance_blocked_title"),
                t("single_instance_blocked_msg"),
            )
        sys.exit(0)

    # First launch goes straight into the main UI — no forced wizard, missing-
    # model download, or engine integrity gate. All env-fill (model download +
    # engine + mirror/proxy source) happens on-demand in the recognition page;
    # the overlay empty-state guide routes there. Smoke (CI) skips dialogs.
    saved = saved or {}

    log_window = LogWindow()
    log_handler = log_window.get_handler()
    logging.getLogger().addHandler(log_handler)

    # UI-3: first-run fallbacks derive from the platform + detected
    # accelerator instead of a hardcoded cuda default that fails on
    # machines without an NVIDIA GPU. M-MATRIX: the recommendation is a
    # registry id (recommend_engine); resolve it to the worker-frontier
    # engine_type (ENGINE_REGISTRY[id].engine_type) before handing it to the
    # panel — the pane stores/persists the worker-frontier type, and the
    # dropdown restores it through engine_id_for_type(). No GUI-side alias map
    # (sensevoice-onnx is now genuinely selectable, not collapsed onto FunASR).
    from livetranslate.asr.registry import ENGINE_REGISTRY, recommend_engine
    from livetranslate.core.systeminfo import detect_accelerator

    accel = detect_accelerator()
    rec = recommend_engine(accel)  # registry id
    recommended_engine = ENGINE_REGISTRY[rec].engine_type  # worker-frontier type
    panel = ControlPanel(
        config,
        saved_settings=saved,
        recommended_engine=recommended_engine,
        recommended_device="cuda" if accel.kind == "cuda" else "cpu",
    )

    overlay = SubtitleOverlay(config["subtitle"])
    if saved:
        ox = saved.get("overlay_x")
        oy = saved.get("overlay_y")
        ow = saved.get("overlay_w")
        oh = saved.get("overlay_h")
        if ox is not None and oy is not None:
            if SubtitleWindow._is_pos_visible(ox, oy):
                overlay.move(ox, oy)
            else:
                screen = QApplication.primaryScreen()
                geo = screen.availableGeometry()
                overlay.move(
                    geo.right() - overlay.width() - 20, geo.bottom() - overlay.height() - 60
                )
        if ow and oh:
            overlay.resize(ow, oh)
    overlay.set_reduce_motion(bool((saved or {}).get("reduce_motion")))
    if not (saved or {}).get("start_hidden"):
        overlay.show()

    # Subtitle window
    subwin_cfg = (saved or {}).get("subtitle_mode")
    subwin = SubtitleWindow(subwin_cfg)
    subwin.set_reduce_motion(bool((saved or {}).get("reduce_motion")))
    subwin_was_enabled = (subwin_cfg or {}).get("enabled", False)

    live_trans = LiveTranslateApp(config)
    live_trans.set_overlay(overlay)
    live_trans.set_subtitle_window(subwin)
    live_trans.set_panel(panel)
    panel.attach_app(live_trans)

    def _deferred_init():
        panel.apply_settings()
        models = panel.get_settings().get("models", [])
        active_idx = panel.get_settings().get("active_model", 0)
        overlay.set_models(models, active_idx)
        target = panel.get_settings().get("target_language", "zh")
        overlay.set_target_language(target)
        asr_lang = panel.get_settings().get("asr_language", "auto")
        overlay.set_source_language(asr_lang)
        style = panel.get_settings().get("style")
        if style:
            overlay.apply_style(style)
        active_model = panel.get_active_model()
        if active_model:
            live_trans.on_model_changed(active_model)

    QTimer.singleShot(100, _deferred_init)

    tray = QSystemTrayIcon()
    tray.setToolTip(t("tray_tooltip"))
    tray.setIcon(_app_icon)

    # All tray/menu/hotkey/cross-window wiring lives in the app shell.
    shell = build_tray_shell(app, live_trans, overlay, subwin, panel, tray, subwin_was_enabled)

    tray.show()

    # Smoke mode proves the full UI composition path but skips capture
    # start: the pipeline would trigger multi-GB model downloads.
    if not _SMOKE:
        QTimer.singleShot(500, shell.on_start)

    signal.signal(signal.SIGINT, lambda *_: shell.on_quit(confirm=False))
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)

    # L4 packaging smoke (CI): full startup path proven, then clean exit.
    if _SMOKE:
        log.info("Smoke mode: auto-quit in 10s")
        QTimer.singleShot(10000, lambda: shell.on_quit(confirm=False))

    exit_code = app.exec()
    # Detach the Qt log handler before teardown: emitting into a destroyed
    # LogWindow raises from the logging machinery during interpreter exit.
    logging.getLogger().removeHandler(log_handler)
    if _SMOKE:
        if UNCAUGHT_ERRORS:
            log.error(f"Smoke failed: {len(UNCAUGHT_ERRORS)} uncaught error(s)")
            for err in UNCAUGHT_ERRORS:
                log.error(f"  - {err}")
            sys.exit(1)
        log.info("Smoke OK: startup path clean")
    sys.exit(exit_code)
