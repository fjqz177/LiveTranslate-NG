"""Diagnostics: platform capability, audio chain, hotkeys, permissions,
accelerator, storage and logs (plan §3.4).

DiagnosticsView is the reusable seven-card widget (embedded in the
settings panel's diagnostics page); DiagnosticsDialog wraps the same
view with a close button for the tray entry. Live state is read through
a duck-typed app reference so the shell can pass the running composition
root without new couplings.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from livetranslate.core import diagnostics
from livetranslate.core.i18n import t
from livetranslate.core.paths import CONFIG_DIR, LOG_DIR, models_dir, transcripts_dir
from livetranslate.core.systeminfo import detect_accelerator
from livetranslate.platform.permissions import NullPermissionStatus

log = logging.getLogger("LiveTranslate.Diagnostics")


class DiagnosticsView(QWidget):
    """Seven-card live diagnostics view (plan §3.4)."""

    def __init__(
        self,
        app_ref: Any,
        parent: QWidget | None = None,
        with_actions: bool = True,
    ) -> None:
        super().__init__(parent)
        self._app = app_ref
        self._summary: dict[str, Any] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # Keep the viewport unfilled so the themed surface shows through.
        # No inline stylesheet here: a selector-less rule would match every
        # widget in the subtree, painting popups/dialogs transparent/black.
        scroll.viewport().setAutoFillBackground(False)
        content = QWidget()
        self._cards = QVBoxLayout(content)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        self._build_cards()
        if with_actions:
            layout.addLayout(self._action_row())
        self._refresh_summary()

    # -- cards ---------------------------------------------------------------

    def _build_cards(self) -> None:
        self._cards.addWidget(self._card_platform())
        self._cards.addWidget(self._card_network())
        self._cards.addWidget(self._card_audio())
        self._cards.addWidget(self._card_hotkeys())
        self._cards.addWidget(self._card_permissions())
        self._cards.addWidget(self._card_accelerator())
        self._cards.addWidget(self._card_storage())
        self._cards.addWidget(self._card_logs())
        self._cards.addStretch(1)

    @staticmethod
    def _group(title: str) -> QGroupBox:
        box = QGroupBox(title)
        box.setLayout(QFormLayout())
        return box

    @staticmethod
    def _add_rows(box: QGroupBox, rows: list[tuple[str, str]]) -> None:
        form = box.layout()
        for label, value in rows:
            value_label = QLabel(value)
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(label, value_label)

    def _card_platform(self) -> QGroupBox:
        box = self._group(t("diag_platform"))
        audio = self._app._audio if hasattr(self._app, "_audio") else None
        mem_warning = getattr(self._app, "_last_memory_warning", None)
        mem_text = f"{mem_warning:.0f} MB" if mem_warning else t("diag_mem_none")
        tray_ok = QSystemTrayIcon.isSystemTrayAvailable()
        rows = [
            (t("diag_os"), sys.platform),
            (t("diag_audio_backend"), audio.name if audio is not None else "unset"),
            (t("diag_wayland"), "yes" if "WAYLAND_DISPLAY" in os.environ else "no"),
            (t("diag_tray"), t("diag_yes") if tray_ok else t("diag_no")),
            (t("diag_mem_status"), mem_text),
        ]
        self._add_rows(box, rows)
        return box

    def _card_network(self) -> QGroupBox:
        """§3.4 翻译/网络卡: model/API (脱敏), proxy mode, recent errors."""
        box = self._group(t("diag_network"))
        panel = getattr(self._app, "_panel", None)
        model = (
            panel.get_active_model()
            if panel is not None and hasattr(panel, "get_active_model")
            else None
        )
        if model:
            proxy = model.get("proxy", "none") or "none"
            proxy_labels = {
                "none": t("proxy_none"),
                "system": t("proxy_system"),
                "custom": t("proxy_custom"),
            }
            rows = [
                (
                    t("diag_model"),
                    f"{model.get('name', '?')} ({model.get('model', '?')})",
                ),
                (
                    t("diag_api_base"),
                    diagnostics.redact_text(str(model.get("api_base", ""))),
                ),
                (t("diag_proxy_mode"), proxy_labels.get(proxy, proxy)),
            ]
        else:
            rows = [(t("diag_model"), "unset")]
        errors = list(getattr(self._app, "_recent_errors", ()))
        rows.append((t("diag_recent_errors"), chr(10).join(errors) if errors else t("diag_none")))
        self._add_rows(box, rows)
        return box

    def _card_audio(self) -> QGroupBox:
        box = self._group(t("diag_audio_chain"))
        audio = self._app._audio if hasattr(self._app, "_audio") else None
        if audio is not None and hasattr(audio, "diagnostics"):
            diag = audio.diagnostics()
            rows = [(str(k), str(v)) for k, v in diag.items()]
        else:
            rows = [(t("diag_status"), "unavailable")]
        self._add_rows(box, rows)
        return box

    def _card_hotkeys(self) -> QGroupBox:
        box = self._group(t("diag_hotkeys"))
        hotkeys = self._app._hotkeys if hasattr(self._app, "_hotkeys") else None
        rows = [(name, str(combo)) for name, combo in getattr(hotkeys, "_combos", {}).items()]
        if not rows:
            rows = [(t("diag_status"), t("diag_none_registered"))]
        self._add_rows(box, rows)
        return box

    def _card_permissions(self) -> QGroupBox:
        box = self._group(t("diag_permissions"))
        permission = NullPermissionStatus()
        rows = [
            (t("diag_perm_mic"), permission.microphone()),
            (t("diag_perm_screen"), permission.screen_recording()),
            (t("diag_perm_access"), permission.accessibility()),
        ]
        self._add_rows(box, rows)
        return box

    def _card_accelerator(self) -> QGroupBox:
        box = self._group(t("diag_accelerator"))
        accel = detect_accelerator()
        engine = getattr(self._app, "_asr_ctl", None)
        engine_type = getattr(engine, "type", None) or ""
        from livetranslate.asr.registry import ENGINE_REGISTRY, recommend_engine

        def _display_name(engine_id: str) -> str:
            spec = ENGINE_REGISTRY.get(engine_id)
            return spec.display_name if spec is not None else engine_id

        engine_id = _display_name(engine_type) if engine_type else t("diag_engine_unset")
        try:
            recommended = _display_name(recommend_engine(accel))
        except Exception:
            recommended = ""
        rows = [
            (t("diag_accel_kind"), accel.display),
            (t("diag_engine"), engine_id),
            (t("diag_engine_recommended"), recommended),
        ]
        self._add_rows(box, rows)
        return box

    def _card_storage(self) -> QGroupBox:
        box = self._group(t("diag_storage"))
        rows = [
            (t("diag_config_dir"), str(CONFIG_DIR)),
            (t("diag_models_dir"), str(models_dir())),
            (t("diag_transcripts_dir"), str(transcripts_dir())),
            (t("diag_log_dir"), str(LOG_DIR)),
        ]
        self._add_rows(box, rows)
        return box

    def _card_logs(self) -> QGroupBox:
        box = self._group(t("diag_logs"))
        log_files = sorted(LOG_DIR.glob("*.log")) if LOG_DIR.is_dir() else []
        if log_files:
            tail = chr(10).join(
                log_files[-1].read_text(encoding="utf-8", errors="replace").splitlines()[-3:]
            )
            rows = [(log_files[-1].name, tail)]
        else:
            rows = [(t("diag_status"), t("diag_no_logs"))]
        self._add_rows(box, rows)
        return box

    # -- actions ---------------------------------------------------------------

    def _action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        copy_btn = QPushButton(t("diag_copy_summary"))
        copy_btn.clicked.connect(self._copy_summary)
        row.addWidget(copy_btn)
        zip_btn = QPushButton(t("diag_pack_zip"))
        zip_btn.clicked.connect(self._pack_zip)
        row.addWidget(zip_btn)
        row.addStretch(1)
        return row

    def _refresh_summary(self) -> None:
        audio = self._app._audio if hasattr(self._app, "_audio") else None
        audio_diag = (
            audio.diagnostics() if audio is not None and hasattr(audio, "diagnostics") else None
        )
        hotkeys = self._app._hotkeys if hasattr(self._app, "_hotkeys") else None
        hotkey_rows = {name: str(combo) for name, combo in getattr(hotkeys, "_combos", {}).items()}
        settings = self._app.get_settings() if hasattr(self._app, "get_settings") else None
        self._summary = diagnostics.collect_summary(
            platform=sys.platform,
            accelerator=detect_accelerator(),
            audio_diag=audio_diag,
            hotkey_status=hotkey_rows,
            permission=NullPermissionStatus(),
            settings=settings,
        )

    def _copy_summary(self) -> None:
        self._refresh_summary()
        QApplication.clipboard().setText(diagnostics.session_id() + chr(10) + str(self._summary))

    def _pack_zip(self) -> None:
        default = str(Path.home() / f"livetranslate-diagnostics-{diagnostics.session_id()}.zip")
        path, _ = QFileDialog.getSaveFileName(self, t("diag_pack_zip"), default, "Zip (*.zip)")
        if not path:
            return
        try:
            report = diagnostics.write_redacted_zip(Path(path))
            QMessageBox.information(
                self,
                t("diag_pack_zip"),
                t("diag_zip_done").format(files=report["files"], size=report["size_bytes"]),
            )
        except OSError as e:
            QMessageBox.warning(self, t("diag_pack_zip"), str(e))


class DiagnosticsDialog(QDialog):
    """Tray entry: the shared diagnostics view plus a close button."""

    def __init__(self, app_ref: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("diag_title"))
        self.resize(660, 640)
        self._view = DiagnosticsView(app_ref, parent=self)

        layout = QVBoxLayout(self)
        layout.addWidget(self._view, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        close_btn = QPushButton(t("close"))
        close_btn.clicked.connect(self.accept)
        row.addWidget(close_btn)
        layout.addLayout(row)

        # Back-compat accessors (tests reach the card list directly)
        self._cards = self._view._cards

    def _refresh_summary(self) -> None:
        self._view._refresh_summary()

    @property
    def _summary(self) -> dict[str, Any]:
        return self._view._summary
