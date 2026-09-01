"""Engine runtime install dialog (recognition page, on-demand).

Replaces both the old first-launch integrity gate and the two-stage
"download-in-wizard + separate RuntimeInstallDialog" flow. Opened from the
recognition page's "install engine" button: it runs the embedded-uv variant
install with live progress streaming into a scrollable output panel,
auto-selects the recommended variant (switchable), offers retry on failure and
never silently skips — the user sees every stage and can close (cancel) any
time, which aborts a running install so the engine area is never stranded.
Model download stays on the unified missing-model check (ModelDownloadDialog).
"""

from __future__ import annotations

import logging
import threading

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from livetranslate.core.i18n import t

log = logging.getLogger("LiveTranslate.EngineBootstrap")

# Rough installed sizes per variant (uncompressed); shown as a hint only.
_VARIANT_SIZE_HINT = {"cpu": "1.5 GB", "cu126": "4.5 GB"}

# Indirection for tests: real path installs the variant; a test swaps this to
# a fake to drive __done__/__failed__ without touching the network or disk.
_install_variant = None  # resolved lazily in _start_install (avoids uv_runner import here)


def _resolve_install_variant():
    """Real install path: uv_runner.install_variant, imported on demand."""
    from livetranslate.core.uv_runner import install_variant

    return install_variant


class EngineBootstrapDialog(QDialog):
    """Blocking install dialog: auto-detected (switchable) variant + progress.

    Opened on-demand from the recognition page (and formerly the first-launch
    gate): the embedded-uv install runs in a worker thread, stage progress and
    errors stream into a scrollable output panel, and the user can close it any
    time (a running install is aborted so the engine area is never stranded in
    the ``installing`` state).
    """

    _progress_signal = pyqtSignal(str)

    def __init__(
        self,
        variant: str,
        mirror: str = "auto",
        torch_mirror: str = "official",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(t("runtime_gate_title"))
        self.setMinimumWidth(520)
        self._variants = ("cpu", "cu126")
        self._mirror = mirror
        self._torch_mirror = torch_mirror
        self._ok = False
        self._error = ""
        self._thread: threading.Thread | None = None
        self._variant = ""
        self._cancelled = False

        layout = QVBoxLayout(self)
        intro = QLabel(
            t("runtime_gate_intro").format(
                variant=variant, size=_VARIANT_SIZE_HINT.get(variant, "?")
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Switchable target variant: auto-detected default, user may pick.
        row = QHBoxLayout()
        row.addWidget(QLabel(t("runtime_variant")))
        self._variant_combo = QComboBox()
        for v in self._variants:
            self._variant_combo.addItem(f"{v} ({_VARIANT_SIZE_HINT.get(v, '?')})", v)
        idx = self._variant_combo.findData(variant)
        self._variant_combo.setCurrentIndex(max(idx, 0))
        row.addWidget(self._variant_combo, 1)
        layout.addLayout(row)

        self._status = QLabel(t("runtime_gate_starting"))
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)  # indeterminate while the install runs
        self._bar.hide()
        layout.addWidget(self._bar)

        # Scrollable live output: every stage message and the uv error tail
        # land here so the user sees real progress, not just a spinner.
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setObjectName("installLog")
        self._log.setMaximumHeight(150)
        self._log.hide()
        layout.addWidget(self._log)

        buttons = QHBoxLayout()
        self._retry_btn = QPushButton(t("runtime_gate_retry"))
        self._retry_btn.clicked.connect(self._start_install)
        self._retry_btn.hide()
        buttons.addWidget(self._retry_btn)
        buttons.addStretch(1)
        # On-demand popup (recognition page): closing is a cancel, not a way
        # past a blocking gate, so "Close" reads better than "稍后再说".
        self._quit_btn = QPushButton(t("btn_close"))
        self._quit_btn.clicked.connect(self.reject)
        buttons.addWidget(self._quit_btn)
        layout.addLayout(buttons)

        self._progress_signal.connect(self._on_progress)
        QTimer.singleShot(100, self._start_install)

    def _start_install(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._retry_btn.hide()
        self._error = ""
        self._status.setText(t("runtime_gate_starting"))
        self._bar.setRange(0, 0)
        self._bar.show()
        variant = self._variant_combo.currentData()
        self._variant = variant
        self._variant_combo.setEnabled(False)

        def _run() -> None:
            from livetranslate.core.version import app_version

            # Real install resolves lazily; tests inject a fake via _install_variant.
            installer = _install_variant or _resolve_install_variant()
            try:
                installer(
                    variant,
                    app_version=app_version(),
                    pypi_mirror=self._mirror,
                    torch_mirror=self._torch_mirror,
                    progress_cb=self._progress_signal.emit,
                )
                self._ok = True
                if self._cancelled:
                    return
                self._progress_signal.emit("__done__")
            except Exception as exc:  # UvRunnerError etc. -> visible retry
                log.error("runtime install failed: %s", exc)
                if self._cancelled:
                    return
                self._error = str(exc)[-400:]
                self._progress_signal.emit("__failed__")

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _on_progress(self, message: str) -> None:
        if message == "__done__":
            self.accept()
            return
        if message == "__failed__":
            self._status.setText(
                t("runtime_gate_failed").format(error=self._error)
                + "\n"
                + t("runtime_gate_full_package")
            )
            self._retry_btn.show()
            self._bar.hide()
            self._append_log(self._error)
            return
        # Live stage text + a bounded busy bar once pip installs (the heavy,
        # bounded part) — a real progress readout rather than a silent label.
        self._status.setText(message)
        self._append_log(message)
        if any(k in message.lower() for k in ("pip install", "installing", "wheels")):
            self._bar.setRange(0, 0)

    def _append_log(self, text: str) -> None:
        """Append a line to the scrollable output panel (shown on first use)."""
        if not text:
            return
        self._log.show()
        self._log.appendPlainText(text)

    def reject(self) -> None:
        """'稍后再说': cancel a running install so state is not stuck installing."""
        self._cancelled = True
        if self._thread is not None and self._thread.is_alive() and self._variant:
            try:
                from livetranslate.core import engine_runtime as er

                er.abort_install(self._variant)
            except Exception:
                log.exception("abort engine install on reject failed")
        super().reject()
