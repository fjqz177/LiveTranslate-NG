"""Engine-variant install / orchestration for the VAD/ASR tab (识别页 §3.2).

M-SPLIT (2026-08-31): extracted verbatim from ``vad_tab.VadTab`` into a
plain-object mixin. The concrete ``VadTab`` keeps the ``engine_install_finished``
and ``runtime_progress`` signals plus the widget state; this mixin holds only
the engine-install / variant-switch / mirror-select methods. Methods run on a
concrete ``VadTab`` instance, so ``self.settings`` (TabBase property) and the
signal attributes remain reachable.

The ``@pyqtSlot()`` decorator on ``_on_engine_install_finished`` was dropped —
its only connect point is a plain callable and a pyqtSlot on a plain-object
mixin risks metaclass / signal-registration trouble.
"""

from PyQt6.QtWidgets import QMessageBox

from livetranslate.core.i18n import t


class _EngineRuntimeMixin:
    """Engine-install / variant orchestration, mixed into ``VadTab``."""

    def _refresh_engine_status(self) -> None:
        import sys as _sys

        from livetranslate.asr.availability import engine_status
        from livetranslate.asr.registry import ENGINE_REGISTRY
        from livetranslate.core import engine_runtime
        from livetranslate.core.systeminfo import detect_accelerator, detect_variant

        engine_id = self._selected_engine_id()
        spec = ENGINE_REGISTRY.get(engine_id)
        if spec is None:
            return
        status = engine_status(engine_id, _sys.platform)
        active = engine_runtime.active_variant()
        accel = detect_accelerator()
        variant = active or detect_variant()
        hardware = accel.device_name or t("runtime_hardware_cpu")

        # P4: engine_status() probes the GUI import spec, which never sees
        # the engine venv (site-packages are injected only into the ASR
        # worker). A venv-backed engine with an active variant is installed —
        # don't keep reporting needs-extras (which re-fires the install
        # button and loops on "variant already installed").
        if status == "needs-extras" and active:
            status = "available"

        if status == "available":
            text = t("engine_status_available")
        elif status == "not-implemented":
            text = t("engine_status_not_implemented")
        elif status == "needs-extras":
            text = t("engine_status_needs_extras").format(size=spec.download_gb)
        elif status == "needs-model":
            # M-MATRIX honesty: sensevoice-onnx has NO auto-downloader (the
            # ONNX model is exported / community-provided), so a needs-model
            # status must not promise "the model downloads automatically on
            # switch" on the flagship CPU-recommended path. Branch on the
            # engine id for the honest export/community copy; every other
            # engine keeps the generic text (matches app.py's modal switch
            # path, which already uses engine_sensevoice_onnx_missing).
            text = (
                t("engine_sensevoice_onnx_missing")
                if engine_id == "sensevoice-onnx"
                else t("engine_status_needs_model")
            )
        else:
            text = t("engine_status_unsupported")
        if active:
            text += f"  ·  {t('runtime_variant')}: {active}  ·  {t('runtime_hardware')}: {hardware}"
        elif not getattr(_sys, "frozen", False):
            text += f"  ·  {t('runtime_hardware')}: {hardware}"
        self._engine_status_label.setText(text)

        frozen = bool(getattr(_sys, "frozen", False))
        # Install button: frozen builds need the engine venv (install the
        # recommended variant); dev builds sync uv extras.
        if status == "needs-extras":
            self._engine_install_btn.setEnabled(True)
            self._engine_install_btn.setToolTip("")
            self._engine_install_btn.setText(
                t("runtime_install_variant").format(variant=variant)
                if frozen
                else t("btn_install_engine")
            )
            self._engine_install_btn.show()
        else:
            self._engine_install_btn.hide()
        # Switch/remove: meaningful only when at least one venv exists.
        if frozen and active:
            self._runtime_switch_btn.setEnabled(False)  # single-variant MVP: switching
            self._runtime_switch_btn.setToolTip(t("runtime_switch_hint"))
            self._runtime_switch_btn.show()
            self._runtime_remove_btn.show()
        else:
            self._runtime_switch_btn.hide()
            self._runtime_remove_btn.hide()

    def _on_mirror_changed(self) -> None:
        s = self.settings
        prefs = dict(s.get("engine_runtime") or {})
        prefs["mirror"] = self._runtime_mirror.currentData()
        s["engine_runtime"] = prefs
        self.auto_save()

    def _on_torch_mirror_changed(self) -> None:
        s = self.settings
        prefs = dict(s.get("engine_runtime") or {})
        prefs["torch_mirror"] = self._torch_mirror.currentData()
        s["engine_runtime"] = prefs
        self.auto_save()

    def _install_engine_deps(self) -> None:
        import sys as _sys

        if not getattr(_sys, "frozen", False):
            self._install_dev_extras()
            return
        self._open_engine_install_popup()

    def _open_engine_install_popup(self) -> None:
        """Frozen: open the engine-install dialog (live output) instead of an
        inline thread. A completed install is picked up by the worker on the
        next start; closing (cancel) aborts a running install so the engine
        area is never stranded in ``installing``."""
        from livetranslate.core.systeminfo import detect_variant
        from livetranslate.ui.dependency_dialog import EngineBootstrapDialog

        prefs = self.settings.get("engine_runtime") or {}
        dlg = EngineBootstrapDialog(
            detect_variant(),
            mirror=prefs.get("mirror", "auto"),
            torch_mirror=prefs.get("torch_mirror", "official"),
            parent=self.panel,
        )
        dlg.exec()
        self._refresh_engine_status()

    def _install_dev_extras(self) -> None:
        import shutil
        import subprocess
        import threading

        from livetranslate.asr.availability import EXTRAS_PROBE_MAP, extras_installed
        from livetranslate.asr.registry import ENGINE_REGISTRY
        from livetranslate.core.paths import PROJECT_ROOT

        if shutil.which("uv") is None:
            QMessageBox.warning(self, t("btn_install_engine"), t("engine_install_needs_uv"))
            return
        spec = ENGINE_REGISTRY[self._selected_engine_id()]
        extras = list(spec.extras)
        # uv sync prunes unlisted groups — keep every already-installed extra.
        for other in EXTRAS_PROBE_MAP:
            if other not in extras and extras_installed(EXTRAS_PROBE_MAP[other]):
                extras.append(other)
        cmd = ["uv", "sync"] + [f"--extra={e}" for e in extras]
        self._install_ok = False
        self._engine_install_btn.setEnabled(False)
        self._engine_install_btn.setText(t("engine_installing"))

        def _run() -> None:
            # uv emits UTF-8; locale decoding (GBK) can crash the reader
            # thread and leave stderr None — decode as UTF-8 with replacement.
            create = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                creationflags=create,
            )
            self._install_ok = proc.returncode == 0
            self._install_tail = (proc.stderr or "")[-400:]
            self.engine_install_finished.emit()

        threading.Thread(target=_run, daemon=True).start()

    def _switch_variant(self) -> None:
        from livetranslate.core import engine_runtime

        variants = engine_runtime.installed_variants()
        if len(variants) < 2:
            QMessageBox.information(self, t("runtime_btn_switch"), t("runtime_switch_hint"))
            return
        current = engine_runtime.active_variant()
        target = next((v for v in variants if v != current), None)
        if target:
            engine_runtime.activate(target)
            self._refresh_engine_status()
            QMessageBox.information(self, t("runtime_btn_switch"), t("runtime_restart_hint"))

    def _remove_variant(self) -> None:
        from livetranslate.core import engine_runtime

        active = engine_runtime.active_variant()
        if not active:
            return
        answer = QMessageBox.question(
            self,
            t("runtime_btn_remove"),
            t("runtime_remove_confirm").format(variant=active),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        engine_runtime.remove_variant(active)
        self._refresh_engine_status()

    def _on_engine_install_finished(self) -> None:
        self._engine_install_btn.setText(t("btn_install_engine"))
        self._refresh_engine_status()
        if not self._install_ok:
            QMessageBox.warning(
                self,
                t("btn_install_engine"),
                t("engine_install_failed") + chr(10) + str(getattr(self, "_install_tail", "")),
            )
