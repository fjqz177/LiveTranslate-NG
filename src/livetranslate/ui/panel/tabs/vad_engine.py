"""Engine availability display for the VAD/ASR tab (识别页 §3.2).

Full-install model (2026-09-01): the engine dependencies (torch /
faster-whisper / funasr) are installed together with the app via pyappify
into the main environment, so there is no runtime engine install, no
variant switching, no mirror selection and no pythonpaths injection. This
mixin only renders the engine availability status.
"""

from livetranslate.core.i18n import t


class _EngineRuntimeMixin:
    """Engine availability-only mixin, mixed into ``VadTab``."""

    def _refresh_engine_status(self) -> None:
        import sys as _sys

        from livetranslate.asr.availability import engine_status
        from livetranslate.asr.registry import ENGINE_REGISTRY

        engine_id = self._selected_engine_id()
        spec = ENGINE_REGISTRY.get(engine_id)
        if spec is None:
            return
        status = engine_status(engine_id, _sys.platform)

        if status == "available":
            text = t("engine_status_available")
        elif status == "not-implemented":
            text = t("engine_status_not_implemented")
        elif status == "needs-model":
            # M-MATRIX honesty: sensevoice-onnx has NO auto-downloader (the
            # ONNX model is exported / community-provided), so keep the honest
            # export/community copy; every other engine keeps the generic text
            # (matches app.py's modal switch path, which already uses
            # engine_sensevoice_onnx_missing).
            text = (
                t("engine_sensevoice_onnx_missing")
                if engine_id == "sensevoice-onnx"
                else t("engine_status_needs_model")
            )
        else:
            text = t("engine_status_unsupported")
        self._engine_status_label.setText(text)
