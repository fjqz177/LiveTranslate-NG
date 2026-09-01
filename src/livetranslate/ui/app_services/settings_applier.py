"""Apply a settings snapshot to the live collaborator objects (M-COMPOSE).

Extracted from ``livetranslate.app``'s ``_on_settings_changed``: a plain
function that routes each settings key to the corresponding collaborator
(VAD / ASR controller / pipeline / audio / overlay / subtitle window /
transcript / base config / engine switcher). Qt-free and cheap to fake-test:
a collaborator is any object exposing the same method names the wiring
expects.
"""

from __future__ import annotations


def apply_settings(
    settings,
    *,
    vad,
    asr_ctl,
    pipeline,
    audio,
    overlay,
    subwin,
    transcript,
    config,
    switcher,
):
    """Route ``settings`` to the live collaborators.

    Reproduces app.py ``_on_settings_changed``: every settings key fans out to
    the object that owns that concern. ``overlay`` and ``subwin`` may be None
    (before the corresponding window is wired up) and are guarded here, exactly
    as the original implementation guarded them.
    """
    vad.update_settings(settings)
    if "style" in settings and overlay:
        overlay.apply_style(settings["style"])
    if "asr_language" in settings:
        asr_ctl.set_language(settings["asr_language"])
        # CORE-11: push the snapshot so the ASR thread never reads the Qt
        # panel's settings dict directly.
        pipeline.set_asr_language(settings["asr_language"])
    if "sensevoice_pad_seconds" in settings:
        # M-MATRIX: key the sensevoice pad on the *current* engine_type so it
        # reaches the ONNX worker (engine_type "sensevoice-onnx") instead of
        # clobbering/losing the legacy FunASR pad. whisper uses its own
        # "whisper" key, so the two never collide.
        engine_type = asr_ctl.type or config["asr"].get("asr_engine", "funasr")
        if engine_type in ("funasr", "sensevoice-onnx"):
            asr_ctl.set_padding(engine_type, settings["sensevoice_pad_seconds"])
    if "whisper_pad_seconds" in settings:
        asr_ctl.set_padding("whisper", settings["whisper_pad_seconds"])
    if any(
        key in settings
        for key in (
            "asr_engine",
            "asr_device",
            "whisper_model_size",
            "funasr_model",
            "hub",
        )
    ):
        switcher.switch(
            settings.get(
                "asr_engine",
                asr_ctl.type or config["asr"].get("asr_engine", "funasr"),
            )
        )
    if "audio_device" in settings:
        old_device = audio.device_id
        audio.switch_device(settings["audio_device"])
        if old_device != settings.get("audio_device"):
            vad.reset()
            if overlay:
                overlay.update_monitor(0.0, 0.0)
    if "mic_device" in settings:
        audio.switch_mic(settings["mic_device"])
    if "incremental_asr" in settings:
        pipeline.set_incremental(settings["incremental_asr"])
    if "interim_interval" in settings:
        pipeline.set_interim_interval(settings["interim_interval"])
    if "target_language" in settings:
        pipeline.set_target_language(settings["target_language"])
        if overlay:
            overlay.set_target_language(settings["target_language"])
    if "timeout" in settings:
        pipeline.translator.set_timeout(settings["timeout"])
    if "auto_save_transcript" in settings:
        transcript.set_enabled(settings["auto_save_transcript"])
    if "log_transcript" in settings:
        pipeline.set_log_transcript(bool(settings["log_transcript"]))
    if "reduce_motion" in settings:
        reduced = bool(settings["reduce_motion"])
        if overlay:
            overlay.set_reduce_motion(reduced)
        if subwin:
            subwin.set_reduce_motion(reduced)
