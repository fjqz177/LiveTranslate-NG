"""Pure worker-config build for the ASR engine switch (M-COMPOSE).

Extracted from ``livetranslate.app``'s ``_switch_asr_engine`` so the
engine-normalize / device / hub / model-choice / signature / display_name /
worker_config / target_state decision can be unit-tested without the Qt
dialog + QTimer + rollback modal skeleton (which stays in the composition
root). Qt-free; the only collaborators it reads are ``modeling.manager`` and
``asr.availability``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from livetranslate.asr.availability import sensevoice_onnx_model_present
from livetranslate.modeling.manager import (
    ASR_DISPLAY_NAMES,
    MODELS_DIR,
    funasr_display_name,
    is_asr_cached,
    local_faster_whisper_display_name,
    normalize_asr_engine_selection,
    resolve_custom_whisper_model,
)


@dataclass
class WorkerSwitchPlan:
    """Pure result of deciding *what* the next ASR worker should be.

    ``sensevoice_missing`` / ``already_ready`` are decision flags that let the
    caller reproduce app.py's early-return gate (sensevoice-onnx artifact
    absent) and signature short-circuit (the target engine already matches the
    running worker). The rest is the concrete ``worker_config`` /
    ``target_state`` handed to the controller, plus the pieces the caller needs
    for the download-dialog decision (``hub`` / ``download_proxy`` /
    ``cache_model_key`` / ``cached``).
    """

    engine_type: str
    funasr_model: str
    device: str
    signature: tuple
    display_name: str
    worker_config: dict
    target_state: dict
    hub: str
    download_proxy: str
    cache_model_key: str
    cached: bool
    sensevoice_missing: bool = False
    already_ready: bool = False


def build_worker_config(config, settings, ctl, engine_type: str) -> WorkerSwitchPlan:
    """Compute the worker-config plan for switching to ``engine_type``.

    Reproduces the pure-build portion of app.py's ``_switch_asr_engine``:
    engine normalize (via ``modeling.manager``), device / hub / download_proxy
    defaults, the whisper-vs-funasr model choice, remote url|token identity,
    signature, display_name, ``worker_config`` (conditional pad_seconds +
    download_root) and ``target_state``. The Qt modal dialog, the
    QTimer poll and the rollback skeleton are *not* part of this function.
    """
    engine_type, funasr_model = normalize_asr_engine_selection(
        engine_type,
        settings.get("funasr_model", ctl.funasr_model_key),
    )

    # M-MATRIX: sensevoice-onnx has no model-download flow (the ONNX export and
    # the community artifact both land in models/sensevoice/). Surface the
    # absence as a decision flag; the caller shows the message.
    sensevoice_missing = engine_type == "sensevoice-onnx" and not sensevoice_onnx_model_present()

    device = settings.get("asr_device", ctl.device)
    hub = settings.get("hub", "ms")
    download_proxy = settings.get("download_proxy", "system")

    model_size = config["asr"]["model_size"]
    model_size = settings.get("whisper_model_size", model_size)
    model_path = None
    cache_model_key = model_size
    if engine_type == "whisper":
        model_path = resolve_custom_whisper_model(model_size)
        if model_path:
            cache_model_key = model_path
    elif engine_type == "funasr":
        cache_model_key = funasr_model

    remote_asr_url = settings.get(
        "remote_asr_url",
        config["asr"].get("remote_asr_url", "http://127.0.0.1:8765"),
    )
    remote_asr_token = settings.get("remote_asr_token", "") or ""

    compute = config["asr"]["compute_type"]
    if engine_type == "whisper":
        signature_model = cache_model_key
    elif engine_type == "funasr":
        signature_model = funasr_model
    elif engine_type == "remote-whisper":
        # URL + token are part of the identity so editing either one triggers a
        # reconnect.
        signature_model = f"{remote_asr_url}|{remote_asr_token}"
    else:
        signature_model = engine_type
    signature = (engine_type, signature_model, device, hub, compute)

    already_ready = ctl.is_ready_with_signature(signature)

    cached = is_asr_cached(engine_type, cache_model_key, hub)

    display_name = ASR_DISPLAY_NAMES.get(engine_type, engine_type)
    if engine_type == "whisper":
        display_model = (
            local_faster_whisper_display_name(model_size) if model_path else model_size
        ) or Path(model_size).name
        display_name = f"Whisper {display_model}"
    elif engine_type == "funasr":
        display_name = funasr_display_name(funasr_model)

    pad_seconds = None
    if engine_type in ("funasr", "sensevoice-onnx"):
        pad_seconds = settings.get(
            "sensevoice_pad_seconds",
            config["asr"].get("sensevoice_pad_seconds", 0.5),
        )
    elif engine_type == "whisper":
        pad_seconds = settings.get(
            "whisper_pad_seconds",
            config["asr"].get("whisper_pad_seconds", 0.5),
        )

    worker_config = {
        "engine_type": engine_type,
        "funasr_model": funasr_model,
        "model_size": cache_model_key,
        "device": device,
        "compute_type": compute,
        "hub": hub,
        "language": settings.get("asr_language", config["asr"].get("language", "auto")),
        "pad_seconds": pad_seconds,
        "download_root": str((MODELS_DIR / "huggingface" / "hub").resolve()),
        "display_name": display_name,
        "remote_asr_url": remote_asr_url,
        "remote_asr_token": remote_asr_token,
    }

    target_state = {
        "type": engine_type,
        "signature": signature,
        "device": device,
        "funasr_model_key": funasr_model if engine_type == "funasr" else ctl.funasr_model_key,
        "whisper_model_size": model_size if engine_type == "whisper" else ctl.whisper_model_size,
        "config": worker_config,
        "display_name": display_name,
        "device_label": (remote_asr_url if engine_type == "remote-whisper" else device),
    }

    return WorkerSwitchPlan(
        engine_type=engine_type,
        funasr_model=funasr_model,
        device=device,
        signature=signature,
        display_name=display_name,
        worker_config=worker_config,
        target_state=target_state,
        hub=hub,
        download_proxy=download_proxy,
        cache_model_key=cache_model_key,
        cached=cached,
        sensevoice_missing=sensevoice_missing,
        already_ready=already_ready,
    )
