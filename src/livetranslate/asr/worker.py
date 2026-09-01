import contextlib
import gc
import importlib
import logging
import os
import sys

import numpy as np

from livetranslate.asr.protocol import error_response, ok_response

log = logging.getLogger("LiveTranslate.ASRWorker")


class ConfigError(ValueError):
    """Invalid ASR worker configuration."""


def _setup_logging():
    if logging.getLogger().handlers:
        return
    # Frozen windowed workers have sys.stdout = None — a StreamHandler bound
    # to None crashes on every emit. Keep logs dropped (NullHandler) there;
    # the parent process surfaces engine errors via the wire protocol.
    if sys.stdout is None:
        handler: logging.Handler = logging.NullHandler()
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logging.getLogger("LiveTranslate").setLevel(logging.DEBUG)


def _parse_device(device: str) -> tuple[str, int]:
    device = str(device or "cpu").split(" (", 1)[0].strip()
    if device.startswith("cuda:"):
        index = int(device.split(":", 1)[1])
        return "cuda", index
    return device, 0


def _load_whisper(config: dict):
    from livetranslate.asr.engines.whisper import ASREngine
    from livetranslate.modeling.manager import (
        MODELS_DIR,
        get_whisper_local_path,
    )

    parsed_device, device_index = _parse_device(config.get("device", "cpu"))
    compute_type = config.get("compute_type", "float16")
    if parsed_device == "cpu" and compute_type == "float16":
        compute_type = "int8"
    download_root = config.get("download_root")
    if not download_root:
        download_root = str((MODELS_DIR / "huggingface" / "hub").resolve())
    model_size = config["model_size"]
    # Load builtin sizes straight from the local cache (ModelScope or HF) so
    # faster-whisper never falls back to its own HuggingFace download; custom
    # model paths pass through untouched.
    local_path = get_whisper_local_path(model_size, hub=config.get("hub", "ms"))
    if local_path:
        log.info(f"Loading Whisper from local cache: {local_path}")
        model_size = local_path
    return ASREngine(
        model_size=model_size,
        device=parsed_device,
        device_index=device_index,
        compute_type=compute_type,
        download_root=download_root,
        pad_seconds=config.get("pad_seconds"),
    )


def _load_funasr(config: dict):
    from livetranslate.asr.engines.funasr import FunASREngine

    return FunASREngine(
        model_key=config.get("funasr_model"),
        device=config.get("device", "cpu"),
        hub=config.get("hub", "ms"),
        pad_seconds=config.get("pad_seconds"),
    )


def _load_anime_whisper(config: dict):
    from livetranslate.asr.engines.anime_whisper import AnimeWhisperEngine

    parsed_device, device_index = _parse_device(config.get("device", "cpu"))
    worker_device = parsed_device if parsed_device == "cpu" else f"cuda:{device_index}"
    return AnimeWhisperEngine(device=worker_device, hub=config.get("hub", "ms"))


def _load_sensevoice_onnx(config: dict):
    from livetranslate.asr.engines.sensevoice_onnx import SenseVoiceOnnxEngine

    return SenseVoiceOnnxEngine(
        model_path=config.get("onnx_model_path"),
        pad_seconds=config.get("pad_seconds"),
    )


_ENGINE_FACTORIES = {
    "whisper": _load_whisper,
    "funasr": _load_funasr,
    "anime-whisper": _load_anime_whisper,
    "sensevoice-onnx": _load_sensevoice_onnx,
}


# Test seam for process-level ASRClient tests (tests/asr/test_client_process.py):
# the variable names a "module:factory" the worker imports AFTER the
# base env import, letting tests spawn a real worker with a fake
# engine (base CI has no real engine deps). Never set in production.
_TEST_FACTORY_ENV = "LIVETRANSLATE_TEST_ENGINE_FACTORY"


def _load_engine(config: dict):
    engine_type = config.get("engine_type")
    try:
        factory = _ENGINE_FACTORIES[engine_type]
    except KeyError:
        test_factory = os.environ.get(_TEST_FACTORY_ENV)
        if test_factory:
            mod_name, _, attr = test_factory.partition(":")
            factory = getattr(importlib.import_module(mod_name), attr)
        else:
            raise ConfigError(
                f"Unknown ASR engine type: {engine_type!r} "
                f"(expected one of {sorted(_ENGINE_FACTORIES)})"
            ) from None

    from livetranslate.modeling.manager import apply_cache_env

    apply_cache_env()

    engine = factory(config)
    # Base-class guarantee: every engine has set_language.
    engine.set_language(config.get("language", "auto"))
    return engine


def _transcribe(engine, payload: dict):
    audio = payload.get("audio")
    if not isinstance(audio, np.ndarray):
        raise TypeError("transcribe payload audio must be a numpy.ndarray")
    if audio.ndim != 1:
        raise TypeError(f"transcribe payload audio must be 1-D mono, got shape {audio.shape}")
    if audio.dtype != np.float32:
        raise TypeError(f"transcribe payload audio must be float32, got {audio.dtype}")
    if audio.size == 0:
        raise ValueError("transcribe payload audio is empty")

    kwargs = {}
    if engine.capabilities.word_timestamps:
        kwargs["word_timestamps"] = bool(payload.get("word_timestamps", False))
    return engine.transcribe(audio, **kwargs)


def _cleanup_engine(engine):
    if engine is not None:
        try:
            engine.unload()
        except Exception:
            log.warning("ASR engine unload failed", exc_info=True)


def handle_message(engine, msg: dict) -> dict:
    """Execute one worker command and return the response frame.

    Pure command dispatch (no pipe access) so every command path is
    unit-testable with a fake engine. Raises on unknown commands; the
    caller turns that into an error response.
    """
    msg_id = msg.get("id")
    msg_type = msg.get("type")
    payload = msg.get("payload") or {}

    if msg_type == "shutdown":
        return ok_response(msg_id, "shutdown")
    if msg_type == "ping":
        return ok_response(msg_id, "pong")
    if msg_type == "transcribe":
        return ok_response(msg_id, "result", _transcribe(engine, payload))
    if msg_type == "set_language":
        engine.set_language(payload.get("language", "auto"))
        return ok_response(msg_id, "ack")
    if msg_type == "set_input_padding":
        engine.set_input_padding(payload.get("pad_seconds"))
        return ok_response(msg_id, "ack")
    raise ValueError(f"Unknown ASR worker command: {msg_type}")


def _release_gpu_memory():
    """Free GPU cache after an inference-heavy command.

    Runs in the worker process after each transcribe. Best effort: CPU-only
    installs skip the CUDA path and failures are never fatal. (This block
    previously sat unreachably after `raise ValueError` in handle_message.)
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def worker_main(conn, config: dict):
    _setup_logging()
    engine = None
    try:
        log.info(
            "ASR worker loading: "
            f"{config.get('engine_type')} on {config.get('device')} "
            f"(pid config={config.get('display_name', '')})"
        )
        engine = _load_engine(config)
        conn.send(
            ok_response(
                None,
                "ready",
                {
                    "engine_type": config.get("engine_type"),
                    "display_name": config.get("display_name"),
                    "device": config.get("device"),
                },
            )
        )
    except BaseException as exc:
        log.error(f"ASR worker load failed: {exc}", exc_info=True)
        try:
            conn.send(error_response(None, exc, recoverable=False))
        finally:
            _cleanup_engine(engine)
            conn.close()
        return

    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break
            except Exception as exc:
                # OSError / unpickling errors must not kill the worker
                # silently: report once, then shut down cleanly.
                log.error(f"ASR worker pipe error: {exc}", exc_info=True)
                break

            if not isinstance(msg, dict):
                log.error(f"ASR worker received non-dict message: {type(msg).__name__}")
                try:
                    conn.send(
                        error_response(
                            None,
                            ValueError(f"invalid message format: {type(msg).__name__}"),
                            recoverable=True,
                        )
                    )
                except (BrokenPipeError, EOFError, OSError):
                    break
                continue

            msg_id = msg.get("id")
            msg_type = msg.get("type")

            try:
                response = handle_message(engine, msg)
            except Exception as exc:
                log.error(f"ASR worker command failed: {msg_type}: {exc}", exc_info=True)
                conn.send(error_response(msg_id, exc, recoverable=True))
                continue
            conn.send(response)
            if msg_type == "transcribe":
                _release_gpu_memory()
            if msg_type == "shutdown":
                break
    finally:
        _cleanup_engine(engine)
        with contextlib.suppress(Exception):
            conn.close()
        log.info("ASR worker stopped")
