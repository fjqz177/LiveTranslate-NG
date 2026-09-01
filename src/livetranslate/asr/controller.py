"""ASR worker lifecycle controller.

Owns the engine state machine that used to live inline in LiveTranslateApp:
deferred settings, engine switching bookkeeping, crash recovery, memory
recycling and idle liveness probing.

UI interactions are injected as callbacks (status label, memory logging,
running flag, memory release) so this module has no Qt dependencies and the
state machine is unit-testable with fake clients.
"""

import contextlib
import logging
import threading
import time

from livetranslate.asr.client import (
    ASRClient,
    ASRWorkerError,
    ASRWorkerExited,
    ASRWorkerTimeout,
)
from livetranslate.asr.remote import RemoteASRError

log = logging.getLogger("LiveTranslate.ASR")

_NO_PENDING = object()


class AsrController:
    """Serializes access to the single active ASR client and its lifecycle."""

    def __init__(
        self,
        initial_device: str,
        initial_whisper_model_size: str,
        initial_funasr_model_key: str,
        status_cb=None,  # callable(str): update the ASR status label
        is_running_cb=None,  # callable() -> bool
        release_memory_cb=None,  # callable()
        mem_cb=None,  # callable(kind, audio_seconds, asr_ms)
    ):
        self._lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._status_cb = status_cb or (lambda text: None)
        self._is_running = is_running_cb or (lambda: True)
        self._release_memory = release_memory_cb or (lambda: None)
        self._mem_cb = mem_cb or (lambda kind, secs, ms: None)

        self._client = None
        self._type = None
        self._signature = None
        self._config = None
        self._ready = False
        self._error_count = 0
        self._device = initial_device
        self._whisper_model_size = initial_whisper_model_size
        self._funasr_model_key = initial_funasr_model_key

        # Settings changed from the Qt thread are deferred here and applied by the
        # ASR thread before its next transcribe, so the UI never blocks on the
        # worker pipe (which may be busy with an in-flight cross-process call).
        # Padding is keyed by engine_type because one settings save updates both
        # the funasr and whisper padding and they must not clobber each other.
        self._pending_language = _NO_PENDING
        self._pending_padding = {}

        # Auto-restart bookkeeping for a worker that dies mid-session.
        # _generation is bumped on every (de)activation so a slow background
        # (re)start can detect that a newer engine switch superseded it and
        # discard its stale worker.
        self._restart_state = None
        self._restart_count = 0
        self._restart_max = 3
        self._generation = 0
        self._recycling = False
        # Idle liveness probe: detects a worker that died while no audio flows.
        self._last_ping = 0.0
        # Proactively recycle the worker once its RSS grows this far past the
        # post-load baseline, to bound native-side (FunASR/CTranslate2) leaks.
        self._worker_baseline_mb = None
        self._recycle_delta_mb = 2048

    # --- Read-only surface used by the app layer ---

    @property
    def client(self):
        with self._lock:
            return self._client

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    @property
    def type(self):
        with self._lock:
            return self._type

    @property
    def device(self):
        with self._lock:
            return self._device

    @property
    def funasr_model_key(self):
        with self._lock:
            return self._funasr_model_key

    @property
    def whisper_model_size(self):
        with self._lock:
            return self._whisper_model_size

    # --- Lifecycle primitives ---

    def mark_unavailable(self, reason: str, client=None):
        with self._lock:
            current = client or self._client
            if client is not None and self._client is not client:
                return
            self._ready = False
            self._client = None
            self._type = None
            self._signature = None
            self._config = None
            self._error_count = 0
            self._restart_state = None
            self._worker_baseline_mb = None
            self._generation += 1
        if current is not None:
            try:
                current.shutdown()
            except Exception:
                with contextlib.suppress(Exception):
                    current.terminate()
        log.warning(f"ASR worker unavailable: {reason}")
        self._status_cb("ASR unavailable")

    def shutdown_worker(self):
        with self._lock:
            client = self._client
            self._client = None
            self._ready = False
            self._type = None
            self._signature = None
            self._config = None
            self._error_count = 0
            self._restart_state = None
            self._worker_baseline_mb = None
            self._generation += 1
        if client is not None:
            log.info(f"Shutting down ASR worker: pid={client.pid}")
            client.shutdown()

    def set_language(self, language: str):
        with self._pending_lock:
            self._pending_language = language

    def set_padding(self, engine_type: str, pad_seconds):
        with self._pending_lock:
            self._pending_padding[engine_type] = pad_seconds

    def is_ready_with_signature(self, signature) -> bool:
        with self._lock:
            client = self._client
            return (
                self._ready
                and client is not None
                and client.status == "ready"
                and self._signature == signature
            )

    def refresh_ready(self):
        """Recompute ready from the live client status (e.g. after a cancelled
        download left the previous worker running)."""
        with self._lock:
            client = self._client
            self._ready = client is not None and client.status == "ready"

    def snapshot_state(self) -> dict | None:
        """Restart-state dict for the current client, or None when idle."""
        with self._lock:
            if self._type is None:
                return None
            config = dict(self._config) if self._config else None
            return {
                "type": self._type,
                "signature": self._signature,
                "device": self._device,
                "funasr_model_key": self._funasr_model_key,
                "whisper_model_size": self._whisper_model_size,
                "config": config,
                "display_name": (config or {}).get("display_name"),
                "device_label": (
                    (config or {}).get("remote_asr_url")
                    if self._type == "remote-whisper"
                    else self._device
                ),
            }

    def detach_current(self):
        """Detach the active client for a switch; bumps the generation so any
        in-flight (re)start discards itself. Returns (old_client, old_state)."""
        with self._lock:
            old_client = self._client
            old_state = self.snapshot_state()
            self._client = None
            self._ready = False
            self._type = None
            self._signature = None
            self._config = None
            self._error_count = 0
            self._restart_state = None
            self._worker_baseline_mb = None
            self._generation += 1
        return old_client, old_state

    def activate(self, client, state: dict):
        with self._lock:
            self._client = client
            self._type = state["type"]
            self._signature = state["signature"]
            self._device = state["device"]
            self._config = dict(state["config"]) if state["config"] else None
            self._funasr_model_key = state["funasr_model_key"]
            self._whisper_model_size = state["whisper_model_size"]
            self._ready = True
            self._error_count = 0
            self._restart_state = dict(state)
            self._restart_count = 0
            self._worker_baseline_mb = None
            self._generation += 1

    def load_engine_client(self, config: dict):
        """Build the ASR backend for a worker config. Local engines run in an isolated
        worker subprocess (ASRClient); remote-whisper is a thin in-process HTTP client
        that needs no subprocess isolation (no native deps, no GPU model to load)."""
        if config.get("engine_type") == "remote-whisper":
            from livetranslate.asr.remote import RemoteASREngine

            url = config.get("remote_asr_url") or "http://127.0.0.1:8765"
            # SEC-5: shared token (X-ASR-Token) when the server requires one.
            engine = RemoteASREngine(server_url=url, token=config.get("remote_asr_token") or "")
            language = config.get("language")
            if language:
                engine.set_language(language)
            return engine
        return self._load_asr_client(config)

    def _load_asr_client(self, worker_config: dict) -> ASRClient:
        # request_timeout bounds how long a hung worker can stall the realtime path
        # before it is killed and auto-restarted. VAD caps segments at a few seconds,
        # so 60s is generous for a healthy transcribe yet far below the old 120s.
        client = ASRClient(worker_config, request_timeout=60.0)
        try:
            client.start()
            client.wait_ready()
            return client
        except Exception:
            client.shutdown()
            raise

    # --- Deferred settings application (ASR thread) ---

    def apply_pending_settings(self, client, asr_type):
        """Apply deferred language/padding just before a transcribe.
        A pending value is cleared only once delivered; worker-death exceptions
        propagate with the pending intact so the restarted worker re-applies it. The
        applied value is written back into the restart config so an auto-restart or
        recycle does not revert a runtime override to the engine-switch-time value."""
        with self._pending_lock:
            language = self._pending_language
            pad_seconds = self._pending_padding.get(asr_type, _NO_PENDING)
        if language is not _NO_PENDING:
            try:
                client.set_language(language)
            except ASRWorkerError as exc:
                log.warning(f"ASR language update failed: {exc}")
            self._update_restart_config(language=language)
            self._clear_pending_language(language)
        if pad_seconds is not _NO_PENDING:
            # Unsupporting engines no-op inside the worker (base-class default),
            # so no client-side capability judgment is needed here.
            try:
                client.set_input_padding(pad_seconds)
            except ASRWorkerError as exc:
                log.warning(f"ASR padding update failed: {exc}")
            self._update_restart_config(pad_seconds=pad_seconds)
            self._clear_pending_padding(asr_type, pad_seconds)

    def _clear_pending_language(self, language):
        with self._pending_lock:
            if self._pending_language is language:
                self._pending_language = _NO_PENDING

    def _clear_pending_padding(self, asr_type, pad_seconds):
        with self._pending_lock:
            if self._pending_padding.get(asr_type) == pad_seconds:
                del self._pending_padding[asr_type]

    def _update_restart_config(self, **kwargs):
        with self._lock:
            if self._restart_state and self._restart_state.get("config"):
                self._restart_state["config"].update(kwargs)

    # --- Transcribe path (ASR thread) ---

    def transcribe(self, audio, kind: str, **kwargs):
        audio_seconds = len(audio) / 16000
        asr_start = time.perf_counter()
        # Snapshot the active client under the lock, then release it: the blocking
        # cross-process transcribe must not hold the lock, or a slow/hung worker
        # would freeze the Qt thread on every settings change. ASRClient serializes
        # its own pipe access, and only this (single) ASR thread calls transcribe.
        with self._lock:
            if not self._ready or self._client is None:
                return None, 0.0
            client = self._client
            asr_type = self._type
        try:
            self.apply_pending_settings(client, asr_type)
            result = client.transcribe(audio, **kwargs)
        except (ASRWorkerExited, ASRWorkerTimeout, RemoteASRError) as exc:
            # Worker/remote-server death: recover via the restart machinery.
            asr_ms = (time.perf_counter() - asr_start) * 1000
            self._mem_cb(f"{kind}:error", audio_seconds, asr_ms)
            self.recover_worker(client, str(exc))
            raise
        except ASRWorkerError as exc:
            asr_ms = (time.perf_counter() - asr_start) * 1000
            self._mem_cb(f"{kind}:error", audio_seconds, asr_ms)
            fatal = False
            with self._lock:
                if self._client is client:
                    self._error_count += 1
                    fatal = not exc.recoverable or self._error_count >= 3
            if fatal:
                self.mark_unavailable(str(exc), client)
            raise
        except Exception:
            asr_ms = (time.perf_counter() - asr_start) * 1000
            self._mem_cb(f"{kind}:error", audio_seconds, asr_ms)
            raise
        with self._lock:
            if self._client is client:
                self._error_count = 0
                self._restart_count = 0
        asr_ms = (time.perf_counter() - asr_start) * 1000
        self._mem_cb(kind, audio_seconds, asr_ms)
        return result, asr_ms

    # --- Crash recovery / recycling / liveness (ASR thread) ---

    def start_worker_from_state(self, state: dict, expected_gen: int) -> bool:
        """Load a worker from a saved state and activate it only if no newer engine
        switch happened in the meantime (generation guard). Runs on the ASR thread;
        the load is intentionally done outside the lock. Returns True on activation."""
        try:
            client = self.load_engine_client(state["config"])
        except Exception as e:
            log.error(f"ASR worker (re)start failed: {e}", exc_info=True)
            return False
        stale = None
        with self._lock:
            if self._generation != expected_gen or not self._is_running():
                stale = client
            else:
                self.activate(client, state)
        if stale is not None:
            log.info("Discarding superseded ASR worker (newer switch won the race)")
            with contextlib.suppress(Exception):
                stale.shutdown()
            return False
        name = state.get("display_name") or state.get("type")
        self._status_cb(f"{name} [{state.get('device_label', state['device'])}]")
        return True

    def recover_worker(self, dead_client, reason: str):
        """Auto-restart a worker that died mid-session. Without this, a single crash
        or transcribe timeout would leave ASR permanently silent for the session."""
        with self._lock:
            if self._client is not dead_client:
                return  # an engine switch already replaced/cleared it
            state = dict(self._restart_state) if self._restart_state else None
            attempt = self._restart_count + 1
            give_up = state is None or not state.get("config") or attempt > self._restart_max
            self._restart_count = attempt
            self._client = None
            self._ready = False
            self._type = None
            self._signature = None
            self._config = None
            self._error_count = 0
            self._worker_baseline_mb = None
            self._generation += 1
            gen = self._generation
        try:
            dead_client.shutdown()
        except Exception:
            with contextlib.suppress(Exception):
                dead_client.terminate()
        if not self._is_running():
            return  # shutting down; do not spawn a replacement worker
        if give_up:
            log.error(
                f"ASR worker died and auto-restart gave up after "
                f"{self._restart_max} attempts: {reason}"
            )
            self._status_cb("ASR unavailable")
            return
        log.warning(
            f"ASR worker died ({reason}); auto-restart attempt {attempt}/{self._restart_max}"
        )
        self._release_memory()
        if self.start_worker_from_state(state, gen):
            log.info(f"ASR worker auto-restarted: {state.get('type')} on {state.get('device')}")
        elif self.client is None:
            self._status_cb("ASR unavailable")

    def maybe_recycle_worker(self):
        """Recycle the worker once its RSS grows well past the post-load baseline, to
        bound native-side leaks that accumulate in the long-lived worker process.
        Called from the ASR thread between segments so the reload gap costs no audio
        beyond what arrives during it."""
        if not self._is_running():
            return
        with self._lock:
            client = self._client
            if not self._ready or client is None or self._recycling:
                return
            state = dict(self._restart_state) if self._restart_state else None
        if state is None or not state.get("config") or client.pid is None:
            return
        try:
            import psutil

            rss = psutil.Process(client.pid).memory_info().rss / 1024 / 1024
        except Exception:
            return
        if self._worker_baseline_mb is None:
            self._worker_baseline_mb = rss
            return
        if rss < self._worker_baseline_mb + self._recycle_delta_mb:
            return
        log.warning(
            f"ASR worker RSS={rss:.0f}MB grew "
            f"{rss - self._worker_baseline_mb:.0f}MB over baseline; recycling"
        )
        self.recycle_worker(client, state)

    def recycle_worker(self, old_client, state: dict):
        # Graceful stop-then-start (no VRAM doubling). The generation guard makes a
        # concurrent engine switch win over this recycle.
        with self._lock:
            if self._client is not old_client:
                return
            self._client = None
            self._ready = False
            self._recycling = True
            self._worker_baseline_mb = None
            self._generation += 1
            gen = self._generation
        try:
            old_client.shutdown()
        except Exception:
            with contextlib.suppress(Exception):
                old_client.terminate()
        self._release_memory()
        if not self._is_running():
            with self._lock:
                self._recycling = False
            return
        try:
            started = self.start_worker_from_state(state, gen)
        finally:
            with self._lock:
                self._recycling = False
        if started:
            log.info(f"ASR worker recycled: {state.get('type')} on {state.get('device')}")
        else:
            log.error("ASR worker recycle failed to restart")
            if self.client is None:
                self._status_cb("ASR unavailable")

    def maybe_ping_worker(self):
        """Detect a worker that died while idle instead of waiting for the next
        60s transcribe timeout to expose it. Runs on the ASR thread."""
        now = time.monotonic()
        if now - self._last_ping < 5.0:
            return
        self._last_ping = now
        with self._lock:
            client = self._client
            ready = self._ready
        if not ready or client is None or client.pid is None:
            return  # in-process engine (remote) has no worker process to probe
        try:
            client.ping()
        except (ASRWorkerExited, ASRWorkerTimeout) as exc:
            log.warning(f"ASR worker ping failed: {exc}")
            self.recover_worker(client, str(exc))
        except ASRWorkerError:
            pass  # recoverable engine error; the transcribe path counts these
