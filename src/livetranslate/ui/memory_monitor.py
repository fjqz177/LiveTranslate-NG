"""Memory diagnostics: RSS/GPU snapshot, threshold warn-once, periodic timer.

Owns the process-RSS baseline tracking and the warn-once/threshold/timer
formatting. The collaborator reads (ASR worker pid, VAD buffer, overlay
message count, pipeline counters, GPU alloc) are injected as a closure so the
composition root keeps those collaborators and reads them lazily — the
AsrController is built before the Pipeline, so the reads must be deferred to
snapshot time, never at construction.

torch is optional (engine groups own it): a base (no-engine) install must not
crash at import, so the import is guarded exactly like the composition root.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import torch
except ImportError:  # base install without engine extras
    torch = None

from PyQt6.QtCore import QTimer

log = logging.getLogger("LiveTranslate.Memory")


class MemoryMonitor:
    """Track main-process RSS + worker/GPU/overlay state, warn once on a
    combined ceiling, and run a periodic log tick while the pipeline runs."""

    def __init__(
        self,
        collaborator_snapshot: Callable[[], dict[str, Any]],
        threshold_mb: int = 4096,
    ) -> None:
        import psutil

        self._proc = psutil.Process(os.getpid())
        self._baseline_mb = self._proc.memory_info().rss / 1024 / 1024
        self._last_mb = self._baseline_mb
        self._asr_call_count = 0
        self._threshold_mb = threshold_mb
        self._warned = False
        self._last_memory_warning: float | None = None
        self._warning_callback: Callable[[float], None] | None = None
        self._periodic_timer: QTimer | None = None
        self._collab = collaborator_snapshot

    def snapshot(self) -> dict[str, Any]:
        """Combine own process RSS with the injected collaborator snapshot.

        The collaborator closure (in the composition root) supplies the ASR
        worker RSS, GPU alloc/reserved, overlay message count, VAD buffer and
        the pipeline counters; this method adds the main-process RSS and the
        combined total used for the threshold check."""
        rss_mb = self._proc.memory_info().rss / 1024 / 1024
        state = self._collab()
        worker_rss = float(state.get("worker_rss", 0.0))
        return {
            "rss": rss_mb,
            "worker_rss": worker_rss,
            "total_rss": rss_mb + worker_rss,
            "gpu_alloc": state.get("gpu_alloc", 0.0),
            "gpu_reserved": state.get("gpu_reserved", 0.0),
            "msgs": state.get("msgs", 0),
            "vad_buf": state.get("vad_buf", 0),
            "asr_count": state.get("asr_count", 0),
            "translate_count": state.get("translate_count", 0),
        }

    def log_after_asr(self, kind: str, audio_seconds: float, asr_ms: float) -> None:
        self._asr_call_count += 1
        snap = self.snapshot()
        delta = snap["rss"] - self._last_mb
        total_delta = snap["rss"] - self._baseline_mb
        self._last_mb = snap["rss"]
        log.info(
            f"MEM[asr#{self._asr_call_count}:{kind}] RSS={snap['rss']:.1f}MB "
            f"(Δ{delta:+.2f} since last, {total_delta:+.1f} since start) "
            f"worker_rss={snap['worker_rss']:.0f}MB "
            f"GPU(main alloc/reserved)={snap['gpu_alloc']:.0f}/{snap['gpu_reserved']:.0f}MB "
            f"audio={audio_seconds:.1f}s asr={asr_ms:.0f}ms "
            f"outputs={snap['asr_count']} msgs={snap['msgs']} vad_buf={snap['vad_buf']}"
        )
        self.check_threshold(snap["total_rss"])

    def log_periodic(self) -> None:
        snap = self.snapshot()
        total_delta = snap["rss"] - self._baseline_mb
        log.info(
            f"MEM[tick] RSS={snap['rss']:.1f}MB ({total_delta:+.1f} since start) "
            f"worker_rss={snap['worker_rss']:.0f}MB "
            f"GPU(main alloc/reserved)={snap['gpu_alloc']:.0f}/{snap['gpu_reserved']:.0f}MB "
            f"msgs={snap['msgs']} asr_calls={self._asr_call_count} "
            f"asr_count={snap['asr_count']} tl_count={snap['translate_count']}"
        )
        self.check_threshold(snap["total_rss"])

    def release_caches(self) -> None:
        gc.collect()
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch is not None and hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def check_threshold(self, rss_mb: float) -> None:
        if self._warned or rss_mb < self._threshold_mb:
            return
        self._warned = True
        self._last_memory_warning = rss_mb
        log.warning(
            f"Memory ceiling reached: combined RSS (main+worker)={rss_mb:.0f}MB "
            f"(threshold {self._threshold_mb}MB). "
            f"Recommend restarting LiveTranslate to free C-side allocator caches."
        )
        if self._warning_callback is not None:
            try:
                self._warning_callback(rss_mb)
            except Exception as e:
                log.warning(f"Memory warning callback failed: {e}")

    def set_warning_callback(self, callback: Callable[[float], None]) -> None:
        self._warning_callback = callback

    def on_pipeline_started(self) -> None:
        # Periodic memory snapshot every 30s
        if self._periodic_timer is None:
            self._periodic_timer = QTimer()
            self._periodic_timer.timeout.connect(self.log_periodic)
            self._periodic_timer.start(30000)
        snap = self.snapshot()
        log.info(
            f"MEM[start] RSS={snap['rss']:.1f}MB "
            f"GPU(alloc/reserved)={snap['gpu_alloc']:.0f}/{snap['gpu_reserved']:.0f}MB "
            f"(baseline for delta tracking)"
        )

    def on_pipeline_stopping(self) -> None:
        if self._periodic_timer is not None:
            with contextlib.suppress(Exception):
                self._periodic_timer.stop()
            self._periodic_timer = None
        snap = self.snapshot()
        total_delta = snap["rss"] - self._baseline_mb
        log.info(
            f"MEM[stop] RSS={snap['rss']:.1f}MB ({total_delta:+.1f} since start) "
            f"GPU(alloc/reserved)={snap['gpu_alloc']:.0f}/{snap['gpu_reserved']:.0f}MB "
            f"asr_calls={self._asr_call_count} outputs={snap['asr_count']}"
        )
