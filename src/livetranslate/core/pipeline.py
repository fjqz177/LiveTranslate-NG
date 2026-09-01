"""Realtime pipeline: capture -> VAD -> ASR -> translation -> outputs.

Owns the two worker threads (capture + ASR), the incremental-ASR state
machine, translation dispatch and the message/stat counters. UI and
persistence side effects go to the injected overlay/subtitle-window/panel/
transcript objects, so the pipeline itself has no Qt dependencies beyond
those references.

Lifecycle hooks (start_hook / stop_hook) let the app layer keep its
memory-diagnostics bookkeeping in the exact same order as before.
"""

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from livetranslate.core.errors import classify_translate_error
from livetranslate.core.i18n import t
from livetranslate.core.privacy import redact_text
from livetranslate.core.segment_text import (
    is_short_utterance,
    split_sentences,
    strip_committed_overlap,
)
from livetranslate.core.transcript_writer import TranscriptWriter
from livetranslate.core.translator import RepetitionError, Translator

log = logging.getLogger("LiveTranslate.Pipeline")

FloatArray = NDArray[np.float32]


# ── Injected collaborators ────────────────────────────────────────────────
# core cannot import the audio/asr/ui modules (layering rule), so these
# Protocols declare the minimal interfaces the pipeline actually uses.


class _VAD(Protocol):
    def is_speaking(self) -> bool: ...
    def effective_silence_limit(self) -> int: ...
    def process_chunk(self, audio_chunk: FloatArray) -> FloatArray | None: ...
    def current_confidence(self) -> float: ...
    def buffered_samples(self) -> int: ...
    def force_flush(self) -> FloatArray | None: ...
    def flush(self) -> FloatArray | None: ...
    def peek_buffer(self) -> tuple[FloatArray, float] | None: ...
    def trim_front(self, n_samples: int) -> None: ...
    def reset(self) -> None: ...


class _AudioCapture(Protocol):
    """Minimal capture contract (the concrete AudioBackend owns start/stop
    lifecycle; the pipeline only consumes chunks)."""

    def read_chunk(self) -> tuple[FloatArray, float | None] | None: ...


class _TranscriptionResult(Protocol):
    text: str
    language: str


class _AsrController(Protocol):
    @property
    def ready(self) -> bool: ...

    def transcribe(
        self, audio: FloatArray, kind: str
    ) -> tuple[_TranscriptionResult | None, float]: ...

    def maybe_recycle_worker(self) -> None: ...
    def maybe_ping_worker(self) -> None: ...
    def shutdown_worker(self) -> None: ...


class _Overlay(Protocol):
    def add_message(
        self, msg_id: int, timestamp: str, text: str, lang: str, asr_ms: float
    ) -> None: ...

    def update_translation(self, msg_id: int, text: str | None, ms: float) -> None: ...
    def update_streaming(self, msg_id: int, partial: str) -> None: ...

    def update_stats(
        self,
        asr_count: int,
        tl_count: int,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
    ) -> None: ...

    def update_monitor(
        self, rms: float, confidence: float, mic_rms: float | None = None
    ) -> None: ...


class _SubtitleWindow(Protocol):
    def get_target_languages(self) -> set[str]: ...
    def isVisible(self) -> bool: ...
    def update_text(self, text: str, translations: dict[str, str]) -> None: ...


class _Panel(Protocol):
    def get_settings(self) -> dict[str, Any]: ...


class Pipeline:
    """Capture/VAD/ASR/translation orchestration."""

    def __init__(
        self,
        config: dict[str, Any],
        vad: _VAD,
        audio: _AudioCapture,
        asr_ctl: _AsrController,
        translator: Translator,
        transcript: TranscriptWriter,
        start_hook: Callable[[], None] | None = None,
        stop_hook: Callable[[], None] | None = None,
        log_transcript: bool = False,
    ) -> None:
        self._config = config
        self._vad = vad
        self._audio = audio
        self._asr_ctl = asr_ctl
        self._translator = translator
        self._transcript = transcript
        self._start_hook: Callable[[], None] = start_hook or (lambda: None)
        self._stop_hook: Callable[[], None] = stop_hook or (lambda: None)
        # SEC-1: speech/translation content never reaches the logs unless the
        # user explicitly opts in (the file handler is DEBUG-level, so this
        # must be gated at the call site, not by log level).
        self._log_transcript: bool = bool(log_transcript)

        self._overlay: _Overlay | None = None
        self._subwin: _SubtitleWindow | None = None
        self._panel: _Panel | None = None
        self._error_reporter: Callable[[str], None] | None = None

        self._running: bool = False
        self._paused: bool = False
        self._capture_thread: threading.Thread | None = None
        self._asr_thread: threading.Thread | None = None
        self._asr_queue: queue.Queue[tuple[str, FloatArray | None] | None] = queue.Queue(maxsize=16)
        self._tl_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=8)

        self._target_language: str = config["translation"]["target_language"]
        # CORE-11: snapshot of the expected ASR language; the ASR thread
        # never touches the Qt panel (app pushes changes via set_asr_language).
        self._asr_language: str = "auto"

        self._asr_count: int = 0
        self._translate_count: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._input_price: float = 0.0
        self._output_price: float = 0.0
        self._msg_id: int = 0
        self._last_original: str = ""
        self._last_msg_id: int = 0
        # CORE-7: the 8-worker translation pool mutates the counters
        # concurrently (read-modify-write must be atomic).
        self._stats_lock = threading.Lock()

        # Incremental ASR state
        self._incremental_enabled: bool = False
        self._interim_interval: float = 2.0
        self._interim_pending: str = ""
        self._interim_active: bool = False
        self._last_interim_samples: int = 0
        self._last_interim_check_time: float = 0.0
        self._interim_committed_tail: str = ""

    # --- Accessors for the app layer ---

    @property
    def running(self) -> bool:
        return self._running

    @property
    def vad(self) -> _VAD:
        return self._vad

    @property
    def audio(self) -> _AudioCapture:
        return self._audio

    @property
    def translator(self) -> Translator:
        return self._translator

    @property
    def transcript(self) -> TranscriptWriter:
        return self._transcript

    @property
    def asr_count(self) -> int:
        return self._asr_count

    @property
    def translate_count(self) -> int:
        return self._translate_count

    @property
    def target_language(self) -> str:
        return self._target_language

    # --- Wiring ---

    def set_overlay(self, overlay: _Overlay) -> None:
        self._overlay = overlay

    def set_error_reporter(self, callback: Callable[[str], None]) -> None:
        """Receive classified translation errors (overlay banner +
        diagnostics recent-errors list)."""
        self._error_reporter = callback

    def set_subtitle_window(self, subwin: _SubtitleWindow) -> None:
        self._subwin = subwin

    def set_panel(self, panel: _Panel) -> None:
        self._panel = panel

    def set_translator(self, translator: Translator) -> None:
        self._translator = translator

    def set_prices(self, input_price: float, output_price: float) -> None:
        self._input_price = input_price
        self._output_price = output_price

    def set_target_language(self, lang: str) -> None:
        self._target_language = lang
        if self._translator:
            self._translator.set_target_language(lang)

    def set_asr_language(self, lang: str) -> None:
        """Expected ASR language for the language filter (CORE-11).

        The value is a snapshot owned by the pipeline — the ASR thread must
        never read the Qt panel's settings dict; the app layer pushes
        changes here via the settings bridge."""
        self._asr_language = lang or "auto"

    def set_incremental(self, enabled: bool) -> None:
        self._incremental_enabled = bool(enabled)

    def set_log_transcript(self, enabled: bool) -> None:
        """Opt-in switch (settings key log_transcript): write full ASR and
        translation text into the logs. Off by default — the privacy policy
        promises transcript content stays out of logs and diagnostics."""
        self._log_transcript = bool(enabled)

    def set_interim_interval(self, seconds: float) -> None:
        self._interim_interval = seconds

    def reset_interim(self) -> None:
        """Reset the incremental-ASR interim state and the VAD buffer.

        The real interim state lives here, not on the app layer. The app used
        to assign dead ``self._interim_*`` attributes that no code ever reads;
        engine switching must now reset this state in place."""
        self._interim_active = False
        self._interim_pending = ""
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""
        if self._vad is not None:
            self._vad.reset()
        log.info("Interim ASR state reset")

    # --- Lifecycle ---

    def start(self) -> None:
        if self._running:
            return
        n = len(self._subwin.get_target_languages()) if self._subwin else 1
        self._tl_executor = ThreadPoolExecutor(max_workers=max(8, n + 1))
        self._asr_queue = queue.Queue(maxsize=16)
        self._running = True
        self._paused = False
        # The concrete audio backend is started by the app layer (it owns
        # device selection); the pipeline only consumes chunks.
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._asr_thread = threading.Thread(target=self._asr_loop, daemon=True)
        self._capture_thread.start()
        self._asr_thread.start()
        self._start_hook()
        log.info("Pipeline started (capture + ASR threads)")

    def stop(self) -> None:
        self._running = False
        # Backend stop is owned by the app layer, right after this join.
        if self._capture_thread:
            self._capture_thread.join(timeout=3)
            self._capture_thread = None
        # CORE-2: never block on the sentinel — evict one slot when the
        # queue is full (or the ASR thread died); the ASR loop exits on
        # `_running=False` anyway, so leftovers are simply discarded. The
        # sentinel only accelerates the shutdown.
        if self._asr_queue.full():
            with contextlib.suppress(queue.Empty):
                self._asr_queue.get_nowait()
        try:
            self._asr_queue.put_nowait(None)
        except queue.Full:
            log.warning("ASR queue refused sentinel; proceeding without it")
        if self._asr_thread:
            self._asr_thread.join(timeout=10)
            if self._asr_thread.is_alive():
                log.warning("ASR thread still running after timeout, proceeding with cleanup")
            self._asr_thread = None

        # CORE-10: flush the remaining VAD buffer on a bounded helper thread.
        # The old code transcribed on the calling (UI) thread — a hung worker
        # froze the app for the full 60s transcribe timeout. Best effort, 3s.
        def _flush_tail() -> None:
            try:
                if self._interim_active:
                    remaining = self._vad.force_flush()
                    if remaining is not None and self._asr_ctl.ready:
                        self._process_interim_final(remaining)
                else:
                    remaining = self._vad.flush()
                    if remaining is not None and self._asr_ctl.ready:
                        self._process_segment(remaining)
            except Exception:
                log.exception("Failed to flush VAD tail on stop")

        flush_thread = threading.Thread(target=_flush_tail, daemon=True)
        flush_thread.start()
        flush_thread.join(timeout=3)
        if flush_thread.is_alive():
            log.warning("VAD tail flush timed out; discarding remaining audio")
        self._interim_active = False
        self._interim_pending = ""
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""
        self._tl_executor.shutdown(wait=True)
        self._transcript.close()
        self._stop_hook()
        self._asr_ctl.shutdown_worker()
        log.info("Pipeline stopped")

    def pause(self) -> None:
        self._paused = True
        self._interim_active = False
        self._interim_pending = ""
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""
        if self._overlay:
            self._overlay.update_monitor(0.0, 0.0)
        log.info("Pipeline paused")

    def resume(self) -> None:
        self._paused = False
        log.info("Pipeline resumed")

    # --- Segment processing (ASR thread) ---

    def _process_segment(self, speech_segment: FloatArray) -> None:
        """Run ASR + translation on a speech segment. Called from ASR thread and stop()."""
        seg_len = len(speech_segment) / 16000
        log.info(f"Speech segment: {seg_len:.1f}s")

        try:
            result, asr_ms = self._asr_ctl.transcribe(speech_segment, "segment")
        except Exception as e:
            log.error(f"ASR error: {redact_text(str(e))}", exc_info=True)
            return
        if asr_ms == 0:
            return
        if asr_ms > 10000:
            log.warning(f"ASR took {asr_ms:.0f}ms, possible hang")
        if result is None:
            return

        original_text = result.text.strip()
        # Skip empty or punctuation-only ASR results
        if not original_text or not any(c.isalnum() for c in original_text):
            log.debug("ASR returned empty/punctuation-only, skipping")
            return

        # Skip suspiciously short text from long segments (likely noise)
        alnum_chars = sum(1 for c in original_text if c.isalnum())
        if seg_len >= 2.0 and alnum_chars <= 3:
            log.debug(
                f"Noise filter: {seg_len:.1f}s segment produced only "
                f"{alnum_chars} alnum chars, skipping"
            )
            return

        source_lang = result.language
        asr_lang_setting = self._asr_language
        if asr_lang_setting != "auto" and source_lang != asr_lang_setting:
            log.info(
                f"Language filter: expected '{asr_lang_setting}' but got '{source_lang}', "
                f"discarding {len(original_text)} chars"
            )
            if self._log_transcript:
                log.info(f"Discarded content: {original_text[:60]}")
            return

        self._asr_count += 1
        self._msg_id += 1
        msg_id = self._msg_id
        timestamp = datetime.now().strftime("%H:%M:%S")
        log.info(f"ASR [{source_lang}] ({asr_ms:.0f}ms): {len(original_text)} chars")
        if self._log_transcript:
            log.info(f"ASR content: {original_text}")

        if self._overlay:
            self._overlay.add_message(msg_id, timestamp, original_text, source_lang, asr_ms)
        self._transcript.write_original(msg_id, timestamp, original_text)

        # Store for subtitle window (translation will be added later)
        self._last_original = original_text
        self._last_msg_id = msg_id

        target_lang = self._target_language

        # Collect extra languages needed by subtitle window (beyond the primary target)
        extra_langs: set[str] = set()
        if self._subwin and self._subwin.isVisible():
            subwin_langs = self._subwin.get_target_languages()
            # Remove primary target and source (no need to translate those)
            extra_langs = subwin_langs - {target_lang, source_lang}

        if source_lang == target_lang:
            log.info(f"Same language ({source_lang}), no translation")
            self._transcript.finalize_no_translation(msg_id)
            if self._overlay:
                self._overlay.update_translation(msg_id, "", 0)
                self._overlay.update_stats(
                    self._asr_count,
                    self._translate_count,
                    self._total_prompt_tokens,
                    self._total_completion_tokens,
                    self._compute_cost(),
                )
            if self._subwin and self._subwin.isVisible():
                # Primary is same language; still need to translate extra langs
                if extra_langs:
                    with contextlib.suppress(RuntimeError):
                        self._tl_executor.submit(
                            self._translate_subwin_only, original_text, source_lang, extra_langs
                        )
                else:
                    self._subwin.update_text(original_text, {target_lang: original_text})
        else:
            try:
                self._tl_executor.submit(
                    self._translate_async,
                    msg_id,
                    original_text,
                    source_lang,
                    extra_langs or None,
                )
            except RuntimeError:
                log.warning("Translation executor shut down, skipping")

    def _do_interim_asr(self) -> bool:
        """Run ASR on current VAD buffer, output complete sentences, trim consumed audio.
        Returns True if any sentences were committed."""
        peek = self._vad.peek_buffer()
        if peek is None:
            return False
        audio, duration = peek

        # Don't bother with very short buffers
        if duration < 1.5:
            return False

        try:
            result, asr_ms = self._asr_ctl.transcribe(audio, "interim")
        except Exception as e:
            log.error(f"Interim ASR error: {redact_text(str(e))}", exc_info=True)
            return False

        if asr_ms == 0:
            return False

        if result is None:
            return False

        full_text = result.text.strip()
        if not full_text or not any(c.isalnum() for c in full_text):
            return False

        # Strip echo from previous commit's overlap
        full_text = strip_committed_overlap(full_text, self._interim_committed_tail)
        if not full_text:
            return False

        split_start = time.perf_counter()
        sentences = split_sentences(full_text, result.language)
        split_ms = (time.perf_counter() - split_start) * 1000
        if len(sentences) <= 1:
            return False
        log.debug(
            f"Interim split [{result.language}] ({split_ms:.1f}ms): {len(sentences)} parts"
        )

        # All but last are complete; last is still being spoken
        complete = sentences[:-1]

        committed_text = ""
        for sent in complete:
            committed_text += sent

        if not committed_text.strip():
            return False

        # Determine trim point: proportional trim with safety margin to
        # reduce echo (word-timestamp alignment was never enabled — it is
        # expensive for repeated interim passes).
        total_samples = len(audio)
        ratio = len(committed_text) / max(len(full_text), 1)
        margin = int(0.3 * 16000)  # 0.3s extra trim to avoid re-recognition
        trim_samples = int(ratio * total_samples) + margin
        # Don't over-trim: keep at least 0.5s for the remaining sentence
        max_trim = total_samples - int(0.5 * 16000)
        trim_samples = min(trim_samples, max(max_trim, 0))
        # Minimum trim to prevent re-recognition loops
        min_trim = int(0.3 * 16000)
        if trim_samples < min_trim and trim_samples > 0:
            trim_samples = min(min_trim, total_samples // 2)

        # Output committed sentences
        actually_committed = False
        for sent in complete:
            text = sent.strip()
            if not text:
                continue
            if is_short_utterance(text):
                self._interim_pending += text
                log.debug(
                    f"Interim short utterance buffered: {len(text)} chars, pending={len(self._interim_pending)} chars"
                )
                continue

            if self._interim_pending:
                text = self._interim_pending + text
                self._interim_pending = ""

            self._process_segment_text(text, result.language, asr_ms)
            actually_committed = True

        if not actually_committed:
            return False

        if trim_samples > 0:
            self._vad.trim_front(trim_samples)

        # Track committed text tail for echo dedup
        self._interim_committed_tail = (
            committed_text[-50:] if len(committed_text) > 50 else committed_text
        )

        self._interim_active = True
        log.info(
            f"Interim ASR: committed {len(complete)} sentence(s), trimmed {trim_samples / 16000:.2f}s"
        )
        return True

    def _process_segment_text(self, text: str, source_lang: str, asr_ms: float = 0) -> None:
        """Output a text result (from interim or final) — similar to _process_segment but skips ASR."""
        original_text = text.strip()
        if not original_text or not any(c.isalnum() for c in original_text):
            return

        asr_lang_setting = self._asr_language
        if asr_lang_setting != "auto" and source_lang != asr_lang_setting:
            log.info(
                f"Language filter: expected '{asr_lang_setting}' but got '{source_lang}', discarding {len(original_text)} chars"
            )
            return

        self._asr_count += 1
        self._msg_id += 1
        msg_id = self._msg_id
        timestamp = datetime.now().strftime("%H:%M:%S")
        log.info(f"ASR [{source_lang}] ({asr_ms:.0f}ms, interim): {len(original_text)} chars")
        if self._log_transcript:
            log.info(f"ASR interim content: {original_text}")

        if self._overlay:
            self._overlay.add_message(msg_id, timestamp, original_text, source_lang, asr_ms)
        self._transcript.write_original(msg_id, timestamp, original_text)

        self._last_original = original_text
        self._last_msg_id = msg_id

        target_lang = self._target_language
        extra_langs: set[str] = set()
        if self._subwin and self._subwin.isVisible():
            subwin_langs = self._subwin.get_target_languages()
            extra_langs = subwin_langs - {target_lang, source_lang}

        if source_lang == target_lang:
            log.info(f"Same language ({source_lang}), no translation")
            self._transcript.finalize_no_translation(msg_id)
            if self._overlay:
                self._overlay.update_translation(msg_id, "", 0)
                self._overlay.update_stats(
                    self._asr_count,
                    self._translate_count,
                    self._total_prompt_tokens,
                    self._total_completion_tokens,
                    self._compute_cost(),
                )
            if self._subwin and self._subwin.isVisible():
                if extra_langs:
                    with contextlib.suppress(RuntimeError):
                        self._tl_executor.submit(
                            self._translate_subwin_only, original_text, source_lang, extra_langs
                        )
                else:
                    self._subwin.update_text(original_text, {target_lang: original_text})
        else:
            try:
                self._tl_executor.submit(
                    self._translate_async, msg_id, original_text, source_lang, extra_langs or None
                )
            except RuntimeError:
                log.warning("Translation executor shut down, skipping")

    def _process_interim_final(self, speech_segment: FloatArray) -> None:
        """Handle VAD flush after interim outputs were already made."""
        seg_len = len(speech_segment) / 16000
        log.info(f"Interim final segment: {seg_len:.1f}s")

        try:
            result, asr_ms = self._asr_ctl.transcribe(speech_segment, "interim_final")
        except Exception as e:
            log.error(f"Interim final ASR error: {redact_text(str(e))}", exc_info=True)
            return
        if asr_ms == 0:
            return

        if result is None:
            # Flush any remaining pending
            if self._interim_pending:
                text = self._interim_pending
                self._interim_pending = ""
                lang = self._asr_language
                if lang == "auto":
                    lang = "unknown"
                self._process_segment_text(text, lang)
            return

        original_text = result.text.strip()

        # Strip echo from previous commit's overlap
        original_text = strip_committed_overlap(original_text, self._interim_committed_tail)

        # Prepend any remaining pending short utterances
        if self._interim_pending:
            original_text = self._interim_pending + original_text
            self._interim_pending = ""

        if not original_text or not any(c.isalnum() for c in original_text):
            return

        # Apply noise filter like _process_segment
        alnum_chars = sum(1 for c in original_text if c.isalnum())
        if seg_len >= 2.0 and alnum_chars <= 3:
            log.debug(
                f"Noise filter: {seg_len:.1f}s segment produced only {alnum_chars} alnum chars, skipping"
            )
            return

        self._process_segment_text(original_text, result.language, asr_ms)

    # --- Worker threads ---

    def _capture_loop(self) -> None:
        silence_chunk = np.zeros(
            int(self._config["audio"]["sample_rate"] * self._config["audio"]["chunk_duration"]),
            dtype=np.float32,
        )
        while self._running:
            item = self._audio.read_chunk()
            if item is None:
                if self._vad.is_speaking() and not self._paused:
                    n = self._vad.effective_silence_limit() + 1
                    for _ in range(n):
                        seg = self._vad.process_chunk(silence_chunk)
                        if seg is not None and self._asr_ctl.ready:
                            self._enqueue_asr("vad_flush", seg)
                            break
                continue

            chunk, mic_rms = item

            if self._paused:
                continue

            rms = float(np.sqrt(np.mean(chunk**2)))

            if self._overlay:
                self._overlay.update_monitor(rms, self._vad.current_confidence(), mic_rms)

            speech_segment = self._vad.process_chunk(chunk)

            if speech_segment is None:
                # Still accumulating — check for interim ASR
                if self._incremental_enabled and self._asr_ctl.ready and self._vad.is_speaking():
                    buf_samples = self._vad.buffered_samples()
                    total_dur = buf_samples / 16000
                    elapsed = (buf_samples - self._last_interim_samples) / 16000
                    now = time.perf_counter()
                    cooldown = now - self._last_interim_check_time
                    if (
                        total_dur >= self._interim_interval
                        and elapsed >= self._interim_interval
                        and cooldown >= 1.0
                    ):
                        self._last_interim_check_time = now
                        self._enqueue_asr("interim", None)
                continue

            if not self._asr_ctl.ready:
                log.debug("ASR not ready, dropping segment")
                continue

            self._enqueue_asr("vad_flush", speech_segment)

    def _put_asr_drop_oldest(self, item: tuple[str, FloatArray | None] | None) -> None:
        """Enqueue with the drop-oldest policy (realtime-first discipline):
        never block — if the queue is full, evict the oldest entry so the
        most recent audio always gets a slot."""
        try:
            self._asr_queue.put_nowait(item)
        except queue.Full:
            try:
                dropped = self._asr_queue.get_nowait()
                if dropped is not None:
                    log.warning(f"ASR queue full, dropped {dropped[0]} segment")
            except queue.Empty:
                pass
            try:
                self._asr_queue.put_nowait(item)
            except queue.Full:
                log.warning("ASR queue still full after drop, skipping segment")

    def _enqueue_asr(self, seg_type: str, segment: FloatArray | None) -> None:
        self._put_asr_drop_oldest((seg_type, segment))

    def _asr_loop(self) -> None:
        while self._running:
            try:
                item = self._asr_queue.get(timeout=1.0)
            except queue.Empty:
                # Idle moment: recycle a bloated worker and probe liveness
                # while no audio is waiting.
                try:
                    self._asr_ctl.maybe_recycle_worker()
                    self._asr_ctl.maybe_ping_worker()
                except Exception:
                    log.error("ASR worker idle maintenance failed", exc_info=True)
                continue

            if item is None:
                break

            seg_type, segment = item

            if seg_type == "vad_flush":
                if segment is None:  # vad_flush always carries audio; guard for the type checker
                    continue
                if self._interim_active:
                    self._process_interim_final(segment)
                else:
                    self._process_segment(segment)
                self._interim_active = False
                self._interim_pending = ""
                self._last_interim_samples = 0
                self._last_interim_check_time = 0.0
                self._interim_committed_tail = ""
            elif seg_type == "interim":
                self._drain_interim_duplicates()
                self._do_interim_asr()
                self._last_interim_samples = self._vad.buffered_samples()

    def _drain_interim_duplicates(self) -> None:
        while True:
            try:
                item = self._asr_queue.get_nowait()
            except queue.Empty:
                break
            if item is None or item[0] != "interim":
                # Put the first non-interim item back under the same
                # drop-oldest discipline — never a blocking put (CORE-2).
                self._put_asr_drop_oldest(item)
                break

    # --- Translation ---

    def _compute_cost(
        self, prompt_tokens: int | None = None, completion_tokens: int | None = None
    ) -> float:
        """Per-million-token cost of the latest stats; callers may pass the
        locked snapshot to avoid reading mid-update values."""
        if prompt_tokens is None:
            prompt_tokens = self._total_prompt_tokens
        if completion_tokens is None:
            completion_tokens = self._total_completion_tokens
        if self._input_price > 0 or self._output_price > 0:
            return (
                prompt_tokens * self._input_price + completion_tokens * self._output_price
            ) / 1_000_000
        return 0.0

    def _translate_async(
        self, msg_id: int, text: str, source_lang: str, extra_langs: set[str] | None = None
    ) -> None:
        """Translate text and update UI with streaming display."""
        try:
            tl_start = time.perf_counter()
            translated: str | None = None
            for partial in self._translator.translate_iter(text, source_lang):
                translated = partial
                if self._overlay:
                    self._overlay.update_streaming(msg_id, partial)
            tl_ms = (time.perf_counter() - tl_start) * 1000
            # CORE-7: the executor runs 8 translation tasks concurrently;
            # counter updates must be atomic (read-modify-write).
            with self._stats_lock:
                self._translate_count += 1
                pt, ct = self._translator.last_usage
                self._total_prompt_tokens += pt
                self._total_completion_tokens += ct
                count = self._translate_count
                total_prompt = self._total_prompt_tokens
                total_completion = self._total_completion_tokens
            cost = self._compute_cost(total_prompt, total_completion)
            log.info(f"Translate ({tl_ms:.0f}ms): {len(translated or '')} chars")
            if self._log_transcript and translated:
                log.info(f"Translate content: {translated}")
            if translated:
                self._transcript.write_translation(msg_id, translated)
            else:
                self._transcript.finalize_no_translation(msg_id)
            if self._overlay:
                self._overlay.update_translation(msg_id, translated, tl_ms)
                self._overlay.update_stats(
                    self._asr_count,
                    count,
                    total_prompt,
                    total_completion,
                    cost,
                )
            if self._subwin and self._subwin.isVisible() and translated:
                tl_dict = {self._target_language: translated}
                if extra_langs:
                    self._translate_extra_langs(text, source_lang, extra_langs, tl_dict)
                self._subwin.update_text(text, tl_dict)
        except RepetitionError:
            log.warning("Repetition loop detected, model may not support structured output well")
            self._transcript.finalize_no_translation(msg_id)
            if self._overlay:
                self._overlay.update_translation(msg_id, f"[{t('error_repetition')}]", 0)
        except Exception as e:
            import openai

            if isinstance(
                e,
                (
                    openai.APIConnectionError,
                    openai.APITimeoutError,
                    openai.AuthenticationError,
                    openai.APIStatusError,
                    TimeoutError,
                    ConnectionError,
                ),
            ):
                log.warning(f"Translate error: {redact_text(str(e))}")
            else:
                log.error(f"Translate error: {redact_text(str(e))}", exc_info=True)
            self._transcript.finalize_no_translation(msg_id)
            key = classify_translate_error(
                e, using_proxy=getattr(self._translator, "proxy", "none") != "none"
            )
            if key:
                copy = t(key)
                self._report_error(copy)
                if self._overlay:
                    self._overlay.update_translation(msg_id, f"[{copy}]", 0)
            elif self._overlay:
                self._overlay.update_translation(msg_id, f"[error: {e}]", 0)

    def _report_error(self, text: str) -> None:
        """Route a classified error to the banner and the app's record."""
        if self._overlay is not None and hasattr(self._overlay, "show_error"):
            self._overlay.show_error(text)
        if self._error_reporter is not None:
            self._error_reporter(text)

    def _translate_extra_langs(
        self, text: str, source_lang: str, extra_langs: set[str], tl_dict: dict[str, str]
    ) -> None:
        """Translate into additional languages for subtitle window (parallel)."""
        from concurrent.futures import as_completed

        def _do_translate(lang: str) -> tuple[str, str]:
            translator = self._translator.with_target_language(lang)
            return lang, translator.translate(text, source_lang)

        futures = [self._tl_executor.submit(_do_translate, lang) for lang in extra_langs]

        def _record(future: Future[tuple[str, str]]) -> None:
            try:
                lang, result = future.result()
                tl_dict[lang] = result
                log.info(f"Extra translate [{lang}]: {len(result)} chars")
                if self._log_transcript:
                    log.info(f"Extra translate content: {result}")
            except Exception as e:
                import openai

                if isinstance(
                    e,
                    (
                        openai.APIConnectionError,
                        openai.APITimeoutError,
                        openai.AuthenticationError,
                        openai.APIStatusError,
                        TimeoutError,
                        ConnectionError,
                    ),
                ):
                    log.warning(f"Extra translate error: {redact_text(str(e))}")
                else:
                    log.error(f"Extra translate error: {redact_text(str(e))}", exc_info=True)

        for future in as_completed(futures):
            _record(future)

    def _translate_subwin_only(self, text: str, source_lang: str, extra_langs: set[str]) -> None:
        """Translate only for subtitle window when primary target == source language."""
        tl_dict = {self._target_language: text}  # same language, use original
        self._translate_extra_langs(text, source_lang, extra_langs, tl_dict)
        if self._subwin and self._subwin.isVisible():
            self._subwin.update_text(text, tl_dict)
