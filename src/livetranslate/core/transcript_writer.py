"""Persistent transcript writer.

Writes ASR results to per-session files (original / translation / all),
appending immediately as each segment is recognized so nothing is lost
when the in-memory message buffer rotates.
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import TextIO

from livetranslate.core.privacy import redact_text

log = logging.getLogger("LiveTranslate.Transcript")


class TranscriptWriter:
    KINDS = ("original", "translation", "all")

    def __init__(self, base_dir: Path):
        self._base_dir = Path(base_dir)
        self._enabled = True
        self._lock = threading.Lock()
        self._files: dict[str, TextIO | None] = {}
        self._paths: dict[str, str] = {}
        self._pending: dict[int, tuple[str, str]] = {}  # msg_id -> (timestamp, original)
        self._opened = False
        self._session_ts: str | None = None

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        with self._lock:
            if enabled == self._enabled:
                if enabled and not self._opened:
                    self._open_session_locked()
                return
            self._enabled = enabled
            if enabled and not self._opened:
                self._open_session_locked()

    def is_enabled(self) -> bool:
        return self._enabled

    def session_paths(self) -> dict[str, str]:
        return dict(self._paths)

    def _open_session_locked(self) -> None:
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.error(redact_text(f"Failed to create transcript dir {self._base_dir}: {e}"))
            return
        self._session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        header_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for kind in self.KINDS:
            path = self._base_dir / f"livetrans_{self._session_ts}_{kind}.txt"
            try:
                # line buffered so tail -f works; append mode in case session reopens
                fp = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115  # kept open for session (tail -f)
                fp.write(f"# Session started at {header_ts}\n")
                self._files[kind] = fp
                self._paths[kind] = str(path)
            except OSError as e:
                log.error(redact_text(f"Failed to open transcript file {path}: {e}"))
                self._files[kind] = None
        self._opened = True
        log.info(redact_text(f"Transcripts -> {self._base_dir}"))

    def write_original(self, msg_id: int, timestamp: str, original: str) -> None:
        if not original:
            return
        with self._lock:
            if not self._enabled:
                return
            if not self._opened:
                self._open_session_locked()
            self._pending[msg_id] = (timestamp, original)
            self._write_locked("original", f"[{timestamp}] {original}\n")

    def write_translation(self, msg_id: int, translation: str) -> None:
        if not translation:
            return
        with self._lock:
            if not self._enabled:
                return
            if not self._opened:
                self._open_session_locked()
            entry = self._pending.pop(msg_id, None)
            if entry is None:
                ts = datetime.now().strftime("%H:%M:%S")
                self._write_locked("translation", f"[{ts}] {translation}\n")
                self._write_locked("all", f"[{ts}] -> {translation}\n\n")
                return
            ts, original = entry
            self._write_locked("translation", f"[{ts}] {translation}\n")
            self._write_locked("all", f"[{ts}] {original}\n  -> {translation}\n\n")

    def finalize_no_translation(self, msg_id: int) -> None:
        """Mark a message complete without a translation (same-language or error)."""
        with self._lock:
            if not self._enabled:
                self._pending.pop(msg_id, None)
                return
            if not self._opened:
                self._open_session_locked()
            entry = self._pending.pop(msg_id, None)
            if entry is None:
                return
            ts, original = entry
            self._write_locked("all", f"[{ts}] {original}\n\n")

    def _write_locked(self, kind: str, text: str) -> None:
        fp = self._files.get(kind)
        if fp is None:
            return
        try:
            fp.write(text)
        except OSError as e:
            log.warning(f"Transcript write failed ({kind}): {e}")

    def close(self) -> None:
        with self._lock:
            for fp in self._files.values():
                if fp is None:
                    continue
                try:
                    fp.flush()
                    fp.close()
                except Exception:
                    pass
            self._files.clear()
            self._pending.clear()
            self._opened = False
