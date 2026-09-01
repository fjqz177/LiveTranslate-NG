"""Windows WASAPI loopback backend (pyaudiowpatch).

Faithful port of the pre-rewrite AudioCapture: command-queue-driven stream
switching (all open/close runs on the read thread), 2s default-device
follow, optional microphone mixing, mic-only mode and silence fallback.
Only the protocol surface changed; the resample math moved to
audio/resample.py verbatim.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import numpy as np
import pyaudiowpatch as pyaudio

from livetranslate.audio.backend import DeviceInfo
from livetranslate.audio.resample import mic_rms, resample_to_mono

log = logging.getLogger("LiveTranslate.Audio")

DEVICE_CHECK_INTERVAL = 2.0  # seconds


def _wasapi_host_index(pa: Any) -> int | None:
    for i in range(pa.get_host_api_count()):
        info = pa.get_host_api_info_by_index(i)
        if "WASAPI" in info["name"]:
            return int(info["index"])
    return None


def _dev_info(dev: Any) -> DeviceInfo:
    return DeviceInfo(
        id=dev["name"],
        name=dev["name"],
        kind="loopback" if dev.get("isLoopbackDevice", False) else "input",
        channels=int(dev["maxInputChannels"] or dev["maxOutputChannels"]),
        default_rate=int(dev["defaultSampleRate"]),
    )


def list_outputs() -> list[DeviceInfo]:
    """WASAPI output devices (loopback-eligible; loopback pairs excluded)."""
    pa = pyaudio.PyAudio()
    try:
        host = _wasapi_host_index(pa)
        if host is None:
            return []
        devices = []
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if (
                dev["hostApi"] == host
                and dev["maxOutputChannels"] > 0
                and not dev.get("isLoopbackDevice", False)
            ):
                devices.append(_dev_info(dev))
        return devices
    finally:
        pa.terminate()


def list_inputs() -> list[DeviceInfo]:
    """WASAPI input (microphone) devices; loopback pairs excluded."""
    pa = pyaudio.PyAudio()
    try:
        host = _wasapi_host_index(pa)
        if host is None:
            return []
        devices = []
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if (
                dev["hostApi"] == host
                and dev["maxInputChannels"] > 0
                and not dev.get("isLoopbackDevice", False)
            ):
                devices.append(_dev_info(dev))
        return devices
    finally:
        pa.terminate()


class WasapiBackend:
    """Capture system audio via WASAPI loopback using pyaudiowpatch.

    device_id semantics: None = follow the system default output,
    "__disabled__" = loopback off (mic-only mode), anything else = the
    WASAPI device name. mic_id: None = disabled, "default" = system default
    input, otherwise a device name.
    """

    name = "wasapi"

    @property
    def device_id(self) -> str | None:
        return self._device_name

    def __init__(self) -> None:
        self.sample_rate = 16000
        self.chunk_duration = 0.032
        self.audio_queue: queue.Queue[tuple[np.ndarray, float | None]] = queue.Queue(maxsize=100)
        self._stream: Any = None
        self._running = False
        self._device_name: str | None = None
        self._pa: Any = None
        self._read_thread: threading.Thread | None = None
        self._native_channels = 2
        self._native_rate = 44100
        self._current_device_name: str | None = None
        self._loopback_disabled = False
        self._lock = threading.Lock()
        # Device/mic changes are posted as commands and executed only by the
        # read thread, so stream open/close never races with stop() or reads.
        self._commands: queue.Queue[tuple[str, str | None]] = queue.Queue()
        # Microphone input
        self._mic_device_name: str | None = None
        self._mic_stream: Any = None
        self._mic_native_rate = 44100
        self._mic_native_channels = 1
        self._mic_buf = np.array([], dtype=np.float32)
        self._last_error: str | None = None

    # -- AudioBackend -------------------------------------------------------

    def list_outputs(self) -> list[DeviceInfo]:
        return list_outputs()

    def list_inputs(self) -> list[DeviceInfo]:
        return list_inputs()

    def start(
        self,
        device_id: str | None = None,
        mic_id: str | None = None,
        sample_rate: int = 16000,
        chunk_ms: int = 32,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_ms / 1000.0
        self._device_name = device_id
        self._mic_device_name = mic_id
        # AUD-2: terminate any previous PortAudio context explicitly instead
        # of relying on __del__ — every start/stop cycle previously leaked
        # one native instance until GC (unpredictable, ignores __del__ bugs).
        if getattr(self, "_pa", None) is not None:
            self._pa.terminate()
        self._pa = pyaudio.PyAudio()
        self._loopback_disabled = device_id == "__disabled__"
        if not self._loopback_disabled:
            self._open_stream()
        else:
            log.info("Loopback disabled (mic-only mode)")
        if self._mic_device_name:
            try:
                self._open_mic_stream()
            except Exception as e:
                log.warning(f"Failed to open mic on start: {e}")
        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        log.info("Audio capture started")

    def read_chunk(self) -> tuple[np.ndarray, float | None] | None:
        try:
            return self.audio_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def switch_device(self, device_id: str | None) -> None:
        if device_id == self._device_name:
            return
        log.info(f"Audio device change requested: {self._device_name} -> {device_id}")
        self._commands.put(("set_device", device_id))

    def switch_mic(self, mic_id: str | None) -> None:
        if mic_id == self._mic_device_name:
            return
        log.info(f"Mic device change requested: {self._mic_device_name} -> {mic_id}")
        self._commands.put(("set_mic", mic_id))

    def stop(self) -> None:
        self._running = False
        self._commands.put(("wake", None))
        if self._read_thread:
            self._read_thread.join(timeout=5)
        # Stream closes happen only after the read thread is gone; the lock
        # still guards against a stuck thread inside _restart_stream().
        with self._lock:
            self._close_stream()
            self._close_mic_stream()
            # AUD-2: release the PortAudio context; __del__ stays as a
            # defensive backstop only.
            if getattr(self, "_pa", None) is not None:
                self._pa.terminate()
                self._pa = None
        log.info("Audio capture stopped")

    def diagnostics(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "device": self._current_device_name or self._device_name,
            "rate": self.sample_rate,
            "status": "running" if self._running else "stopped",
            "last_error": self._last_error,
        }

    # -- WASAPI plumbing (ported from the pre-rewrite AudioCapture) ---------

    def _get_wasapi_info(self) -> Any:
        for i in range(self._pa.get_host_api_count()):
            info = self._pa.get_host_api_info_by_index(i)
            if "WASAPI" in info["name"]:
                return info
        return None

    @staticmethod
    def _query_current_default() -> str | None:
        """Create a fresh PA instance to get the actual current default device."""
        pa = pyaudio.PyAudio()
        try:
            for i in range(pa.get_host_api_count()):
                info = pa.get_host_api_info_by_index(i)
                if "WASAPI" in info["name"]:
                    return str(pa.get_device_info_by_index(info["defaultOutputDevice"])["name"])
        finally:
            pa.terminate()
        return None

    def _find_loopback_device(self) -> Any:
        """Find WASAPI loopback device for the default output."""
        wasapi_info = self._get_wasapi_info()
        if wasapi_info is None:
            raise RuntimeError("WASAPI host API not found")

        default_output = self._pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        log.info(f"Default output: {default_output['name']}")

        target_name = self._device_name or default_output["name"]

        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if (
                dev["hostApi"] == wasapi_info["index"]
                and dev.get("isLoopbackDevice", False)
                and target_name in dev["name"]
            ):
                return dev

        # Fallback: any loopback device
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice", False):
                return dev

        raise RuntimeError("No WASAPI loopback device found")

    def _open_stream(self) -> None:
        """Open stream for current default loopback device."""
        loopback_dev = self._find_loopback_device()
        self._native_channels = int(loopback_dev["maxInputChannels"])
        self._native_rate = int(loopback_dev["defaultSampleRate"])
        self._current_device_name = loopback_dev["name"]

        log.info(f"Loopback device: {loopback_dev['name']}")
        log.info(
            f"Native: {self._native_rate}Hz, {self._native_channels}ch -> {self.sample_rate}Hz mono"
        )

        native_chunk = int(self._native_rate * self.chunk_duration)

        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._native_channels,
            rate=self._native_rate,
            input=True,
            input_device_index=loopback_dev["index"],
            frames_per_buffer=native_chunk,
        )

    def _close_stream(self) -> None:
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _find_mic_device(self) -> Any:
        """Find WASAPI input device matching _mic_device_name, or system default."""
        wasapi_info = self._get_wasapi_info()
        if wasapi_info is None:
            raise RuntimeError("WASAPI host API not found")
        if self._mic_device_name in ("__default__", "default"):
            dev = self._pa.get_device_info_by_index(wasapi_info["defaultInputDevice"])
            if dev["maxInputChannels"] > 0:
                return dev
            raise RuntimeError("Default input device has no input channels")
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if (
                dev["hostApi"] == wasapi_info["index"]
                and dev["maxInputChannels"] > 0
                and not dev.get("isLoopbackDevice", False)
                and dev["name"] == self._mic_device_name
            ):
                return dev
        raise RuntimeError(f"Mic device not found: {self._mic_device_name}")

    def _open_mic_stream(self) -> None:
        """Open microphone input stream."""
        dev = self._find_mic_device()
        self._mic_native_channels = int(dev["maxInputChannels"])
        self._mic_native_rate = int(dev["defaultSampleRate"])
        native_chunk = int(self._mic_native_rate * self.chunk_duration)
        log.info(
            f"Mic device: {dev['name']} ({self._mic_native_rate}Hz, {self._mic_native_channels}ch)"
        )
        self._mic_stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._mic_native_channels,
            rate=self._mic_native_rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=native_chunk,
        )

    def _close_mic_stream(self) -> None:
        if self._mic_stream:
            try:
                self._mic_stream.stop_stream()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None

    def _restart_stream(self) -> None:
        """Restart stream with new default device."""
        with self._lock:
            self._close_stream()
            # stop() may have terminated _pa already (join timeout) — the
            # read thread must not crash on a None context.
            if self._pa is not None:
                self._pa.terminate()
            self._pa = pyaudio.PyAudio()
            if not self._loopback_disabled:
                self._open_stream()
            else:
                log.info("Loopback disabled (mic-only mode)")
            # Re-open mic if active
            if self._mic_device_name:
                self._close_mic_stream()
                try:
                    self._open_mic_stream()
                except Exception as e:
                    log.warning(f"Failed to re-open mic after restart: {e}")

    @staticmethod
    def _pop_nowait(q: queue.Queue) -> object | None:
        """One non-blocking pop; the exception boundary lives here so the
        drain loop stays exception-free (PERF203)."""
        try:
            return q.get_nowait()
        except queue.Empty:
            return None

    @staticmethod
    def _drain_audio_queue(q: queue.Queue) -> None:
        """Drop every queued chunk (device changed; stale audio is poison)."""
        while WasapiBackend._pop_nowait(q) is not None:
            pass

    def _drain_commands(self) -> None:
        """Execute pending device/mic change commands (read thread only)."""
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return
            kind, value = cmd
            if kind == "set_device":
                device_name = value
                if device_name == self._device_name:
                    continue
                self._device_name = device_name
                self._loopback_disabled = device_name == "__disabled__"
                log.info(f"Audio device changed: -> {device_name}")
                try:
                    self._restart_stream()
                    self._drain_audio_queue(self.audio_queue)
                    log.info(f"Audio capture restarted on: {self._current_device_name}")
                except Exception as e:
                    log.error(f"Restart after device change failed: {e}")
                    self._last_error = str(e)
                    time.sleep(0.5)
            elif kind == "set_mic":
                device_name = value
                if device_name == self._mic_device_name:
                    continue
                self._mic_device_name = device_name
                log.info(f"Mic device changed: -> {device_name}")
                self._close_mic_stream()
                self._mic_buf = np.array([], dtype=np.float32)
                if device_name:
                    try:
                        self._open_mic_stream()
                        log.info(f"Mic stream opened: {device_name}")
                    except Exception as e:
                        log.error(f"Failed to open mic: {e}")
                        self._last_error = str(e)
                else:
                    log.info("Mic disabled")
            # "wake" is a no-op used to unblock stop()

    def _read_loop(self) -> None:
        """Synchronous read loop in a background thread."""
        last_device_check = time.monotonic()

        while self._running:
            self._drain_commands()

            # Auto-switch only when using system default
            now = time.monotonic()
            if now - last_device_check > DEVICE_CHECK_INTERVAL:
                last_device_check = now
                if self._device_name is None:
                    try:
                        current_default = self._query_current_default()
                        if (
                            current_default
                            and self._current_device_name
                            and current_default not in self._current_device_name
                        ):
                            log.info(
                                f"System default output changed: "
                                f"{self._current_device_name} -> {current_default}"
                            )
                            log.info("Restarting audio capture for new device...")
                            self._restart_stream()
                            log.info(f"Audio capture restarted on: {self._current_device_name}")
                    except Exception as e:
                        log.warning(f"Device check error: {e}")

            # Read loopback chunk or generate silence for mic-only mode
            loopback_audio: np.ndarray | None = None
            if self._loopback_disabled:
                time.sleep(self.chunk_duration)
                n_samples = int(self.sample_rate * self.chunk_duration)
                loopback_audio = np.zeros(n_samples, dtype=np.float32)
            else:
                native_chunk = int(self._native_rate * self.chunk_duration)
                try:
                    data = None
                    with self._lock:
                        if not self._stream:
                            time.sleep(0.005)
                            continue
                        if self._stream.get_read_available() >= native_chunk:
                            data = self._stream.read(native_chunk, exception_on_overflow=False)
                    if data is not None:
                        loopback_audio = resample_to_mono(
                            data,
                            self._native_channels,
                            self._native_rate,
                            self.sample_rate,
                        )
                except Exception as e:
                    if not self._commands.empty():
                        continue
                    log.warning(f"Read error (device may have changed): {e}")
                    self._last_error = str(e)
                    try:
                        time.sleep(0.5)
                        self._restart_stream()
                        log.info("Stream restarted after read error")
                    except Exception as re:
                        log.error(f"Restart failed: {re}")
                        time.sleep(1)
                    continue

            # Drain all available mic data into buffer
            if self._mic_stream:
                try:
                    avail = self._mic_stream.get_read_available()
                    if avail > 0:
                        mic_data = self._mic_stream.read(avail, exception_on_overflow=False)
                        mic_16k = resample_to_mono(
                            mic_data,
                            self._mic_native_channels,
                            self._mic_native_rate,
                            self.sample_rate,
                        )
                        self._mic_buf = np.concatenate([self._mic_buf, mic_16k])
                except Exception as e:
                    log.warning(f"Mic read error: {e}")

            if loopback_audio is None:
                time.sleep(0.005)
                continue

            # Mix: take matching length from mic buffer
            audio = loopback_audio
            rms: float | None = None
            if len(self._mic_buf) > 0:
                n = len(loopback_audio)
                if len(self._mic_buf) >= n:
                    mic_chunk = self._mic_buf[:n]
                    self._mic_buf = self._mic_buf[n:]
                else:
                    mic_chunk = np.zeros(n, dtype=np.float32)
                    mic_chunk[: len(self._mic_buf)] = self._mic_buf
                    self._mic_buf = np.array([], dtype=np.float32)
                rms = mic_rms(mic_chunk)
                audio = loopback_audio + mic_chunk

            try:
                self.audio_queue.put_nowait((audio, rms))
            except queue.Full:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait((audio, rms))

    def __del__(self) -> None:
        if getattr(self, "_pa", None):
            self._pa.terminate()
