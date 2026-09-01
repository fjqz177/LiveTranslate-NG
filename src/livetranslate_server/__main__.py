"""Remote ASR server for LiveTranslate, using faster-whisper.

Run this on a machine with a GPU, then point LiveTranslate's "Remote Whisper"
engine at it (Settings -> Recognition -> Remote ASR Server URL). The client
(livetranslate.asr.remote) POSTs raw float32 PCM (16 kHz mono) to /transcribe
and gets back the transcription as JSON.

    uv run livetranslate-server --port 8765 --model large-v3 --device cuda

Or from a plain pip environment:

    pip install faster-whisper fastapi uvicorn numpy
    livetranslate-server --port 8765 --model large-v3 --device cuda --compute-type float16

Security (SEC-5): the server binds 127.0.0.1 by default. To serve the LAN,
bind 0.0.0.0 explicitly AND set a token so random LAN clients cannot inject
audio (GPU DoS / plaintext speech):

    livetranslate-server --host 0.0.0.0 --token <shared-secret> ...

The client then needs the same token (Settings -> Recognition -> Remote ASR
Server Token). Request bodies are capped at 25 MB (~6.5 minutes of audio).

Notes:
- For CUDA, faster-whisper/CTranslate2 needs the CUDA 12 cuBLAS and cuDNN 9
  libraries on the library path (e.g. `pip install nvidia-cublas-cu12
  nvidia-cudnn-cu12`, or a system CUDA install).
- The model is downloaded from Hugging Face on first run; set the HF_ENDPOINT
  env var to a mirror if direct access is slow.
"""

import argparse
import asyncio
import logging
import struct
import time

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

log = logging.getLogger("ASR-Server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(title="Remote ASR Server")
_model: WhisperModel = None
# Serialize GPU access: the model can only run one transcription at a time.
_gpu_lock = asyncio.Lock()

# SEC-5: cap request bodies (~6.5 minutes of 16 kHz mono float32). A LAN
# abuser must not be able to exhaust GPU memory with a multi-GB body.
MAX_BODY_BYTES = 25 * 1024 * 1024


def require_token(request: Request) -> None:
    """FastAPI dependency: enforces the shared token when one is configured."""
    expected = getattr(app.state, "token", None)
    if expected and request.headers.get("x-asr-token", "") != expected:
        raise HTTPException(status_code=401, detail="missing or invalid X-ASR-Token")


@app.on_event("startup")
def load_model():
    global _model
    args = app.state.args
    log.info(f"Loading model: {args.model} on {args.device} ({args.compute_type})")
    _model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    log.info(f"Model ready: {args.model}")


def _parse_request(request_body: bytes):
    """Decode the wire format: [uint32 lang_len][lang utf-8][float32 PCM]. Raises
    ValueError on any malformed/attacker-supplied body so the caller returns 400."""
    if len(request_body) < 4:
        raise ValueError("request too short")
    lang_len = struct.unpack("<I", request_body[:4])[0]
    if 4 + lang_len > len(request_body):
        raise ValueError("language length exceeds body")
    language = (
        request_body[4 : 4 + lang_len].decode("utf-8", errors="replace") if lang_len > 0 else None
    )
    if language in ("auto", ""):
        language = None
    audio_bytes = request_body[4 + lang_len :]
    if len(audio_bytes) % 4 != 0:
        raise ValueError("audio byte length is not a multiple of 4")
    return language, np.frombuffer(audio_bytes, dtype=np.float32)


def _run_transcription(audio: np.ndarray, language):
    segments, info = _model.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    text_parts = [seg.text.strip() for seg in segments]
    return " ".join(text_parts).strip(), info.language


@app.post("/transcribe")
async def transcribe(request: Request, _auth: None = Depends(require_token)):
    """Accept raw float32 PCM audio at 16kHz mono. Return transcription."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_BODY_BYTES:
                return JSONResponse({"error": "request too large"}, status_code=413)
        except ValueError:
            return JSONResponse({"error": "invalid content-length"}, status_code=400)
    request_body = await request.body()
    if len(request_body) > MAX_BODY_BYTES:
        return JSONResponse({"error": "request too large"}, status_code=413)
    try:
        language, audio = _parse_request(request_body)
    except (ValueError, struct.error) as e:
        return JSONResponse({"error": f"bad request: {e}"}, status_code=400)

    duration = len(audio) / 16000
    t0 = time.time()
    # Run the blocking GPU call off the event loop, one at a time.
    async with _gpu_lock:
        full_text, detected_lang = await run_in_threadpool(_run_transcription, audio, language)
    elapsed = time.time() - t0

    log.info(
        f"Transcribed {duration:.1f}s audio in {elapsed:.2f}s: [{detected_lang}] {full_text[:80]}"
    )

    if not full_text:
        return {"text": None, "language": detected_lang, "elapsed": elapsed}

    return {
        "text": full_text,
        "language": detected_lang,
        "language_name": detected_lang,
        "elapsed": elapsed,
    }


@app.get("/health", dependencies=[Depends(require_token)])
async def health():
    return {"status": "ok", "model": app.state.args.model}


def main():
    parser = argparse.ArgumentParser(description="Remote ASR Server")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default 127.0.0.1; use 0.0.0.0 for LAN sharing, "
        "and pair it with --token)",
    )
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--model", default="medium", help="Whisper model size")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--compute-type", default="float16", help="Compute type")
    parser.add_argument(
        "--token",
        default="",
        help="Shared token (SEC-5): when set, every request must carry the "
        "X-ASR-Token header with this value",
    )
    args = parser.parse_args()

    app.state.args = args
    app.state.token = args.token or None
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
