import contextlib
import logging
import os
import re
from pathlib import Path

from livetranslate.core.paths import models_dir

log = logging.getLogger("LiveTranslate.ModelManager")

_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@contextlib.contextmanager
def _proxy_env(proxy: str):
    """Temporarily route all download backends through a proxy.

    proxy:
        "system" / "" / None -> leave ambient env & OS proxy untouched
        "none"               -> force-disable any proxy for this download
        a URL                -> send urllib/requests/httpx traffic through it

    Covers torch.hub (urllib) and the httpx-based hub_downloader (SelfServe
    P0-A2) — both honor the *_PROXY env vars; urllib additionally gets an
    explicit opener so a previously cached default opener cannot bypass the
    setting.
    """
    import urllib.request

    if proxy in ("system", "", None):
        yield
        return
    saved_env: dict = {key: os.environ.get(key) for key in _PROXY_ENV_KEYS}
    saved_no_proxy = os.environ.get("NO_PROXY")
    saved_opener = getattr(urllib.request, "_opener", None)
    try:
        if proxy == "none":
            for key in _PROXY_ENV_KEYS:
                os.environ.pop(key, None)
            os.environ["NO_PROXY"] = "*"
            handler = urllib.request.ProxyHandler({})
        else:
            for key in _PROXY_ENV_KEYS:
                os.environ[key] = proxy
            os.environ.pop("NO_PROXY", None)
            handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        urllib.request.install_opener(urllib.request.build_opener(handler))
        log.info(f"Download proxy active: {_mask_proxy_url(proxy)}")
        yield
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if saved_no_proxy is None:
            os.environ.pop("NO_PROXY", None)
        else:
            os.environ["NO_PROXY"] = saved_no_proxy
        urllib.request.install_opener(saved_opener)


MODELS_DIR = models_dir()

# Registry tables and pure normalization helpers live in model_registry.py
# (no I/O, fully unit-tested). Re-exported here for backward compatibility;
# new code should import from model_registry directly.
# Filesystem cache detection lives in model_cache.py (models_dir
# parameterized, unit-tested against historical cache-layout bugs).
# Re-exported here for backward compatibility.
from livetranslate.modeling.cache import (
    _has_silero_pkg,
    _ms_model_path,
    get_local_model_path,
    get_whisper_local_path,
    is_asr_cached,
    is_faster_whisper_model_dir,
    is_silero_cached,
    list_local_faster_whisper_models,
    local_faster_whisper_display_name,
    qwen_weights_present,
    resolve_custom_whisper_model,
)
from livetranslate.modeling.registry import (
    ASR_DISPLAY_NAMES,
    ASR_MODEL_IDS,
    ASR_MODEL_IDS_HF,
    DEFAULT_FUNASR_MODEL,
    FUNASR_LEGACY_ENGINE_ALIASES,
    FUNASR_MODEL_PROFILES,
    asr_model_id,
    funasr_display_name,
    funasr_model_id,
    funasr_model_options,
    funasr_profile,
    funasr_supports_padding,
    migrate_funasr_settings,
    normalize_asr_engine_selection,
    normalize_funasr_model_key,
    whisper_model_id,
)
from livetranslate.modeling.registry import (
    CACHE_MODELS as _CACHE_MODELS,
)
from livetranslate.modeling.registry import (
    MODEL_SIZE_BYTES as _MODEL_SIZE_BYTES,
)
from livetranslate.modeling.registry import (
    WHISPER_SIZES as _WHISPER_SIZES,
)

# Backward-compatible surface: downstream modules historically import these
# names from model_manager; keep them importable here.
__all__ = [
    "ASR_DISPLAY_NAMES",
    "ASR_MODEL_IDS",
    "ASR_MODEL_IDS_HF",
    "DEFAULT_FUNASR_MODEL",
    "FUNASR_LEGACY_ENGINE_ALIASES",
    "FUNASR_MODEL_PROFILES",
    "asr_model_id",
    "funasr_display_name",
    "funasr_model_id",
    "funasr_model_options",
    "funasr_profile",
    "funasr_supports_padding",
    "get_local_model_path",
    "get_whisper_local_path",
    "is_asr_cached",
    "is_faster_whisper_model_dir",
    "is_silero_cached",
    "list_local_faster_whisper_models",
    "local_faster_whisper_display_name",
    "migrate_funasr_settings",
    "normalize_asr_engine_selection",
    "normalize_funasr_model_key",
    "qwen_weights_present",
    "resolve_custom_whisper_model",
]


def apply_cache_env():
    """Point all model caches to ./models/."""
    resolved = str(MODELS_DIR.resolve())
    os.environ["MODELSCOPE_CACHE"] = os.path.join(resolved, "modelscope")
    os.environ["HF_HOME"] = os.path.join(resolved, "huggingface")
    os.environ["TORCH_HOME"] = os.path.join(resolved, "torch")
    log.info(f"Cache env set: {resolved}")


def get_missing_models(engine, model_size, hub) -> list:
    missing = []
    if not is_silero_cached():
        missing.append(
            {
                "name": "Silero VAD",
                "type": "silero-vad",
                "estimated_bytes": _MODEL_SIZE_BYTES["silero-vad"],
            }
        )
    if not is_asr_cached(engine, model_size, hub):
        if engine == "whisper" and model_size not in _WHISPER_SIZES:
            return missing
        if engine == "funasr" or engine in FUNASR_LEGACY_ENGINE_ALIASES:
            model_key = (
                FUNASR_LEGACY_ENGINE_ALIASES[engine]
                if engine in FUNASR_LEGACY_ENGINE_ALIASES
                else normalize_funasr_model_key(model_size)
            )
            profile = funasr_profile(model_key)
            key = f"funasr:{model_key}"
            display = profile["display_name"]
            estimated_bytes = profile["estimated_bytes"]
        elif engine == "whisper":
            key = engine if engine != "whisper" else f"whisper-{model_size}"
            display = f"Whisper {model_size}"
            estimated_bytes = _MODEL_SIZE_BYTES.get(key, 0)
        else:
            key = engine
            display = ASR_DISPLAY_NAMES.get(engine, engine)
            estimated_bytes = _MODEL_SIZE_BYTES.get(key, 0)
        missing.append(
            {
                "name": display,
                "type": key,
                "estimated_bytes": estimated_bytes,
            }
        )
    return missing


# SEC-4: torch.hub fallback for Silero VAD. Pinned to a stable release tag
# instead of the mutable master branch; the silero-vad pip package (checked
# first in download_silero) remains the primary source.
SILERO_HUB_REF = "snakers4/silero-vad:v5.1"

# SSL-inspecting proxies (Python 3.13 VERIFY_X509_STRICT) break strict
# verification. Relaxing verification silently would gut the supply chain
# trust model — it only happens when the user explicitly opts in.
RELAX_SSL_ENV = "LIVETRANSLATE_RELAX_SSL"

_PROXY_USERINFO_RE = re.compile(r"(?<=//)[^/@\s]+(?=@)")


def _mask_proxy_url(url: str) -> str:
    """Mask the userinfo segment of a proxy URL (CORE-4): 'http://user:pw@h:1'
    -> 'http://***@h:1'. Proxy credentials must never reach logs or the
    diagnostics bundle."""
    return _PROXY_USERINFO_RE.sub("***", url)


def download_silero(proxy: str = "system"):
    if _has_silero_pkg():
        log.info("Silero VAD bundled by silero-vad package, no download needed")
        return
    import torch

    log.info("Downloading Silero VAD...")
    with _proxy_env(proxy):
        try:
            model, _ = torch.hub.load(
                repo_or_dir=SILERO_HUB_REF,
                model="silero_vad",
                trust_repo=True,
            )
        except Exception as exc:
            if "CERTIFICATE_VERIFY" not in str(exc):
                raise
            if os.environ.get(RELAX_SSL_ENV) != "1":
                raise RuntimeError(
                    "SSL certificate verification failed while downloading Silero VAD. "
                    "Fix your system certificates/proxy — or explicitly allow the "
                    "insecure fallback with " + RELAX_SSL_ENV + "=1 "
                    "(only for SSL-inspecting corporate proxies)."
                ) from exc
            log.warning("SSL strict verification failed, retrying with relaxed flags (opt-in)")
            model, _ = _load_silero_relaxed_ssl()
    del model
    log.info("Silero VAD downloaded")


def _load_silero_relaxed_ssl():
    # Python 3.13 enables VERIFY_X509_STRICT by default, rejecting certificates
    # without an Authority Key Identifier (common behind SSL-inspecting proxies).
    # Only reachable through the explicit LIVETRANSLATE_RELAX_SSL=1 opt-in.
    import ssl

    import torch

    strict = getattr(ssl, "VERIFY_X509_STRICT", 0)
    original = ssl._create_default_https_context

    def relaxed_context(*args, **kwargs):
        ctx = ssl.create_default_context(*args, **kwargs)
        ctx.verify_flags &= ~strict
        return ctx

    ssl._create_default_https_context = relaxed_context
    try:
        return torch.hub.load(
            repo_or_dir=SILERO_HUB_REF,
            model="silero_vad",
            trust_repo=True,
            force_reload=True,
        )
    finally:
        ssl._create_default_https_context = original


def ensure_qwen_weights(model_dir, hub: str = "ms") -> None:
    """Fetch Qwen3-0.6B weights into a nano model's embedded subdir (one-time).

    Kept off the ASR worker startup path: its 180s ready timeout would otherwise
    kill the process mid-download on slow links.
    """
    qwen_dir = Path(model_dir) / "Qwen3-0.6B"
    if not qwen_dir.is_dir():
        return
    if any(f.suffix in (".safetensors", ".bin") for f in qwen_dir.iterdir()):
        return
    log.info("Downloading Qwen3-0.6B weights (one-time)...")
    from livetranslate.modeling.hub_downloader import download_repo

    download_repo(
        "Qwen/Qwen3-0.6B",
        "hf" if hub == "hf" else "ms",
        local_dir=qwen_dir,
        ignore_patterns=("*.gguf",),
    )
    log.info("Qwen3-0.6B weights downloaded")


def download_asr(engine, model_size="medium", hub="ms", proxy="system"):
    from livetranslate.modeling.hub_downloader import download_repo

    resolved = str(MODELS_DIR.resolve())
    ms_cache = Path(resolved) / "modelscope"
    hf_cache = Path(resolved) / "huggingface" / "hub"
    with _proxy_env(proxy):
        if engine == "funasr" or engine in FUNASR_LEGACY_ENGINE_ALIASES:
            model_key = (
                FUNASR_LEGACY_ENGINE_ALIASES[engine]
                if engine in FUNASR_LEGACY_ENGINE_ALIASES
                else normalize_funasr_model_key(model_size)
            )
            if hub == "ms":
                model_id = funasr_model_id(model_key, "ms")
                log.info(f"Downloading {model_id} from ModelScope...")
                download_repo(model_id, "ms", cache_dir=ms_cache)
            else:
                model_id = funasr_model_id(model_key, "hf")
                log.info(f"Downloading {model_id} from HuggingFace...")
                download_repo(model_id, "hf", cache_dir=hf_cache)
            funasr_dir = get_local_model_path("funasr", hub=hub, funasr_model=model_key)
            neutralize_funasr_requirements(funasr_dir)
            if funasr_dir and funasr_profile(model_key)["family"] == "funasr-nano":
                ensure_qwen_weights(funasr_dir, hub=hub)
        elif engine == "anime-whisper":
            # HF-only, ignore hub setting
            model_id = ASR_MODEL_IDS[engine]
            log.info(f"Downloading {model_id} from HuggingFace...")
            download_repo(model_id, "hf", cache_dir=hf_cache)
        elif engine == "whisper":
            if model_size not in _WHISPER_SIZES:
                raise ValueError(f"Invalid local faster-whisper model: {model_size}")
            model_id = whisper_model_id(model_size)
            if hub == "ms":
                log.info(f"Downloading {model_id} from ModelScope...")
                download_repo(model_id, "ms", cache_dir=ms_cache)
            else:
                log.info(f"Downloading {model_id} from HuggingFace...")
                download_repo(model_id, "hf", cache_dir=hf_cache)
    log.info(f"ASR model downloaded: {engine}")


def neutralize_funasr_requirements(model_dir) -> None:
    """Skip FunASR's load-time `pip install -r requirements.txt`.

    With trust_remote_code=True, FunASR detects requirements.txt in the model
    dir and runs pip in a subprocess whose output is swallowed (PIPE). On a slow
    or proxy-blocked PyPI this hangs indefinitely with no log output, and it can
    pull heavy unused deps (e.g. gradio). All real deps already live in the venv,
    so rename the file out of the way to make the check miss.
    """
    if not model_dir:
        return
    req = Path(model_dir) / "requirements.txt"
    if req.exists():
        try:
            req.replace(req.with_name("requirements.txt.bundled"))
            log.info(f"Skipped FunASR requirements install: {req}")
        except OSError as exc:
            log.warning(f"Failed to neutralize {req}: {exc}")


def dir_size(path) -> int:
    total = 0
    try:
        for f in Path(path).rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except (OSError, PermissionError):
        pass
    return total


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.1f} MB"
    return f"{size_bytes / (1024**3):.2f} GB"


def get_cache_entries():
    """Scan ./models/ for cached models."""
    entries = []
    hf_base = MODELS_DIR / "huggingface" / "hub"
    torch_base = MODELS_DIR / "torch" / "hub"

    for entry in _CACHE_MODELS:
        if len(entry) == 3:
            name, engine, model_key = entry
        else:
            name, engine = entry
            model_key = None
        ms_org, ms_model = asr_model_id(engine, "ms", model_key).split("/")
        hf_org, hf_model = asr_model_id(engine, "hf", model_key).split("/")
        ms_path = _ms_model_path(ms_org, ms_model)
        hf_path = hf_base / f"models--{hf_org}--{hf_model}"
        if ms_path.exists():
            entries.append((f"{name} (ModelScope)", ms_path))
        if hf_path.exists():
            entries.append((f"{name} (HuggingFace)", hf_path))

    for size in _WHISPER_SIZES:
        org, model = whisper_model_id(size).split("/")
        ms_path = _ms_model_path(org, model)
        hf_path = hf_base / f"models--Systran--faster-whisper-{size}"
        if ms_path.exists() and is_asr_cached("whisper", size, "ms"):
            entries.append((f"Whisper {size} (ModelScope)", ms_path))
        if hf_path.exists() and is_asr_cached("whisper", size, "hf"):
            entries.append((f"Whisper {size}", hf_path))

    entries.extend(
        (f"Whisper Local: {item['name']}", Path(item["path"]))
        for item in list_local_faster_whisper_models()
    )

    if torch_base.exists():
        for d in sorted(torch_base.glob("snakers4_silero-vad*")):
            if d.is_dir():
                entries.append(("Silero VAD", d))
                break

    return entries
