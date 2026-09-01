import json
import logging
import threading
import time
from collections.abc import Iterator
from typing import Any, cast

import httpx
from openai import BadRequestError, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from livetranslate.core.i18n import LANGUAGE_DISPLAY

log = logging.getLogger("LiveTranslate.TL")

DEFAULT_PROMPT = (
    "You are a real-time subtitle translator. Translate {source_lang} into {target_lang}.\n"
    "Rules:\n"
    "- Output ONLY one single best translation, nothing else.\n"
    "- Never include alternatives, parenthetical options, annotations, or explanations.\n"
    "- Keep proper nouns, names, and brand names untranslated.\n"
    "- Translate repeated expressions concisely, not mechanically word-for-word.\n"
    "- Keep subtitles fluent and natural; avoid overly literal or stiff phrasing.\n"
    "- Auto-correct likely ASR errors based on context and common sense."
)

PROMPT_PRESETS = {
    "daily": (
        "You are a real-time subtitle translator for casual conversation. "
        "Translate {source_lang} into {target_lang}.\n"
        "Rules:\n"
        "- Output ONLY one single best translation, nothing else.\n"
        "- Never include alternatives, parenthetical options, annotations, or explanations.\n"
        "- Keep proper nouns, names, and brand names untranslated.\n"
        "- Use natural, casual, everyday language. Keep it conversational and concise.\n"
        "- Auto-correct likely ASR errors based on context and common sense."
    ),
    "esports": (
        "You are a real-time subtitle translator for esports/gaming live streams. "
        "Translate {source_lang} into {target_lang}.\n"
        "Rules:\n"
        "- Output ONLY one single best translation, nothing else.\n"
        "- Never include alternatives, parenthetical options, annotations, or explanations.\n"
        "- Keep player names (IGN), team names, game terms, and brand names untranslated.\n"
        "- Use energetic, concise language appropriate for competitive gaming commentary.\n"
        "- Auto-correct likely ASR errors based on context and common sense."
    ),
    "anime": (
        "You are a real-time subtitle translator for anime, movies, and TV shows. "
        "Translate {source_lang} into {target_lang}.\n"
        "Rules:\n"
        "- Output ONLY one single best translation, nothing else.\n"
        "- Never include alternatives, parenthetical options, annotations, or explanations.\n"
        "- Keep character names, place names, and cultural terms untranslated.\n"
        "- Use natural, expressive language that matches the tone and emotion of the dialogue.\n"
        "- Auto-correct likely ASR errors based on context and common sense."
    ),
    "webid": (
        "You are a real-time subtitle translator for an online identity-verification "
        "(WebID / video KYC) call. Translate {source_lang} into {target_lang}.\n"
        "Rules:\n"
        "- Output ONLY one single best translation, nothing else.\n"
        "- Never include alternatives, parenthetical options, annotations, or explanations.\n"
        "- Context: a verification agent and a customer on a video call inspect ID documents "
        "(passport, ID card). Words about reading or seeing refer to the document or the camera "
        "image, NOT literacy — e.g. 'I can't read it' means the text/photo is unclear, not that "
        "the person is illiterate.\n"
        "- Render camera/document instructions naturally (hold it up, tilt it, move closer, "
        "lighting, focus, read the number aloud, turn it over).\n"
        "- Keep names, document numbers, and verification codes exactly as spoken.\n"
        "- Auto-correct likely ASR errors based on this verification context."
    ),
}


def make_openai_client(
    api_base: str,
    api_key: str,
    proxy: str = "none",
    timeout: float | None = None,
) -> OpenAI:
    # Heterogeneous kwargs bag: the SDK declares httpx2 types for timeout/http_client,
    # but this project passes its httpx 0.28 client (the openai 3.x baseline shim).
    kwargs: dict[str, Any] = {"base_url": api_base, "api_key": api_key}
    if timeout is not None:
        kwargs["timeout"] = httpx.Timeout(timeout, connect=5.0)
    if proxy == "system":
        pass
    elif proxy in ("none", "", None):
        kwargs["http_client"] = httpx.Client(trust_env=False)
    else:
        kwargs["http_client"] = httpx.Client(proxy=proxy)
    return OpenAI(**kwargs)


class RepetitionError(Exception):
    """Raised when model output contains repetition loops."""

    pass


_OVERRIDE_KEYS = (
    "temperature",
    "top_p",
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
    "seed",
)


def _uses_deepseek_thinking_control(api_base: str | None, model: str | None) -> bool:
    """Whether this endpoint/model uses DeepSeek's nested thinking toggle."""
    endpoint = str(api_base or "").lower()
    model_id = str(model or "").lower()
    return "deepseek" in model_id or "api.deepseek.com" in endpoint


class Translator:
    """LLM-based translation using OpenAI-compatible API."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        target_language: str = "zh",
        max_tokens: int = 256,
        temperature: float = 0.3,
        streaming: bool = True,
        system_prompt: str | None = None,
        proxy: str = "none",
        no_system_role: bool = False,
        no_think: bool = False,
        json_response: bool = False,
        timeout: float = 10,
        overrides: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        json_schema_mode: bool = False,
    ) -> None:
        self._client: OpenAI = make_openai_client(api_base, api_key, proxy, timeout=timeout)
        self._proxy: str = proxy
        self._no_system_role: bool = no_system_role
        self._no_think: bool = no_think
        self._uses_deepseek_thinking: bool = _uses_deepseek_thinking_control(api_base, model)
        self._json_response: bool = json_response
        # CORE-8: most OpenAI-compatible endpoints (DeepSeek, Qwen) only
        # support {"type": "json_object"}; the strict json_schema variant is
        # an explicit per-model opt-in (kept on by migration for models that
        # had json_response enabled before this switch existed).
        self._json_schema_mode: bool = bool(json_schema_mode)
        if no_think:
            thinking_param = (
                "thinking.type=disabled"
                if self._uses_deepseek_thinking
                else "enable_thinking=false"
            )
            log.info(f"Translator: no_think enabled for {model} via {thinking_param}")
        if json_response:
            log.info(f"Translator: json_response enabled for {model}")
        self._model: str = model
        self._target_language: str = target_language
        self._max_tokens: int = max_tokens
        self._temperature: float = temperature
        self._streaming: bool = streaming
        self._timeout: float = timeout
        self._overrides: dict[str, Any] = {
            k: v for k, v in (overrides or {}).items() if v is not None
        }
        self._extra_body: dict[str, Any] = dict(extra_body) if extra_body else {}
        if self._overrides:
            log.info(f"Translator overrides: {self._overrides}")
        if self._extra_body:
            log.info(f"Translator extra_body: {self._extra_body}")
        self._system_prompt_template: str = system_prompt or DEFAULT_PROMPT
        self._context_turns: int = 0
        # CORE-7: _history is mutated/read from the 8-worker translation
        # pool — all access goes through the lock.
        self._history_lock = threading.Lock()
        self._history: list[tuple[str, str]] = []  # (source_text, translated_text)
        self._last_prompt_tokens: int = 0
        self._last_completion_tokens: int = 0

    @property
    def last_usage(self) -> tuple[int, int]:
        """(prompt_tokens, completion_tokens) from last translate call."""
        return self._last_prompt_tokens, self._last_completion_tokens

    @property
    def proxy(self) -> str:
        return self._proxy

    def set_target_language(self, target_language: str) -> None:
        self._target_language = target_language

    def set_timeout(self, timeout: float) -> None:
        self._timeout = timeout
        self._client = self._client.copy(timeout=timeout)

    def set_context_turns(self, n: int) -> None:
        self._context_turns = n
        if n == 0:
            with self._history_lock:
                self._history.clear()

    def clear_history(self) -> None:
        with self._history_lock:
            self._history.clear()

    def _format_context(self) -> str:
        if self._context_turns <= 0:
            return ""
        with self._history_lock:
            if not self._history:
                return ""
            history = list(self._history[-self._context_turns :])
        lines: list[str] = []
        for src, tgt in history:
            lines.append(f"Source: {src}")
            lines.append(f"Translation: {tgt}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def with_target_language(self, target_language: str) -> "Translator":
        """Create a new Translator with a different target language, sharing the same client."""
        t = Translator.__new__(Translator)
        t._client = self._client
        t._proxy = self._proxy
        t._no_system_role = self._no_system_role
        t._no_think = self._no_think
        t._uses_deepseek_thinking = self._uses_deepseek_thinking
        t._json_response = self._json_response
        t._json_schema_mode = self._json_schema_mode
        t._model = self._model
        t._target_language = target_language
        t._max_tokens = self._max_tokens
        t._temperature = self._temperature
        t._streaming = self._streaming
        t._timeout = self._timeout
        t._overrides = dict(self._overrides)
        t._extra_body = dict(self._extra_body)
        t._system_prompt_template = self._system_prompt_template
        t._context_turns = 0
        t._history_lock = threading.Lock()
        t._history = []
        t._last_prompt_tokens = 0
        t._last_completion_tokens = 0
        return t

    def _build_system_prompt(self, source_lang: str) -> str:
        src = LANGUAGE_DISPLAY.get(source_lang, source_lang)
        tgt = LANGUAGE_DISPLAY.get(self._target_language, self._target_language)
        try:
            prompt = self._system_prompt_template.format(
                source_lang=src,
                target_lang=tgt,
                context=self._format_context(),
            )
        except (KeyError, IndexError, ValueError) as e:
            log.warning(f"Bad prompt template, falling back to default: {e}")
            prompt = DEFAULT_PROMPT.format(source_lang=src, target_lang=tgt)
        if self._json_response:
            prompt += '\nRespond in JSON format: {"t": "translated text"}'
        return prompt

    def _build_messages(self, system_prompt: str, text: str) -> list[ChatCompletionMessageParam]:
        msgs: list[ChatCompletionMessageParam]
        if self._no_system_role:
            msgs = [{"role": "user", "content": f"{system_prompt}\n{text}"}]
        else:
            msgs = [{"role": "system", "content": system_prompt}]
            # Append recent history as context
            if self._context_turns > 0 and "{context}" not in self._system_prompt_template:
                with self._history_lock:
                    history = list(self._history[-self._context_turns :])
                for src, tgt in history:
                    msgs.append({"role": "user", "content": src})
                    msgs.append({"role": "assistant", "content": tgt})
            msgs.append({"role": "user", "content": text})
        return msgs

    def _append_history(self, text: str, result: str) -> None:
        if self._context_turns > 0 and result:
            with self._history_lock:
                self._history.append((text, result))
                max_keep = self._context_turns + 2
                if len(self._history) > max_keep:
                    self._history = self._history[-self._context_turns :]

    def _build_request_kwargs(
        self, system_prompt: str, text: str, stream: bool = False
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._build_messages(system_prompt, text),
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        for k in _OVERRIDE_KEYS:
            if k in self._overrides:
                kwargs[k] = self._overrides[k]
        extra_body: dict[str, Any] = {}
        if self._no_think:
            if self._uses_deepseek_thinking:
                extra_body["thinking"] = {"type": "disabled"}
            else:
                extra_body["enable_thinking"] = False
        if self._extra_body:
            extra_body.update(self._extra_body)
        if extra_body:
            kwargs["extra_body"] = extra_body
        if self._json_response:
            # CORE-8: json_object is the portable mode (DeepSeek/Qwen
            # compatible); the strict json_schema variant requires explicit
            # opt-in via json_schema_mode.
            if self._json_schema_mode:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "translation",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"t": {"type": "string"}},
                            "required": ["t"],
                            "additionalProperties": False,
                        },
                    },
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}
        if stream:
            kwargs["stream"] = True
        return kwargs

    def translate(self, text: str, source_language: str = "en") -> str:
        system_prompt = self._build_system_prompt(source_language)
        if self._streaming:
            result = self._translate_streaming(system_prompt, text)
        else:
            result = self._translate_sync(system_prompt, text)
        if self._check_repetition(result):
            raise RepetitionError(result)
        self._append_history(text, result)
        return result

    def translate_iter(self, text: str, source_language: str = "en") -> Iterator[str]:
        """Generator that yields accumulated partial text, then final result.

        Non-streaming or json_response mode: yields once with the final result.
        Streaming mode: yields partial accumulated text as chunks arrive.
        The final yielded value is always the complete translation.
        Caller should use the last yielded value as the final result.
        """
        system_prompt = self._build_system_prompt(source_language)
        if not self._streaming:
            result = self._translate_sync(system_prompt, text)
            self._append_history(text, result)
            yield result
            return

        # Streaming path
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        base_kwargs = self._build_request_kwargs(system_prompt, text, stream=True)
        try:
            stream = self._client.chat.completions.create(
                **base_kwargs,
                stream_options={"include_usage": True},
            )
        except BadRequestError as e:
            # CORE-6: retry once WITHOUT stream_options, and only for a 400 —
            # some endpoints reject the usage-option outright. 401/429/network
            # errors must propagate (retrying them doubles the request load
            # on a rate-limited endpoint and swallows the real cause).
            log.info(f"Endpoint rejected stream_options, retrying without: {str(e)[:120]}")
            stream = self._client.chat.completions.create(**base_kwargs)

        deadline = time.monotonic() + self._timeout
        chunks: list[str] = []
        for chunk in stream:
            if time.monotonic() > deadline:
                stream.close()
                raise TimeoutError(f"Translation exceeded {self._timeout}s total timeout")
            if hasattr(chunk, "usage") and chunk.usage:
                self._last_prompt_tokens = chunk.usage.prompt_tokens or 0
                self._last_completion_tokens = chunk.usage.completion_tokens or 0
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
                    if not self._json_response:
                        yield "".join(chunks)
        result = "".join(chunks).strip()
        if self._json_response:
            result = self._extract_json_translation(result)
        if self._check_repetition(result):
            raise RepetitionError(result)
        self._append_history(text, result)
        yield result

    def _extract_json_translation(self, raw: str) -> str:
        """Extract translation from JSON response, fallback to raw text."""
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "t" in data:
                return cast("str", data["t"])  # schema guarantees {"t": "<str>"}
        except (json.JSONDecodeError, TypeError):
            pass
        return raw

    @staticmethod
    def _check_repetition(text: str) -> bool:
        """Detect repetition loops in model output."""
        if not text or len(text) < 40:
            return False
        return any(text[plen : plen * 2] == text[:plen] for plen in range(8, len(text) // 2 + 1))

    def _translate_sync(self, system_prompt: str, text: str) -> str:
        kwargs = self._build_request_kwargs(system_prompt, text, stream=False)
        resp = self._client.chat.completions.create(**kwargs)
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        if resp.usage:
            self._last_prompt_tokens = resp.usage.prompt_tokens or 0
            self._last_completion_tokens = resp.usage.completion_tokens or 0
        result = cast("str", resp.choices[0].message.content).strip()
        if self._json_response:
            result = self._extract_json_translation(result)
        return result

    def _translate_streaming(self, system_prompt: str, text: str) -> str:
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        base_kwargs = self._build_request_kwargs(system_prompt, text, stream=True)
        try:
            stream = self._client.chat.completions.create(
                **base_kwargs,
                stream_options={"include_usage": True},
            )
        except BadRequestError as e:
            # CORE-6: see _translate_iter — narrow stream_options fallback.
            log.info(f"Endpoint rejected stream_options, retrying without: {str(e)[:120]}")
            stream = self._client.chat.completions.create(**base_kwargs)

        deadline = time.monotonic() + self._timeout
        chunks: list[str] = []
        for chunk in stream:
            if time.monotonic() > deadline:
                stream.close()
                raise TimeoutError(f"Translation exceeded {self._timeout}s total timeout")
            if hasattr(chunk, "usage") and chunk.usage:
                self._last_prompt_tokens = chunk.usage.prompt_tokens or 0
                self._last_completion_tokens = chunk.usage.completion_tokens or 0
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
        result = "".join(chunks).strip()
        if self._json_response:
            result = self._extract_json_translation(result)
        return result
