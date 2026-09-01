"""End-to-end parity: SenseVoice legacy (funasr) vs ONNX engine.

Gated behind the engine marker: requires engine-funasr, the exported
ONNX model and the bundled example audio. Features are computed ONCE
(deterministic, dither=0) and fed to both paths — the legacy path takes
precomputed fbank+LFR features via data_type="fbank" — so the comparison
isolates the neural part (encoder + CTC + decode).
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")

from livetranslate.core.paths import models_dir  # noqa: E402

pytestmark = pytest.mark.engine

_EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "modelscope"
    / "models"
    / "iic--SenseVoiceSmall"
    / "snapshots"
    / "master"
    / "example"
    / "zh.mp3"
)
_ONNX_MODEL = models_dir() / "sensevoice" / "sensevoice-small.onnx"


def _load_audio() -> np.ndarray:
    wav, sr = sf.read(_EXAMPLE, dtype="float32")
    audio = wav.mean(axis=1) if wav.ndim > 1 else wav
    t = torch.from_numpy(audio.astype("float32"))
    if sr != 16000:
        import torchaudio.functional as F

        t = F.resample(t, sr, 16000)
    return t.numpy()[: 16000 * 6]


@pytest.mark.skipif(not _EXAMPLE.exists(), reason="bundled example audio missing")
@pytest.mark.skipif(not _ONNX_MODEL.exists(), reason="onnx model missing (run the export script)")
def test_onnx_matches_legacy_text():
    from funasr import AutoModel

    from livetranslate.asr.engines.sensevoice_onnx import (
        SenseVoiceOnnxEngine,
        _apply_lfr,
        _fbank,
    )

    audio = _load_audio()
    # Deterministic features shared by BOTH paths.
    speech = _apply_lfr(_fbank(audio.astype("float32")))[None].astype("float32")

    legacy = AutoModel(model="iic/SenseVoiceSmall", hub="ms", device="cpu", disable_update=True)
    result = legacy.generate(
        input=torch.from_numpy(speech),
        input_len=torch.tensor([speech.shape[1]]),
        data_type="fbank",
        cache={},
        language="auto",
        # withitn conditioning matches the ONNX export; ITN postprocessing
        # stays off so the comparison isolates the neural path.
        use_itn=False,
        text_norm="withitn",
        batch_size_s=0,
        disable_pbar=True,
    )
    legacy_text = result[0]["text"]

    onnx = SenseVoiceOnnxEngine(pad_seconds=0)
    logits = onnx._session.run(None, {"speech": speech})[0]
    from livetranslate.asr.engines.sensevoice_onnx import _decode

    onnx_text, onnx_lang = _decode(logits, onnx._tokenizer)

    import re

    legacy_clean = re.sub(r"<\|[^|]+\|>", "", legacy_text).strip()
    assert onnx_text == legacy_clean
    assert onnx_lang == "zh"
