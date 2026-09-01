"""Export the local FunASR SenseVoiceSmall model to ONNX for the
torch-free sensevoice-onnx engine.

Usage (dev machine with engine-funasr installed and the model cached):

    uv run python scripts/export_sensevoice_onnx.py

The script loads the funasr SenseVoiceSmall checkpoint, wraps the encoder
+ CTC head into a traceable module and emits:

    <models_dir>/sensevoice/sensevoice-small.onnx

ONNX contract (inputs/outputs the engine consumes):
    inputs:  speech (B, T, 560) float32 fbank+context features
    outputs: logits (B, T, vocab) float32 - argmax = token ids

The engine reproduces the frontend (fbank + cmvn + 7-frame context) and
the SentencePiece decode; this script only exports the neural part.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _EncoderCTC(torch.nn.Module):
    """Traceable wrapper: encoder -> ctc linear -> log_softmax."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, speech):
        """speech: (B, T, 560). Returns logits (B, T, vocab)."""
        batch = speech.shape[0]
        lengths = torch.full((batch,), speech.shape[1], dtype=torch.int64)
        # Text prefix tokens: [sos, lid(auto), eos, textnorm(withitn)]
        sos, eos = self.model.sos, self.model.eos
        lid_auto = next(iter(self.model.lid_int_dict))
        withitn = next(iter(self.model.textnorm_int_dict))
        text = torch.tensor([[sos, lid_auto, eos, withitn]], dtype=torch.int64).repeat(batch, 1)
        encoder_out, _encoder_out_lens = self.model.encode(speech, lengths, text)
        # CTC over the FULL encoder output: the first 4 frames carry the
        # language / emotion / textnorm special tokens (<|zh|>...), exactly
        # like the legacy inference path.
        return self.model.ctc.log_softmax(encoder_out)


def main() -> None:
    from funasr import AutoModel

    # funasr 1.4.x cannot load a local directory path ("model ... is not
    # registered"); the registered ModelScope id resolves through the local
    # cache (MODELSCOPE_CACHE), which is exactly what the app sets.
    print("Loading funasr model via registered id iic/SenseVoiceSmall")
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        device="cpu",
        hub="ms",
        disable_update=True,
    )
    inner: torch.nn.Module = model.model
    inner.eval()

    wrapper = _EncoderCTC(inner)
    dummy = torch.zeros(1, 60, 560, dtype=torch.float32)
    from livetranslate.core.paths import models_dir

    out_dir = models_dir() / "sensevoice"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sensevoice-small.onnx"

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            input_names=["speech"],
            output_names=["logits"],
            dynamic_axes={"speech": {0: "batch", 1: "time"}, "logits": {0: "batch", 1: "time"}},
            opset_version=17,
            # Legacy TorchScript exporter: the dynamo path pulls onnxscript
            # (unneeded weight for an export-time tool).
            dynamo=False,
        )
    print(f"Exported: {out_path}")


if __name__ == "__main__":
    main()
