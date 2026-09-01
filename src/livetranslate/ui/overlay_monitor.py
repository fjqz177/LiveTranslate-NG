import os

import psutil
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from livetranslate.core.theme import (
    META_COLOR_ASR,
    META_COLOR_SEPARATOR,
    META_COLOR_SOURCE_LANG,
    META_COLOR_TL,
)

_BAR_CSS_TPL = """
    QProgressBar {{
        background: rgba(255,255,255,15);
        border: 1px solid rgba(255,255,255,30);
        border-radius: 3px;
        text-align: center;
        font-size: 8pt;
        color: #aaa;
    }}
    QProgressBar::chunk {{
        background: {color};
        border-radius: 2px;
    }}
"""


class MonitorBar(QWidget):
    """Compact system monitor displayed in the overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        row1 = QHBoxLayout()
        row1.setSpacing(6)

        # MIC bar (hidden when mic is disabled)
        self._mic_lbl = QLabel("MIC")
        self._mic_lbl.setFixedWidth(26)
        self._mic_lbl.setFont(QFont("Consolas", 8))
        self._mic_lbl.setStyleSheet("color: #888; background: transparent;")
        self._mic_lbl.setVisible(False)
        row1.addWidget(self._mic_lbl)

        self._mic_bar = QProgressBar()
        self._mic_bar.setRange(0, 100)
        self._mic_bar.setFixedHeight(14)
        self._mic_bar.setTextVisible(True)
        self._mic_bar.setFormat("%v%")
        self._mic_bar.setStyleSheet(_BAR_CSS_TPL.format(color="#c586c0"))
        self._mic_bar.setVisible(False)
        row1.addWidget(self._mic_bar)

        rms_lbl = QLabel("RMS:")
        rms_lbl.setFixedWidth(26)
        rms_lbl.setFont(QFont("Consolas", 8))
        rms_lbl.setStyleSheet("color: #888; background: transparent;")
        row1.addWidget(rms_lbl)

        self._rms_bar = QProgressBar()
        self._rms_bar.setRange(0, 100)
        self._rms_bar.setFixedHeight(14)
        self._rms_bar.setTextVisible(True)
        self._rms_bar.setFormat("%v%")
        self._rms_bar.setStyleSheet(_BAR_CSS_TPL.format(color="#4ec9b0"))
        row1.addWidget(self._rms_bar)

        vad_lbl = QLabel("VAD:")
        vad_lbl.setFixedWidth(26)
        vad_lbl.setFont(QFont("Consolas", 8))
        vad_lbl.setStyleSheet("color: #888; background: transparent;")
        row1.addWidget(vad_lbl)

        self._vad_bar = QProgressBar()
        self._vad_bar.setRange(0, 100)
        self._vad_bar.setFixedHeight(14)
        self._vad_bar.setTextVisible(True)
        self._vad_bar.setFormat("%v%")
        self._vad_bar.setStyleSheet(_BAR_CSS_TPL.format(color="#dcdcaa"))
        row1.addWidget(self._vad_bar)

        layout.addLayout(row1)

        # Cost is shown in its own label so long counters can never push it
        # out of view; the stats label must not drive the window width.
        self._cost_label = QLabel()
        self._cost_label.setFont(QFont("Consolas", 8))
        self._cost_label.setStyleSheet("color: #fa5; background: transparent;")
        self._cost_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(self._cost_label, 0, Qt.AlignmentFlag.AlignRight)

        self._stats_label = QLabel()
        self._stats_label.setFont(QFont("Consolas", 8))
        self._stats_label.setStyleSheet("color: #888; background: transparent;")
        self._stats_label.setTextFormat(Qt.TextFormat.RichText)
        self._stats_label.setWordWrap(True)
        # Ignored horizontally: the label wraps instead of widening the window.
        self._stats_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(self._stats_label)

        self._proc = psutil.Process(os.getpid())
        self._proc.cpu_percent(interval=None)  # Prime the counter
        self._cpu = 0
        self._ram_mb = 0.0
        self._gpu_text = "N/A"
        self._asr_device = ""
        self._asr_count = 0
        self._tl_count = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost = 0.0
        self._last_stats_text = None
        self._last_cost_text = None

        self._sys_timer = QTimer(self)
        self._sys_timer.timeout.connect(self._update_system)
        self._sys_timer.start(1000)
        self._update_system()
        self._refresh_stats()

    def update_audio(self, rms: float, vad: float, mic_rms=None):
        self._rms_bar.setValue(min(100, int(rms * 500)))
        self._vad_bar.setValue(min(100, int(vad * 100)))
        mic_active = mic_rms is not None
        if self._mic_lbl.isVisible() != mic_active:
            self._mic_lbl.setVisible(mic_active)
            self._mic_bar.setVisible(mic_active)
        if mic_active:
            self._mic_bar.setValue(min(100, int(mic_rms * 500)))

    def update_asr_device(self, device: str):
        self._asr_device = device
        self._refresh_stats()

    def update_pipeline_stats(
        self, asr_count, tl_count, prompt_tokens, completion_tokens, cost=0.0
    ):
        self._asr_count = asr_count
        self._tl_count = tl_count
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._cost = cost
        self._refresh_stats()

    def _update_system(self):
        try:
            self._cpu = int(self._proc.cpu_percent(interval=None) / os.cpu_count())
            self._ram_mb = self._proc.memory_info().rss / 1024 / 1024
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024 / 1024
                self._gpu_text = f"{alloc:.0f}MB"
            else:
                self._gpu_text = "N/A"
        except Exception:
            self._gpu_text = "N/A"
        self._refresh_stats()

    @staticmethod
    def _fmt_count(value: int) -> str:
        """Compact token count: 1234 -> 1.2k, 1_500_000 -> 1.5M."""
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}k"
        return str(int(value))

    @staticmethod
    def _fmt_cost(value: float) -> str:
        """Cost with adaptive precision so it stays short (<= 8 chars)."""
        if value >= 10_000:
            return f"{value:.0f}"
        if value >= 1_000:
            return f"{value:.1f}"
        if value >= 100:
            return f"{value:.2f}"
        if value >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"

    def _refresh_stats(self):
        total = self._prompt_tokens + self._completion_tokens
        tokens_str = self._fmt_count(total)
        dev_str = ""
        if self._asr_device:
            dev_color = "#4ec9b0" if "cuda" in self._asr_device.lower() else "#dcdcaa"
            dev_str = (
                f'<span style="color:{dev_color};">{self._asr_device}</span> '
                f'<span style="color:#555;">|</span> '
            )
        # Cost label: always rendered, even at 0, so its space is reserved and
        # it can never be pushed out by growing counters.
        from livetranslate.core.i18n import get_lang

        symbol = "¥" if get_lang() == "zh" else "$"
        cost_text = f"{symbol}{self._fmt_cost(self._cost)}" if self._cost > 0 else f"{symbol}0.0000"
        if cost_text != self._last_cost_text:
            self._last_cost_text = cost_text
            self._cost_label.setText(cost_text)
            self._cost_label.setToolTip(f"{symbol}{self._cost:.6f}")
        stats_text = (
            f"{dev_str}"
            f'<span style="color:{META_COLOR_SOURCE_LANG};">CPU</span> {self._cpu}% '
            f'<span style="color:{META_COLOR_SOURCE_LANG};">RAM</span> {self._ram_mb:.0f}MB '
            f'<span style="color:{META_COLOR_SOURCE_LANG};">GPU</span> {self._gpu_text} '
            f'<span style="color:{META_COLOR_SEPARATOR};">|</span> '
            f'<span style="color:{META_COLOR_ASR};">ASR</span> {self._asr_count} '
            f'<span style="color:{META_COLOR_TL};">TL</span> {self._tl_count} '
            f'<span style="color:#c9c;">Tok</span> {tokens_str} '
            f'<span style="color:#666;">({self._fmt_count(self._prompt_tokens)}'
            f"\u2191{self._fmt_count(self._completion_tokens)}\u2193)</span>"
        )
        # Skip the per-second relayout when nothing changed
        if stats_text != self._last_stats_text:
            self._last_stats_text = stats_text
            self._stats_label.setText(stats_text)
