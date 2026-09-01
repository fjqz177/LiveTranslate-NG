from typing import ClassVar

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from livetranslate.core.i18n import t


class ErrorBanner(QWidget):
    """Slim warning bar under the header (plan §3.5.6 错误态 + §3.6 copy).

    Shows a classified translate error; clicking the diagnostics button
    routes the user to the network card, ✕ dismisses (auto-hides in 8s).
    """

    diagnostics_clicked = pyqtSignal()
    dismissed = pyqtSignal()

    # error: amber tint (§3.5.2 warning); info: neutral blue tint
    _STYLES: ClassVar[dict[str, str]] = {
        "error": (
            "QWidget#errorBanner { background: rgba(90,70,20,220); border-radius: 4px; }"
            "QLabel { background: transparent; }"
        ),
        "info": (
            "QWidget#errorBanner { background: rgba(30,60,90,220); border-radius: 4px; }"
            "QLabel { background: transparent; }"
        ),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("errorBanner")
        self.setStyleSheet(self._STYLES["error"])
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 6, 4)
        layout.setSpacing(6)

        icon = QLabel("⚠")
        icon.setStyleSheet("color: #D29922; font-weight: 600;")
        layout.addWidget(icon)

        self._label = QLabel("")
        self._label.setWordWrap(True)
        self._label.setStyleSheet("color: #F2F4F7;")
        layout.addWidget(self._label, 1)

        self._diag_btn = QPushButton(t("banner_open_diagnostics"))
        self._diag_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._diag_btn.setStyleSheet(
            "QPushButton { background: transparent; border: 1px solid rgba(210,153,34,120);"
            " border-radius: 4px; color: #D29922; padding: 2px 8px; font-size: 11px; }"
            "QPushButton:hover { background: rgba(210,153,34,40); }"
        )
        self._diag_btn.clicked.connect(self.diagnostics_clicked.emit)
        layout.addWidget(self._diag_btn)

        dismiss = QPushButton("✕")
        dismiss.setToolTip(t("banner_dismiss"))
        dismiss.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        dismiss.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: #A9B2BF;"
            " padding: 0 4px; }"
            "QPushButton:hover { color: #F2F4F7; }"
        )
        dismiss.clicked.connect(self.dismissed.emit)
        layout.addWidget(dismiss)

        self._auto_hide = QTimer(self)
        self._auto_hide.setSingleShot(True)
        self._auto_hide.setInterval(8000)
        self._auto_hide.timeout.connect(self.hide)

        self.hide()

    def show_text(self, text: str, kind: str = "error") -> None:
        self.setStyleSheet(self._STYLES.get(kind, self._STYLES["error"]))
        self._label.setText(text)
        self.show()
        self._auto_hide.start()
