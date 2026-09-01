"""About tab (关于): version, update check, project links, license
and the changelog (plan §3.2 item 7).
"""

from __future__ import annotations

import threading

from PyQt6.QtCore import Qt, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from livetranslate.core.i18n import t
from livetranslate.core.version import app_version
from livetranslate.ui.dialogs import _load_latest_changelog
from livetranslate.ui.panel._tab_base import TabBase

PROJECT_URL = "https://github.com/fjqz177/LiveTranslate"
ISSUES_URL = f"{PROJECT_URL}/issues"


class AboutTab(TabBase):
    # kind, payload (new_version for "new", detail str otherwise)
    update_result = pyqtSignal(str, object)

    def __init__(self, panel):
        super().__init__(panel)
        layout = QVBoxLayout(self)

        name = QLabel(f"LiveTranslate  ·  {t('about_version').format(version=app_version())}")
        name.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        layout.addWidget(name)
        desc = QLabel(t("about_desc"))
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # -- Update --
        update_group = QGroupBox(t("group_update"))
        update_layout = QHBoxLayout(update_group)
        self._check_btn = QPushButton(t("btn_check_update"))
        self._check_btn.clicked.connect(self._check_update)
        self.update_result.connect(self._on_update_result)
        update_layout.addWidget(self._check_btn)
        update_layout.addStretch(1)
        layout.addWidget(update_group)

        # -- Links --
        links_group = QGroupBox(t("group_links"))
        links_layout = QHBoxLayout(links_group)
        repo_btn = QPushButton(t("btn_open_repo"))
        repo_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(PROJECT_URL)))
        links_layout.addWidget(repo_btn)
        issues_btn = QPushButton(t("btn_open_issues"))
        issues_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(ISSUES_URL)))
        links_layout.addWidget(issues_btn)
        links_layout.addStretch(1)
        layout.addWidget(links_group)

        # -- License & privacy --
        license_group = QGroupBox(t("group_license"))
        license_layout = QVBoxLayout(license_group)
        license_label = QLabel(t("about_license_text"))
        license_label.setWordWrap(True)
        license_label.setTextInteractionFlags(
            license_label.textInteractionFlags() | Qt.TextInteractionFlag.TextSelectableByMouse
        )
        license_layout.addWidget(license_label)
        layout.addWidget(license_group)

        # -- Changelog --
        cl_group = QGroupBox(t("group_changelog"))
        cl_layout = QVBoxLayout(cl_group)
        _, html = _load_latest_changelog()
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(html)
        cl_layout.addWidget(browser)
        layout.addWidget(cl_group, 1)

    def collect(self):
        # The update group no longer persists any settings (channel is gone).
        return

    def _check_update(self):
        """Ask GitHub for the latest release and compare — off the UI thread.

        The self-hosted Ed25519 update bridge is gone: we read the version
        truth directly from GitHub's releases API and hand the user a link.
        """
        from livetranslate.core.updater import check_latest_release

        self._check_btn.setEnabled(False)
        self._check_btn.setText(t("checking_update"))

        current = app_version()

        def _run() -> None:
            result = check_latest_release(current)
            payload: object = result.new_version if result.kind == "new" else result.detail
            self.update_result.emit(result.kind, payload)

        threading.Thread(target=_run, daemon=True).start()

    @pyqtSlot(str, object)
    def _on_update_result(self, kind: str, payload: object) -> None:
        self._check_btn.setEnabled(True)
        self._check_btn.setText(t("btn_check_update"))
        if kind == "new":
            from livetranslate.core.updater import RELEASE_PAGE_URL

            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Information)
            box.setWindowTitle(t("update_available_title"))
            box.setText(t("update_available_msg").format(version=str(payload)))
            open_btn = box.addButton(t("update_open_download"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Close)
            box.exec()
            if box.clickedButton() is open_btn:
                QDesktopServices.openUrl(QUrl(RELEASE_PAGE_URL))
        elif kind == "uptodate":
            QMessageBox.information(
                self,
                t("btn_check_update"),
                t("update_uptodate").format(version=app_version()),
            )
        elif kind == "none":
            QMessageBox.information(
                self,
                t("btn_check_update"),
                t("update_not_available").format(version=app_version()),
            )
        else:
            QMessageBox.warning(
                self,
                t("btn_check_update"),
                t("update_check_failed").format(detail=str(payload)),
            )
