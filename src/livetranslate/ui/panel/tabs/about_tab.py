"""About tab (关于): version, project links, license and the changelog
(plan §3.2 item 7).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
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
    def __init__(self, panel):
        super().__init__(panel)
        layout = QVBoxLayout(self)

        name = QLabel(f"LiveTranslate  ·  {t('about_version').format(version=app_version())}")
        name.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        layout.addWidget(name)
        desc = QLabel(t("about_desc"))
        desc.setWordWrap(True)
        layout.addWidget(desc)

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
