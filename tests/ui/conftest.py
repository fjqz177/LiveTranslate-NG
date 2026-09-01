"""UI test fixtures.

ControlPanel applies its chrome at the application level (popups and
dialogs resolve palettes from the app palette), so every test that builds
a panel mutates the shared session-scoped qapp. This autouse fixture
snapshots and restores the app palette + stylesheet around each test so
the theme never leaks across tests or files.
"""

import pytest


@pytest.fixture(autouse=True)
def _app_theme_isolation(qapp):
    saved_palette = qapp.palette()
    saved_stylesheet = qapp.styleSheet()
    yield
    qapp.setPalette(saved_palette)
    qapp.setStyleSheet(saved_stylesheet)
