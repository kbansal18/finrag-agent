"""Smoke test: the Streamlit script runs top to bottom without raising.

Skipped where the data has not been built (e.g. CI), since the app needs the index.
"""

import pytest

from finrag.config import PROJECT_ROOT, get_settings

pytest.importorskip("streamlit")
needs_data = pytest.mark.skipif(
    not (get_settings().data_dir / "index" / "section" / "meta.json").exists(),
    reason="search index not built",
)


@needs_data
def test_app_renders_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Ask", "Evaluation results", "How it works"]
