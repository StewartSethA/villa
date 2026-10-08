import pytest


@pytest.fixture(autouse=True)
def _legacy_track_limit(monkeypatch):
    """Legacy tests were written against the 2^24-track cap (it shapes stripe heights and ordering).  They keep it via the documented override;
    tests that exercise the LIFTED limit (test_trackfix.py) pass their own environment explicitly."""
    monkeypatch.setenv("ROUTEB_KEEP_TRACK_LIMIT", "1")
