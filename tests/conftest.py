import os
from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_feed_xml(fixtures_dir):
    return (fixtures_dir / "sample_feed.xml").read_text()


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provides a temporary data directory for tests."""
    for sub in ("episodes", "transcripts", "interjections", "output"):
        (tmp_path / sub).mkdir()
    return tmp_path


@pytest.fixture
def gemini_api_key():
    """Provide an LLM judge API key, preferring OpenRouter over direct Gemini.

    Returns the key string. The backend is available via judge_backend fixture.
    """
    key = os.environ.get("LWL_OPENROUTER_API_KEY", "")
    if key:
        return key
    key = os.environ.get("LWL_GEMINI_API_KEY", "")
    if key:
        return key
    pytest.skip("Neither LWL_OPENROUTER_API_KEY nor LWL_GEMINI_API_KEY is set")


@pytest.fixture
def judge_backend():
    """Return which backend to use for the LLM judge."""
    if os.environ.get("LWL_OPENROUTER_API_KEY", ""):
        return "openrouter"
    return "gemini"
