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
