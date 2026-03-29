from datetime import datetime, timezone
from unittest.mock import patch
from xml.etree import ElementTree

import pytest
from fastapi.testclient import TestClient

from lex_without_lex.models import Episode, EpisodeState
from lex_without_lex.server import app, render_feed_xml
from lex_without_lex.config import Settings


@pytest.fixture
def client():
    # Disable lifespan to avoid background task in tests
    app.router.lifespan_context = None
    return TestClient(app)


@pytest.fixture
def sample_episodes():
    return [
        EpisodeState(
            episode=Episode(
                guid="ep-1",
                title="Guest Talk: AI Future",
                published=datetime(2025, 3, 10, tzinfo=timezone.utc),
                audio_url="https://example.com/ep1.mp3",
                duration_seconds=3600,
                description="A deep conversation about AI.",
            ),
            b2_url="https://b2.example.com/episodes/ep-1.mp3",
            status="uploaded",
        ),
        EpisodeState(
            episode=Episode(
                guid="ep-2",
                title="Guest Talk: Robotics",
                published=datetime(2025, 3, 5, tzinfo=timezone.utc),
                audio_url="https://example.com/ep2.mp3",
                duration_seconds=7200,
                description="Discussion on robotics.",
            ),
            b2_url="https://b2.example.com/episodes/ep-2.mp3",
            status="uploaded",
        ),
    ]


class TestHealthEndpoint:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestFeedXmlEndpoint:
    def test_returns_xml(self, client, tmp_path):
        # Create empty state file
        state_file = tmp_path / "state.json"
        state_file.write_text("{}")

        with patch("lex_without_lex.server.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            mock_settings.base_url = "https://test.example.com"
            resp = client.get("/feed.xml")

        assert resp.status_code == 200
        assert "application/xml" in resp.headers["content-type"]

    def test_empty_feed_is_valid_xml(self, client, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("{}")

        with patch("lex_without_lex.server.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            mock_settings.base_url = "https://test.example.com"
            resp = client.get("/feed.xml")

        root = ElementTree.fromstring(resp.text)
        assert root.tag == "rss"


class TestRenderFeedXml:
    def test_contains_episodes(self, sample_episodes):
        settings = Settings(base_url="https://test.example.com")
        xml = render_feed_xml(sample_episodes, settings)
        root = ElementTree.fromstring(xml)

        items = root.findall(".//item")
        assert len(items) == 2

    def test_episode_has_enclosure(self, sample_episodes):
        settings = Settings(base_url="https://test.example.com")
        xml = render_feed_xml(sample_episodes, settings)
        root = ElementTree.fromstring(xml)

        enclosures = root.findall(".//enclosure")
        assert len(enclosures) == 2
        assert "b2.example.com" in enclosures[0].get("url", "")

    def test_empty_episodes(self):
        settings = Settings(base_url="https://test.example.com")
        xml = render_feed_xml([], settings)
        root = ElementTree.fromstring(xml)
        items = root.findall(".//item")
        assert len(items) == 0

    def test_feed_title(self, sample_episodes):
        settings = Settings(base_url="https://test.example.com")
        xml = render_feed_xml(sample_episodes, settings)
        root = ElementTree.fromstring(xml)
        title = root.find(".//channel/title")
        assert title is not None
        assert "Lex Without Lex" in title.text
