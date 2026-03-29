from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
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


def _make_episodes():
    """Helper to create test episodes with varying dates."""
    return [
        Episode(
            guid="ep-new",
            title="New Episode",
            published=datetime(2026, 3, 25, tzinfo=timezone.utc),
            audio_url="https://example.com/new.mp3",
            duration_seconds=3600,
            description="A new episode.",
        ),
        Episode(
            guid="ep-old-1",
            title="Old Episode One",
            published=datetime(2026, 1, 10, tzinfo=timezone.utc),
            audio_url="https://example.com/old1.mp3",
            duration_seconds=7200,
            description="An old episode.",
        ),
        Episode(
            guid="ep-old-2",
            title="Old Episode Two",
            published=datetime(2025, 6, 5, tzinfo=timezone.utc),
            audio_url="https://example.com/old2.mp3",
            duration_seconds=5400,
            description="Another old episode.",
        ),
    ]


class TestTriggerProcessing:
    def test_returns_202(self, client):
        with patch("lex_without_lex.server.process_new_episodes", new_callable=AsyncMock):
            resp = client.post("/process")
        assert resp.status_code == 202
        assert resp.json() == {"status": "processing"}

    def test_calls_process_new_episodes(self, client):
        with patch("lex_without_lex.server.process_new_episodes", new_callable=AsyncMock) as mock_proc:
            client.post("/process")
        mock_proc.assert_called_once()


class TestListEpisodes:
    def test_returns_episodes_with_status(self, client, tmp_path):
        episodes = _make_episodes()
        state = {
            "ep-new": EpisodeState(episode=episodes[0], status="uploaded"),
        }

        with (
            patch("lex_without_lex.server.get_episodes", new_callable=AsyncMock, return_value=episodes),
            patch("lex_without_lex.server.settings") as mock_settings,
        ):
            mock_settings.feed_url = "https://example.com/feed"
            mock_settings.data_dir = tmp_path
            mock_settings.episodes_after = datetime(2026, 3, 20, tzinfo=timezone.utc)

            from lex_without_lex.pipeline import save_state
            save_state(state, tmp_path / "state.json")

            resp = client.get("/episodes")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["episodes"]) == 3

        by_guid = {ep["guid"]: ep for ep in data["episodes"]}
        assert by_guid["ep-new"]["status"] == "uploaded"
        assert by_guid["ep-new"]["after_cutoff"] is True
        assert by_guid["ep-old-1"]["status"] is None
        assert by_guid["ep-old-1"]["after_cutoff"] is False
        assert by_guid["ep-old-2"]["after_cutoff"] is False

    def test_empty_feed(self, client, tmp_path):
        with (
            patch("lex_without_lex.server.get_episodes", new_callable=AsyncMock, return_value=[]),
            patch("lex_without_lex.server.settings") as mock_settings,
        ):
            mock_settings.feed_url = "https://example.com/feed"
            mock_settings.data_dir = tmp_path
            mock_settings.episodes_after = datetime(2026, 3, 20, tzinfo=timezone.utc)
            resp = client.get("/episodes")

        assert resp.status_code == 200
        assert resp.json()["episodes"] == []


class TestProcessSpecificEpisodes:
    def test_returns_202_with_guids(self, client):
        with patch(
            "lex_without_lex.server._process_selected_episodes", new_callable=AsyncMock
        ):
            resp = client.post("/episodes/process", json={"guids": ["ep-old-1", "ep-old-2"]})

        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "processing"
        assert data["guids"] == ["ep-old-1", "ep-old-2"]

    def test_calls_helper_with_guids(self, client):
        with patch(
            "lex_without_lex.server._process_selected_episodes", new_callable=AsyncMock
        ) as mock_proc:
            client.post("/episodes/process", json={"guids": ["ep-old-1"]})

        mock_proc.assert_called_once()
        args = mock_proc.call_args
        assert args[0][0] == ["ep-old-1"]

    def test_empty_guids(self, client):
        with patch(
            "lex_without_lex.server._process_selected_episodes", new_callable=AsyncMock
        ):
            resp = client.post("/episodes/process", json={"guids": []})

        assert resp.status_code == 202
        assert resp.json()["guids"] == []
