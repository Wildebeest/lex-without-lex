import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from xml.etree import ElementTree

import pytest
from fastapi.testclient import TestClient

from lex_without_lex.models import (
    EditList,
    Episode,
    EpisodeState,
    SegmentAction,
)
from lex_without_lex.server import app, render_feed_xml
from lex_without_lex.config import Settings


def _parse_ndjson_lines(text: str) -> list[dict]:
    """Parse non-empty lines from an ndjson streaming response."""
    return [json.loads(line) for line in text.strip().splitlines() if line.strip()]


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
                title="#501 – Guest Talk: AI Future",
                published=datetime(2025, 3, 10, tzinfo=timezone.utc),
                audio_url="https://example.com/ep1.mp3",
                duration_seconds=3600,
                description="A deep conversation about AI.",
                link="https://example.com/ep1",
                itunes_author="Lex Fridman",
                itunes_episode_type="full",
                episode_number=501,
            ),
            b2_url="https://b2.example.com/episodes/ep-1.mp3",
            b2_file_name="episodes/ep-1.mp3",
            output_size_bytes=50000000,
            output_duration_seconds=1800,
            status="uploaded",
        ),
        EpisodeState(
            episode=Episode(
                guid="ep-2",
                title="#502 – Guest Talk: Robotics",
                published=datetime(2025, 3, 5, tzinfo=timezone.utc),
                audio_url="https://example.com/ep2.mp3",
                duration_seconds=7200,
                description="Discussion on robotics.",
                itunes_author="Lex Fridman",
                itunes_episode_type="full",
                episode_number=502,
            ),
            b2_url="https://b2.example.com/episodes/ep-2.mp3",
            b2_file_name="episodes/ep-2.mp3",
            output_size_bytes=80000000,
            output_duration_seconds=3600,
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


NS = {
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "atom": "http://www.w3.org/2005/Atom",
    "podcast": "https://podcastindex.org/namespace/1.0",
}


class TestRenderFeedXml:
    def test_contains_episodes(self, sample_episodes):
        settings = Settings(base_url="https://test.example.com")
        xml = render_feed_xml(sample_episodes, settings)
        root = ElementTree.fromstring(xml)

        items = root.findall(".//item")
        assert len(items) == 2

    def test_episode_has_enclosure_with_302_url(self, sample_episodes):
        settings = Settings(base_url="https://test.example.com")
        xml = render_feed_xml(sample_episodes, settings)
        root = ElementTree.fromstring(xml)

        enclosures = root.findall(".//enclosure")
        assert len(enclosures) == 2
        # Enclosure URL should point to our 302 redirect endpoint
        assert "/episodes/" in enclosures[0].get("url", "")
        assert "/audio" in enclosures[0].get("url", "")
        assert "test.example.com" in enclosures[0].get("url", "")

    def test_enclosure_has_correct_length(self, sample_episodes):
        settings = Settings(base_url="https://test.example.com")
        xml = render_feed_xml(sample_episodes, settings)
        root = ElementTree.fromstring(xml)

        enclosures = root.findall(".//enclosure")
        assert enclosures[0].get("length") == "50000000"

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

    def test_has_itunes_image(self, sample_episodes):
        settings = Settings(base_url="https://test.example.com")
        xml = render_feed_xml(sample_episodes, settings)
        root = ElementTree.fromstring(xml)
        img = root.find(".//channel/itunes:image", NS)
        assert img is not None
        assert img.get("href") != ""

    def test_has_podcast_medium(self, sample_episodes):
        settings = Settings(base_url="https://test.example.com")
        xml = render_feed_xml(sample_episodes, settings)
        root = ElementTree.fromstring(xml)
        medium = root.find(".//channel/podcast:medium", NS)
        assert medium is not None
        assert medium.text == "podcast"

    def test_has_last_build_date(self, sample_episodes):
        settings = Settings(base_url="https://test.example.com")
        xml = render_feed_xml(sample_episodes, settings)
        root = ElementTree.fromstring(xml)
        lbd = root.find(".//channel/lastBuildDate")
        assert lbd is not None
        assert lbd.text is not None

    def test_episode_has_itunes_duration(self, sample_episodes):
        settings = Settings(base_url="https://test.example.com")
        xml = render_feed_xml(sample_episodes, settings)
        root = ElementTree.fromstring(xml)
        items = root.findall(".//item")
        duration = items[0].find("itunes:duration", NS)
        assert duration is not None
        # output_duration_seconds=1800 -> 0:30:00
        assert duration.text == "0:30:00"

    def test_episode_has_episode_type(self, sample_episodes):
        settings = Settings(base_url="https://test.example.com")
        xml = render_feed_xml(sample_episodes, settings)
        root = ElementTree.fromstring(xml)
        items = root.findall(".//item")
        ep_type = items[0].find("itunes:episodeType", NS)
        assert ep_type is not None
        assert ep_type.text == "full"

    def test_episode_has_episode_number(self, sample_episodes):
        settings = Settings(base_url="https://test.example.com")
        xml = render_feed_xml(sample_episodes, settings)
        root = ElementTree.fromstring(xml)
        items = root.findall(".//item")
        ep_num = items[0].find("itunes:episode", NS)
        assert ep_num is not None
        assert ep_num.text == "501"

    def test_episode_has_chapters_link(self, sample_episodes):
        settings = Settings(base_url="https://test.example.com")
        xml = render_feed_xml(sample_episodes, settings)
        root = ElementTree.fromstring(xml)
        items = root.findall(".//item")
        chapters = items[0].find("podcast:chapters", NS)
        assert chapters is not None
        assert "/chapters.json" in chapters.get("url", "")

    def test_has_atom_self_link(self, sample_episodes):
        settings = Settings(base_url="https://test.example.com")
        xml = render_feed_xml(sample_episodes, settings)
        root = ElementTree.fromstring(xml)
        atom_link = root.find(".//channel/atom:link[@rel='self']", NS)
        assert atom_link is not None
        assert atom_link.get("href") == "https://test.example.com/feed.xml"


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
    def test_returns_streaming_response(self, client):
        with patch("lex_without_lex.server.process_new_episodes", new_callable=AsyncMock), \
             patch("lex_without_lex.server.KEEPALIVE_INTERVAL", 0.01):
            resp = client.post("/process")
        assert resp.status_code == 200
        lines = _parse_ndjson_lines(resp.text)
        assert lines[0] == {"status": "processing"}
        assert lines[-1] == {"status": "complete"}

    def test_calls_process_new_episodes(self, client):
        with patch("lex_without_lex.server.process_new_episodes", new_callable=AsyncMock) as mock_proc, \
             patch("lex_without_lex.server.KEEPALIVE_INTERVAL", 0.01):
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

    def test_handles_naive_cutoff_datetime(self, client, tmp_path):
        """Cutoff from config is naive; feed dates are tz-aware. Should not crash."""
        episodes = _make_episodes()

        with (
            patch("lex_without_lex.server.get_episodes", new_callable=AsyncMock, return_value=episodes),
            patch("lex_without_lex.server.settings") as mock_settings,
        ):
            mock_settings.feed_url = "https://example.com/feed"
            mock_settings.data_dir = tmp_path
            mock_settings.episodes_after = datetime(2026, 3, 20)  # naive

            (tmp_path / "state.json").write_text("{}")
            resp = client.get("/episodes")

        assert resp.status_code == 200
        by_guid = {ep["guid"]: ep for ep in resp.json()["episodes"]}
        assert by_guid["ep-new"]["after_cutoff"] is True
        assert by_guid["ep-old-1"]["after_cutoff"] is False

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
    def test_returns_streaming_with_guids(self, client):
        with patch(
            "lex_without_lex.server._process_selected_episodes", new_callable=AsyncMock
        ), patch("lex_without_lex.server.KEEPALIVE_INTERVAL", 0.01):
            resp = client.post("/episodes/process", json={"guids": ["ep-old-1", "ep-old-2"]})

        assert resp.status_code == 200
        lines = _parse_ndjson_lines(resp.text)
        assert lines[0]["status"] == "processing"
        assert lines[0]["guids"] == ["ep-old-1", "ep-old-2"]
        assert lines[-1] == {"status": "complete"}

    def test_calls_helper_with_guids(self, client):
        with patch(
            "lex_without_lex.server._process_selected_episodes", new_callable=AsyncMock
        ) as mock_proc, patch("lex_without_lex.server.KEEPALIVE_INTERVAL", 0.01):
            client.post("/episodes/process", json={"guids": ["ep-old-1"]})

        mock_proc.assert_called_once()
        args = mock_proc.call_args
        assert args[0][0] == ["ep-old-1"]

    def test_empty_guids(self, client):
        with patch(
            "lex_without_lex.server._process_selected_episodes", new_callable=AsyncMock
        ), patch("lex_without_lex.server.KEEPALIVE_INTERVAL", 0.01):
            resp = client.post("/episodes/process", json={"guids": []})

        assert resp.status_code == 200
        lines = _parse_ndjson_lines(resp.text)
        assert lines[0]["guids"] == []


class TestEpisodeAudioRedirect:
    def test_returns_302(self, client, tmp_path, sample_episodes):
        from lex_without_lex.pipeline import save_state
        state = {ep.episode.guid: ep for ep in sample_episodes}
        save_state(state, tmp_path / "state.json")

        mock_storage = MagicMock()
        mock_storage.get_download_auth_url.return_value = "https://b2.example.com/file?auth=token"

        with (
            patch("lex_without_lex.server.settings") as mock_settings,
            patch("lex_without_lex.server.B2Storage", return_value=mock_storage),
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.b2_key_id = "key"
            mock_settings.b2_application_key = "secret"
            mock_settings.b2_bucket_name = "bucket"
            resp = client.get("/episodes/ep-1/audio", follow_redirects=False)

        assert resp.status_code == 302
        assert "b2.example.com" in resp.headers["location"]

    def test_returns_404_for_unknown(self, client, tmp_path):
        (tmp_path / "state.json").write_text("{}")
        with patch("lex_without_lex.server.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            resp = client.get("/episodes/nonexistent/audio", follow_redirects=False)

        assert resp.status_code == 404


class TestEpisodeChapters:
    def test_returns_chapters_json(self, client, tmp_path, sample_episodes):
        from lex_without_lex.pipeline import save_state

        # Patch episode to have description with chapters
        sample_episodes[0].episode.content_encoded = (
            "<p>OUTLINE:</p><p>(0:00) – Intro<br/>(30:00) – Topic</p>"
        )
        state = {ep.episode.guid: ep for ep in sample_episodes}
        save_state(state, tmp_path / "state.json")

        with patch("lex_without_lex.server.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            resp = client.get("/episodes/ep-1/chapters.json")

        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "1.2.0"
        assert len(data["chapters"]) == 2
        assert data["chapters"][0]["title"] == "Intro"

    def test_returns_empty_chapters_when_none(self, client, tmp_path, sample_episodes):
        from lex_without_lex.pipeline import save_state

        sample_episodes[0].episode.description = "No timestamps here."
        sample_episodes[0].episode.content_encoded = ""
        state = {ep.episode.guid: ep for ep in sample_episodes}
        save_state(state, tmp_path / "state.json")

        with patch("lex_without_lex.server.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            resp = client.get("/episodes/ep-1/chapters.json")

        assert resp.status_code == 200
        data = resp.json()
        assert data["chapters"] == []

    def test_returns_404_for_unknown(self, client, tmp_path):
        (tmp_path / "state.json").write_text("{}")
        with patch("lex_without_lex.server.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            resp = client.get("/episodes/nonexistent/chapters.json")

        assert resp.status_code == 404
