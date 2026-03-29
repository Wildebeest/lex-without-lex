from datetime import datetime, timezone

import pytest

from lex_without_lex.models import Episode, EpisodeState
from lex_without_lex.pipeline import load_state, save_state


class TestStateRoundtrip:
    def test_save_and_load(self, tmp_path):
        state_file = tmp_path / "state.json"
        episode = Episode(
            guid="test-ep-1",
            title="Test Episode",
            published=datetime(2025, 1, 1, tzinfo=timezone.utc),
            audio_url="https://example.com/test.mp3",
            duration_seconds=3600,
        )
        state = {
            "test-ep-1": EpisodeState(
                episode=episode,
                audio_path="/tmp/test.mp3",
                status="downloaded",
            )
        }

        save_state(state, state_file)
        loaded = load_state(state_file)

        assert "test-ep-1" in loaded
        assert loaded["test-ep-1"].status == "downloaded"
        assert loaded["test-ep-1"].episode.title == "Test Episode"
        assert loaded["test-ep-1"].audio_path == "/tmp/test.mp3"

    def test_load_missing_file(self, tmp_path):
        state_file = tmp_path / "nonexistent.json"
        state = load_state(state_file)
        assert state == {}

    def test_multiple_episodes(self, tmp_path):
        state_file = tmp_path / "state.json"
        state = {}
        for i in range(3):
            ep = Episode(
                guid=f"ep-{i}",
                title=f"Episode {i}",
                published=datetime(2025, 1, i + 1, tzinfo=timezone.utc),
                audio_url=f"https://example.com/ep{i}.mp3",
            )
            state[f"ep-{i}"] = EpisodeState(episode=ep, status="new")

        save_state(state, state_file)
        loaded = load_state(state_file)
        assert len(loaded) == 3

    def test_creates_parent_dirs(self, tmp_path):
        state_file = tmp_path / "deep" / "nested" / "state.json"
        ep = Episode(
            guid="ep-1",
            title="Test",
            published=datetime(2025, 1, 1, tzinfo=timezone.utc),
            audio_url="https://example.com/test.mp3",
        )
        state = {"ep-1": EpisodeState(episode=ep)}
        save_state(state, state_file)
        assert state_file.exists()
