import json

import httpx
import pytest
import respx

import lex_without_lex.transcriber as transcriber_mod
from lex_without_lex.transcriber import (
    GEMINI_API_URL,
    GEMINI_UPLOAD_URL,
    TRANSCRIPTION_PROMPT,
    _strip_code_fences,
    parse_gemini_response,
    transcribe_episode,
)


@pytest.fixture
def sample_gemini_response(fixtures_dir):
    return json.loads((fixtures_dir / "sample_transcript.json").read_text())


class TestParseGeminiResponse:
    def test_basic_parsing(self, sample_gemini_response):
        transcript = parse_gemini_response(sample_gemini_response, "ep-001")
        assert transcript.episode_guid == "ep-001"
        assert len(transcript.segments) == 8

    def test_speaker_labels(self, sample_gemini_response):
        transcript = parse_gemini_response(sample_gemini_response)
        speakers = {seg.speaker for seg in transcript.segments}
        assert speakers == {"lex", "guest"}

    def test_timestamps_monotonic(self, sample_gemini_response):
        transcript = parse_gemini_response(sample_gemini_response)
        for i in range(1, len(transcript.segments)):
            assert transcript.segments[i].start_ms >= transcript.segments[i - 1].start_ms

    def test_segments_have_text(self, sample_gemini_response):
        transcript = parse_gemini_response(sample_gemini_response)
        for seg in transcript.segments:
            assert len(seg.text) > 0

    def test_preserves_raw_response(self, sample_gemini_response):
        transcript = parse_gemini_response(sample_gemini_response)
        assert transcript.raw_response == sample_gemini_response

    def test_raises_on_empty_candidates(self):
        with pytest.raises(ValueError, match="No candidates"):
            parse_gemini_response({"candidates": []})

    def test_raises_on_empty_parts(self):
        with pytest.raises(ValueError, match="No parts"):
            parse_gemini_response({"candidates": [{"content": {"parts": []}}]})


class TestStripCodeFences:
    def test_strips_json_fences(self):
        text = '```json\n{"key": "value"}\n```'
        assert _strip_code_fences(text) == '{"key": "value"}'

    def test_strips_plain_fences(self):
        text = '```\n{"key": "value"}\n```'
        assert _strip_code_fences(text) == '{"key": "value"}'

    def test_no_fences_passthrough(self):
        text = '{"key": "value"}'
        assert _strip_code_fences(text) == '{"key": "value"}'


class TestTranscriptionPrompt:
    def test_prompt_requests_json(self):
        assert "JSON" in TRANSCRIPTION_PROMPT

    def test_prompt_mentions_lex(self):
        assert "Lex Fridman" in TRANSCRIPTION_PROMPT

    def test_prompt_mentions_timestamps(self):
        assert "start_ms" in TRANSCRIPTION_PROMPT
        assert "end_ms" in TRANSCRIPTION_PROMPT


class TestTranscribeEpisode:
    @respx.mock
    @pytest.mark.asyncio
    async def test_calls_gemini_api(self, tmp_path, sample_gemini_response):
        # Create fake audio file
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"fake audio bytes")

        # Mock upload endpoint
        upload_url = "https://storage.googleapis.com/upload/fake"
        respx.post(GEMINI_UPLOAD_URL).respond(
            200,
            headers={"X-Goog-Upload-URL": upload_url},
        )
        respx.put(upload_url).respond(
            200,
            json={"file": {"uri": "gs://bucket/test.mp3"}},
        )

        # Mock generateContent endpoint
        respx.post(GEMINI_API_URL).respond(200, json=sample_gemini_response)

        async with httpx.AsyncClient() as client:
            transcript = await transcribe_episode(
                audio_path, "fake-api-key", "ep-001", client=client
            )

        assert transcript.episode_guid == "ep-001"
        assert len(transcript.segments) == 8

    @respx.mock
    @pytest.mark.asyncio
    async def test_raises_on_api_error(self, tmp_path):
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"fake audio bytes")

        respx.post(GEMINI_UPLOAD_URL).respond(
            200,
            headers={"X-Goog-Upload-URL": "https://storage.googleapis.com/upload/fake"},
        )
        respx.put("https://storage.googleapis.com/upload/fake").respond(
            200,
            json={"file": {"uri": "gs://bucket/test.mp3"}},
        )
        respx.post(GEMINI_API_URL).respond(500)

        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await transcribe_episode(audio_path, "fake-key", "ep-001", client=client)

    @respx.mock
    @pytest.mark.asyncio
    async def test_chunked_upload(self, tmp_path, sample_gemini_response, monkeypatch):
        """Upload is split into multiple chunks when file exceeds chunk size."""
        monkeypatch.setattr(transcriber_mod, "UPLOAD_CHUNK_SIZE", 8)

        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"a" * 20)  # 20 bytes → 3 chunks (8+8+4)

        upload_url = "https://storage.googleapis.com/upload/fake"
        respx.post(GEMINI_UPLOAD_URL).respond(
            200,
            headers={"X-Goog-Upload-URL": upload_url},
        )
        put_route = respx.put(upload_url).respond(
            200,
            json={"file": {"uri": "gs://bucket/test.mp3"}},
        )
        respx.post(GEMINI_API_URL).respond(200, json=sample_gemini_response)

        async with httpx.AsyncClient() as client:
            transcript = await transcribe_episode(
                audio_path, "fake-api-key", "ep-001", client=client
            )

        assert transcript.episode_guid == "ep-001"
        assert put_route.call_count == 3

        # Verify chunk offsets and commands
        calls = put_route.calls
        assert calls[0].request.headers["x-goog-upload-offset"] == "0"
        assert calls[0].request.headers["x-goog-upload-command"] == "upload"
        assert calls[1].request.headers["x-goog-upload-offset"] == "8"
        assert calls[1].request.headers["x-goog-upload-command"] == "upload"
        assert calls[2].request.headers["x-goog-upload-offset"] == "16"
        assert calls[2].request.headers["x-goog-upload-command"] == "upload, finalize"
