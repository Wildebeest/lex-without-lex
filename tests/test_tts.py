import httpx
import pytest
import respx

from lex_without_lex.models import Interjection
from lex_without_lex.tts import (
    ELEVENLABS_TTS_URL,
    _cache_key,
    generate_all_interjections,
    generate_interjection,
)

VOICE_ID = "ktrGUw7rURIQyMrQZqCu"


class TestCacheKey:
    def test_deterministic(self):
        assert _cache_key("hello world") == _cache_key("hello world")

    def test_normalized(self):
        assert _cache_key("Hello World") == _cache_key("hello world")
        assert _cache_key("  hello world  ") == _cache_key("hello world")

    def test_different_text_different_key(self):
        assert _cache_key("hello") != _cache_key("goodbye")


class TestGenerateInterjection:
    @respx.mock
    @pytest.mark.asyncio
    async def test_calls_elevenlabs_api(self, tmp_path):
        url = ELEVENLABS_TTS_URL.format(voice_id=VOICE_ID)
        fake_audio = b"fake mp3 audio bytes"
        respx.post(url).respond(200, content=fake_audio)

        cache_dir = tmp_path / "cache"
        async with httpx.AsyncClient() as client:
            result = await generate_interjection(
                "On the topic of AI...", VOICE_ID, "fake-key", cache_dir, client
            )

        assert result.exists()
        assert result.read_bytes() == fake_audio
        # Verify API key header
        assert respx.calls[0].request.headers["xi-api-key"] == "fake-key"

    @respx.mock
    @pytest.mark.asyncio
    async def test_uses_cache(self, tmp_path):
        url = ELEVENLABS_TTS_URL.format(voice_id=VOICE_ID)
        route = respx.post(url).respond(200, content=b"new audio")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Pre-create cached file
        cache_file = cache_dir / f"{_cache_key('hello')}.mp3"
        cache_file.write_bytes(b"cached audio")

        async with httpx.AsyncClient() as client:
            result = await generate_interjection(
                "hello", VOICE_ID, "fake-key", cache_dir, client
            )

        assert result.read_bytes() == b"cached audio"
        assert not route.called

    @respx.mock
    @pytest.mark.asyncio
    async def test_raises_on_api_error(self, tmp_path):
        url = ELEVENLABS_TTS_URL.format(voice_id=VOICE_ID)
        respx.post(url).respond(401)

        cache_dir = tmp_path / "cache"
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await generate_interjection(
                    "test", VOICE_ID, "fake-key", cache_dir, client
                )


class TestGenerateAllInterjections:
    @respx.mock
    @pytest.mark.asyncio
    async def test_generates_all(self, tmp_path):
        url = ELEVENLABS_TTS_URL.format(voice_id=VOICE_ID)
        respx.post(url).respond(200, content=b"audio")

        interjections = [
            Interjection(insert_after_ms=5000, text="First interjection"),
            Interjection(insert_after_ms=15000, text="Second interjection"),
        ]

        cache_dir = tmp_path / "cache"
        async with httpx.AsyncClient() as client:
            result = await generate_all_interjections(
                interjections, VOICE_ID, "fake-key", cache_dir, client
            )

        assert 5000 in result
        assert 15000 in result
        assert result[5000].exists()
        assert result[15000].exists()
