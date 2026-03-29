import httpx
import pytest
import respx

from lex_without_lex.downloader import download_episode


class TestDownloadEpisode:
    @respx.mock
    @pytest.mark.asyncio
    async def test_creates_file(self, tmp_path):
        url = "https://media.example.com/episode.mp3"
        content = b"fake audio content " * 100
        respx.get(url).respond(200, content=content)

        dest = tmp_path / "episodes" / "episode.mp3"
        async with httpx.AsyncClient() as client:
            result = await download_episode(url, dest, client=client)

        assert result == dest
        assert dest.exists()
        assert dest.read_bytes() == content

    @respx.mock
    @pytest.mark.asyncio
    async def test_skips_existing_file(self, tmp_path):
        url = "https://media.example.com/episode.mp3"
        route = respx.get(url).respond(200, content=b"new content")

        dest = tmp_path / "episode.mp3"
        dest.write_bytes(b"existing content")

        async with httpx.AsyncClient() as client:
            result = await download_episode(url, dest, client=client)

        assert result == dest
        assert dest.read_bytes() == b"existing content"
        assert not route.called

    @respx.mock
    @pytest.mark.asyncio
    async def test_creates_parent_dirs(self, tmp_path):
        url = "https://media.example.com/episode.mp3"
        respx.get(url).respond(200, content=b"audio")

        dest = tmp_path / "deep" / "nested" / "episode.mp3"
        async with httpx.AsyncClient() as client:
            result = await download_episode(url, dest, client=client)

        assert result == dest
        assert dest.exists()

    @respx.mock
    @pytest.mark.asyncio
    async def test_raises_on_http_error(self, tmp_path):
        url = "https://media.example.com/episode.mp3"
        respx.get(url).respond(404)

        dest = tmp_path / "episode.mp3"
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await download_episode(url, dest, client=client)
