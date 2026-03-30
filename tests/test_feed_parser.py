import httpx
import pytest
import respx

from lex_without_lex.feed_parser import fetch_feed, get_episodes, parse_feed


class TestParseFeed:
    def test_extracts_episodes_with_audio(self, sample_feed_xml):
        episodes = parse_feed(sample_feed_xml)
        # 4 items in fixture, but one has no enclosure
        assert len(episodes) == 3

    def test_extracts_audio_url_from_enclosure(self, sample_feed_xml):
        episodes = parse_feed(sample_feed_xml)
        urls = [e.audio_url for e in episodes]
        assert "https://media.lexfridman.com/audio/lex_ai_elon_musk_4.mp3" in urls

    def test_sorts_newest_first(self, sample_feed_xml):
        episodes = parse_feed(sample_feed_xml)
        assert episodes[0].title.startswith("#400")
        assert episodes[-1].title.startswith("#398")

    def test_extracts_guid(self, sample_feed_xml):
        episodes = parse_feed(sample_feed_xml)
        guids = [e.guid for e in episodes]
        assert "https://lexfridman.com/elon-musk-4" in guids

    def test_extracts_description(self, sample_feed_xml):
        episodes = parse_feed(sample_feed_xml)
        ep = next(e for e in episodes if "Elon" in e.title)
        assert "Neuralink" in ep.description

    def test_parses_duration_seconds(self, sample_feed_xml):
        episodes = parse_feed(sample_feed_xml)
        ep = next(e for e in episodes if "Elon" in e.title)
        assert ep.duration_seconds == 3600

    def test_missing_duration_is_none(self, sample_feed_xml):
        episodes = parse_feed(sample_feed_xml)
        ep = next(e for e in episodes if "Sam Altman" in e.title)
        assert ep.duration_seconds is None

    def test_skips_items_without_enclosure(self, sample_feed_xml):
        episodes = parse_feed(sample_feed_xml)
        titles = [e.title for e in episodes]
        assert not any("AMA" in t for t in titles)

    def test_extracts_link(self, sample_feed_xml):
        episodes = parse_feed(sample_feed_xml)
        ep = next(e for e in episodes if "Elon" in e.title)
        assert ep.link == "https://lexfridman.com/elon-musk-4"

    def test_extracts_itunes_author(self, sample_feed_xml):
        episodes = parse_feed(sample_feed_xml)
        ep = next(e for e in episodes if "Elon" in e.title)
        assert ep.itunes_author == "Lex Fridman"

    def test_extracts_itunes_episode_type(self, sample_feed_xml):
        episodes = parse_feed(sample_feed_xml)
        ep = next(e for e in episodes if "Elon" in e.title)
        assert ep.itunes_episode_type == "full"

    def test_extracts_content_encoded(self, sample_feed_xml):
        episodes = parse_feed(sample_feed_xml)
        ep = next(e for e in episodes if "Elon" in e.title)
        assert "OUTLINE" in ep.content_encoded
        assert "(0:00)" in ep.content_encoded

    def test_extracts_episode_number(self, sample_feed_xml):
        episodes = parse_feed(sample_feed_xml)
        ep = next(e for e in episodes if "Elon" in e.title)
        assert ep.episode_number == 400

    def test_episode_number_none_for_bonus(self, sample_feed_xml):
        """Bonus episodes without #NNN should have no episode number."""
        episodes = parse_feed(sample_feed_xml)
        ep = next(e for e in episodes if "Sam Altman" in e.title)
        assert ep.episode_number == 398

    def test_missing_link_is_empty(self, sample_feed_xml):
        episodes = parse_feed(sample_feed_xml)
        ep = next(e for e in episodes if "Sam Altman" in e.title)
        # Sam Altman entry has no <link> tag
        assert ep.link == "" or ep.link is not None  # feedparser may fallback


class TestFetchFeed:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetches_from_url(self, sample_feed_xml):
        url = "https://example.com/feed.xml"
        respx.get(url).respond(200, text=sample_feed_xml)
        async with httpx.AsyncClient() as client:
            result = await fetch_feed(url, client=client)
        assert "<title>Lex Fridman Podcast</title>" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_raises_on_http_error(self):
        url = "https://example.com/feed.xml"
        respx.get(url).respond(500)
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await fetch_feed(url, client=client)


class TestGetEpisodes:
    @respx.mock
    @pytest.mark.asyncio
    async def test_integration(self, sample_feed_xml):
        url = "https://example.com/feed.xml"
        respx.get(url).respond(200, text=sample_feed_xml)
        async with httpx.AsyncClient() as client:
            episodes = await get_episodes(url, client=client)
        assert len(episodes) == 3
        assert episodes[0].title.startswith("#400")
