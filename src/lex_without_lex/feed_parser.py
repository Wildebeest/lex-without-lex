from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
import httpx

from .models import Episode


async def fetch_feed(url: str, client: httpx.AsyncClient | None = None) -> str:
    """Fetch raw RSS XML from URL."""
    if client is None:
        async with httpx.AsyncClient() as c:
            resp = await c.get(url)
            resp.raise_for_status()
            return resp.text
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.text


def parse_feed(xml: str) -> list[Episode]:
    """Parse RSS XML into Episode models. Returns episodes sorted newest-first.

    Skips entries that have no audio enclosure.
    """
    feed = feedparser.parse(xml)
    episodes: list[Episode] = []

    for entry in feed.entries:
        # Find audio enclosure
        audio_url = None
        for link in getattr(entry, "enclosures", []):
            if link.get("type", "").startswith("audio/") or link.get("href", "").endswith(".mp3"):
                audio_url = link["href"]
                break

        if not audio_url:
            continue

        # Parse publication date
        published = _parse_date(entry)

        # Parse duration (itunes:duration can be seconds or HH:MM:SS)
        duration = _parse_duration(entry)

        title = entry.get("title", "Untitled")

        # Extract content:encoded (feedparser normalizes to entry.content)
        content_encoded = ""
        content_list = getattr(entry, "content", None)
        if content_list and isinstance(content_list, list):
            content_encoded = content_list[0].get("value", "")
        if not content_encoded:
            content_encoded = entry.get("summary", "")

        # Extract itunes:image (feedparser normalizes to entry.image)
        itunes_image_url = ""
        image = getattr(entry, "image", None)
        if image and isinstance(image, dict):
            itunes_image_url = image.get("href", "")

        # Parse episode number from title (e.g. "#494 – ...")
        episode_number = _parse_episode_number(title)

        episodes.append(
            Episode(
                guid=entry.get("id", entry.get("link", audio_url)),
                title=title,
                published=published,
                audio_url=audio_url,
                duration_seconds=duration,
                description=entry.get("summary", ""),
                link=entry.get("link", ""),
                itunes_author=entry.get("author", ""),
                itunes_episode_type=entry.get("itunes_episodetype", "full"),
                itunes_image_url=itunes_image_url,
                content_encoded=content_encoded,
                episode_number=episode_number,
            )
        )

    episodes.sort(key=lambda e: e.published, reverse=True)
    return episodes


def _parse_date(entry) -> datetime:
    """Extract and parse publication date from a feed entry."""
    date_str = entry.get("published", "")
    if date_str:
        try:
            return parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            pass
    return datetime.now(tz=timezone.utc)


def _parse_duration(entry) -> int | None:
    """Parse itunes:duration which can be seconds int or HH:MM:SS string."""
    raw = entry.get("itunes_duration")
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    raw = str(raw).strip()
    if raw.isdigit():
        return int(raw)
    parts = raw.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        pass
    return None


def _parse_episode_number(title: str) -> int | None:
    """Extract episode number from title like '#494 – ...'."""
    m = re.search(r"#(\d+)", title)
    return int(m.group(1)) if m else None


async def get_episodes(url: str, client: httpx.AsyncClient | None = None) -> list[Episode]:
    """Convenience: fetch + parse."""
    xml = await fetch_feed(url, client)
    return parse_feed(xml)
