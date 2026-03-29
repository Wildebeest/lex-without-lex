from __future__ import annotations

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

        episodes.append(
            Episode(
                guid=entry.get("id", entry.get("link", audio_url)),
                title=entry.get("title", "Untitled"),
                published=published,
                audio_url=audio_url,
                duration_seconds=duration,
                description=entry.get("summary", ""),
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


async def get_episodes(url: str, client: httpx.AsyncClient | None = None) -> list[Episode]:
    """Convenience: fetch + parse."""
    xml = await fetch_feed(url, client)
    return parse_feed(xml)
