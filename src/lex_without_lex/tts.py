from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from .models import Interjection

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


def _cache_key(text: str) -> str:
    """SHA256 hash of normalized text for caching."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


async def generate_interjection(
    text: str,
    voice_id: str,
    api_key: str,
    cache_dir: Path,
    client: httpx.AsyncClient | None = None,
) -> Path:
    """Generate TTS audio for interjection text. Returns path to audio file.

    Uses content-hash caching: if cache file exists, skip API call.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_cache_key(text)}.mp3"

    if cache_file.exists() and cache_file.stat().st_size > 0:
        return cache_file

    if client is None:
        async with httpx.AsyncClient() as c:
            return await _do_generate(c, text, voice_id, api_key, cache_file)
    return await _do_generate(client, text, voice_id, api_key, cache_file)


async def _do_generate(
    client: httpx.AsyncClient,
    text: str,
    voice_id: str,
    api_key: str,
    cache_file: Path,
) -> Path:
    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)
    resp = await client.post(
        url,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        },
    )
    resp.raise_for_status()
    cache_file.write_bytes(resp.content)
    return cache_file


async def generate_all_interjections(
    interjections: list[Interjection],
    voice_id: str,
    api_key: str,
    cache_dir: Path,
    client: httpx.AsyncClient | None = None,
) -> dict[int, Path]:
    """Generate all interjections, return mapping of insert_after_ms -> audio_path."""
    result: dict[int, Path] = {}
    for inj in interjections:
        path = await generate_interjection(
            inj.text, voice_id, api_key, cache_dir, client
        )
        result[inj.insert_after_ms] = path
    return result
