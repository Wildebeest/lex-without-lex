from __future__ import annotations

from pathlib import Path

import httpx


async def download_episode(
    url: str,
    dest: Path,
    client: httpx.AsyncClient | None = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Stream-download audio file. Skips if dest already exists.

    Returns path to downloaded file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        return dest

    if client is None:
        async with httpx.AsyncClient() as c:
            return await _do_download(c, url, dest, chunk_size)
    return await _do_download(client, url, dest, chunk_size)


async def _do_download(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    chunk_size: int,
) -> Path:
    async with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            async for chunk in resp.aiter_bytes(chunk_size):
                f.write(chunk)
    return dest
