from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import RedirectResponse, StreamingResponse
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

from .config import Settings
from .chapters import parse_chapters_from_description, remap_chapters, chapters_to_json
from .feed_parser import get_episodes
from .models import EditList, EpisodeState
from .pipeline import load_state, process_episode, process_new_episodes, save_state
from .storage import B2Storage

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

settings = Settings()

KEEPALIVE_INTERVAL: float = 30


def _get_jinja_env() -> Environment:
    template_dir = Path(__file__).parent / "templates"
    return Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)


async def _scheduled_check() -> None:
    """Background task that checks for new episodes periodically."""
    while True:
        try:
            await process_new_episodes(settings)
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Scheduled check failed")
        await asyncio.sleep(6 * 3600)  # 6 hours


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_scheduled_check())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Lex Without Lex", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/feed.xml")
async def podcast_feed() -> Response:
    """Serve podcast RSS feed XML."""
    state_file = settings.data_dir / "state.json"
    state = load_state(state_file)

    # Only include uploaded episodes
    episodes = sorted(
        [es for es in state.values() if es.status == "uploaded"],
        key=lambda es: es.episode.published,
        reverse=True,
    )

    xml = render_feed_xml(episodes, settings)
    return Response(content=xml, media_type="application/xml")


def render_feed_xml(
    episodes: list[EpisodeState],
    settings: Settings,
) -> str:
    """Render RSS XML using Jinja2 template."""
    env = _get_jinja_env()
    template = env.get_template("feed.xml.j2")
    now = datetime.now(tz=timezone.utc)
    return template.render(
        episodes=episodes,
        base_url=settings.base_url,
        now=now,
    )


@app.get("/episodes/{episode_id:path}/audio")
async def episode_audio(episode_id: str) -> RedirectResponse:
    """302 redirect to B2 authorized download URL for traffic visibility."""
    state_file = settings.data_dir / "state.json"
    state = load_state(state_file)

    es = state.get(episode_id)
    if es is None or es.status != "uploaded":
        return Response(status_code=404, content="Episode not found")

    storage = B2Storage(
        settings.b2_key_id,
        settings.b2_application_key,
        settings.b2_bucket_name,
    )
    if not es.b2_file_name:
        safe = episode_id.replace("/", "_").replace(":", "_")
        es.b2_file_name = f"episodes/{safe}.mp3"

    url = storage.get_download_auth_url(es.b2_file_name)
    logger.info("Audio redirect: %s -> %s", episode_id, url[:80])
    return RedirectResponse(url=url, status_code=302)


@app.get("/episodes/{episode_id:path}/chapters.json")
async def episode_chapters(episode_id: str) -> Response:
    """Serve JSON chapters for an episode, remapped to edited timeline."""
    state_file = settings.data_dir / "state.json"
    state = load_state(state_file)

    es = state.get(episode_id)
    if es is None or es.status != "uploaded":
        return Response(status_code=404, content="Episode not found")

    # Parse chapters from description
    description = es.episode.content_encoded or es.episode.description
    chapters = parse_chapters_from_description(description)
    if not chapters:
        return Response(
            content='{"version":"1.2.0","chapters":[]}',
            media_type="application/json",
        )

    # Remap through edit list if available
    if es.edit_list_path and Path(es.edit_list_path).exists():
        edit_list = EditList.model_validate_json(Path(es.edit_list_path).read_text())
        chapters = remap_chapters(chapters, edit_list)

    return Response(content=chapters_to_json(chapters), media_type="application/json")


# --- Request/Response models ---


class EpisodeInfo(BaseModel):
    guid: str
    title: str
    published: datetime
    duration_seconds: int | None
    description: str
    status: str | None  # None if not yet processed
    after_cutoff: bool


class EpisodeListResponse(BaseModel):
    episodes: list[EpisodeInfo]
    cutoff_date: datetime


class ProcessEpisodesRequest(BaseModel):
    guids: list[str]


# --- Endpoints ---


@app.post("/process")
async def trigger_processing():
    """Manually trigger the processing pipeline.

    Returns a streaming response that stays open during processing,
    preventing Fly.io from auto-stopping the machine.
    """
    async def stream() -> AsyncGenerator[str, None]:
        yield json.dumps({"status": "processing"}) + "\n"
        task = asyncio.create_task(process_new_episodes(settings))
        while not task.done():
            await asyncio.sleep(KEEPALIVE_INTERVAL)
            yield "\n"  # keepalive
        exc = task.exception() if not task.cancelled() else None
        if exc:
            yield json.dumps({"status": "error", "error": str(exc)}) + "\n"
        else:
            yield json.dumps({"status": "complete"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/episodes", response_model=EpisodeListResponse)
async def list_episodes():
    """List all episodes from the feed with their processing status."""
    episodes = await get_episodes(settings.feed_url)
    state_file = settings.data_dir / "state.json"
    state = load_state(state_file)

    cutoff = settings.episodes_after
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)

    items = []
    for ep in episodes:
        pub = ep.published
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        existing = state.get(ep.guid)
        items.append(
            EpisodeInfo(
                guid=ep.guid,
                title=ep.title,
                published=ep.published,
                duration_seconds=ep.duration_seconds,
                description=ep.description,
                status=existing.status if existing else None,
                after_cutoff=pub >= cutoff,
            )
        )

    return EpisodeListResponse(episodes=items, cutoff_date=settings.episodes_after)


async def _process_selected_episodes(guids: list[str], settings: Settings) -> None:
    """Process specific episodes by guid, bypassing the date cutoff."""
    state_file = settings.data_dir / "state.json"
    state = load_state(state_file)

    episodes = await get_episodes(settings.feed_url)
    episodes_by_guid = {e.guid: e for e in episodes}

    for guid in guids:
        episode = episodes_by_guid.get(guid)
        if episode is None:
            logger.warning("Guid not found in feed: %s", guid)
            continue
        if guid in state and state[guid].status == "uploaded":
            logger.info("Skipping already-uploaded episode: %s", episode.title)
            continue
        try:
            episode_state = await process_episode(episode, settings, state.get(guid))
            state[guid] = episode_state
            save_state(state, state_file)
        except Exception:
            logger.exception("Failed to process episode: %s", episode.title)
            if guid not in state:
                state[guid] = EpisodeState(episode=episode, status="error")
            else:
                state[guid].status = "error"
            save_state(state, state_file)


@app.post("/episodes/process")
async def process_specific_episodes(request: ProcessEpisodesRequest):
    """Process specific episodes by guid, bypassing the date cutoff.

    Returns a streaming response that stays open during processing,
    preventing Fly.io from auto-stopping the machine.
    """
    guids = request.guids

    async def stream() -> AsyncGenerator[str, None]:
        yield json.dumps({"status": "processing", "guids": guids}) + "\n"
        task = asyncio.create_task(
            _process_selected_episodes(guids, settings)
        )
        while not task.done():
            await asyncio.sleep(KEEPALIVE_INTERVAL)
            yield "\n"  # keepalive
        exc = task.exception() if not task.cancelled() else None
        if exc:
            yield json.dumps({"status": "error", "error": str(exc)}) + "\n"
        else:
            yield json.dumps({"status": "complete"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
