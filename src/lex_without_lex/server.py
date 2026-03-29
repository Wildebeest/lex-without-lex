from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Response
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

from .config import Settings
from .feed_parser import get_episodes
from .models import EpisodeState
from .pipeline import load_state, process_episode, process_new_episodes, save_state

logger = logging.getLogger(__name__)

settings = Settings()


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


def render_feed_xml(episodes: list[EpisodeState], settings: Settings) -> str:
    """Render RSS XML using Jinja2 template."""
    env = _get_jinja_env()
    template = env.get_template("feed.xml.j2")
    return template.render(episodes=episodes, base_url=settings.base_url)


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


@app.post("/process", status_code=202)
async def trigger_processing(background_tasks: BackgroundTasks):
    """Manually trigger the processing pipeline."""
    background_tasks.add_task(process_new_episodes, settings)
    return {"status": "processing"}


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


@app.post("/episodes/process", status_code=202)
async def process_specific_episodes(
    request: ProcessEpisodesRequest,
    background_tasks: BackgroundTasks,
):
    """Process specific episodes by guid, bypassing the date cutoff."""
    background_tasks.add_task(_process_selected_episodes, request.guids, settings)
    return {"status": "processing", "guids": request.guids}
