from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from jinja2 import Environment, FileSystemLoader

from .config import Settings
from .models import EpisodeState
from .pipeline import load_state, process_new_episodes

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
