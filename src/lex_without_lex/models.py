from datetime import datetime

from pydantic import BaseModel


class Episode(BaseModel):
    guid: str
    title: str
    published: datetime
    audio_url: str
    duration_seconds: int | None = None
    description: str = ""


class EpisodeState(BaseModel):
    """Tracks processing state for an episode."""

    episode: Episode
    audio_path: str | None = None
    transcript_path: str | None = None
    edit_list_path: str | None = None
    output_path: str | None = None
    b2_url: str | None = None
    status: str = "new"  # new | downloaded | transcribed | edited | assembled | uploaded
