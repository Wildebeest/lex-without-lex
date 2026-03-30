from datetime import datetime

from pydantic import BaseModel


class Episode(BaseModel):
    guid: str
    title: str
    published: datetime
    audio_url: str
    duration_seconds: int | None = None
    description: str = ""
    link: str = ""
    itunes_author: str = ""
    itunes_episode_type: str = "full"
    itunes_image_url: str = ""
    content_encoded: str = ""
    episode_number: int | None = None


class EpisodeState(BaseModel):
    """Tracks processing state for an episode."""

    episode: Episode
    audio_path: str | None = None
    transcript_path: str | None = None
    edit_list_path: str | None = None
    output_path: str | None = None
    b2_url: str | None = None
    b2_file_name: str | None = None
    output_size_bytes: int | None = None
    output_duration_seconds: int | None = None
    status: str = "new"  # new | downloaded | transcribed | edited | assembled | uploaded


class TranscriptSegment(BaseModel):
    """A contiguous segment of speech by one speaker."""

    speaker: str  # "lex" or "guest"
    text: str
    start_ms: int
    end_ms: int


class Transcript(BaseModel):
    episode_guid: str
    segments: list[TranscriptSegment]
    raw_response: dict | None = None


class SegmentAction(BaseModel):
    """An action on a time range of the source audio."""

    action: str  # "keep" or "cut"
    start_ms: int
    end_ms: int
    speaker: str = ""
    reason: str = ""


class Interjection(BaseModel):
    """A TTS interjection to insert at a specific point."""

    insert_after_ms: int  # insertion point in source timeline
    text: str  # text for TTS
    context: str = ""  # why this interjection is needed


class EditList(BaseModel):
    episode_guid: str
    segments: list[SegmentAction]
    interjections: list[Interjection]
    summary: str = ""
    raw_response: str = ""


class Chapter(BaseModel):
    """A chapter marker with a timestamp and title."""

    start_seconds: float
    title: str
