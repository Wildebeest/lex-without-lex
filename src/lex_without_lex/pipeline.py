from __future__ import annotations

import json
import logging
from datetime import timezone
from pathlib import Path

import httpx

from .config import Settings
from .downloader import download_episode
from .editor import generate_edit_list, validate_edit_list
from .feed_parser import get_episodes
from .models import EditList, Episode, EpisodeState, Transcript
from .storage import B2Storage
from .transcriber import transcribe_episode
from .tts import generate_all_interjections
from .audio import assemble_audio

logger = logging.getLogger(__name__)


def load_state(state_file: Path) -> dict[str, EpisodeState]:
    """Load processing state from JSON file."""
    if not state_file.exists():
        return {}
    data = json.loads(state_file.read_text())
    return {guid: EpisodeState.model_validate(v) for guid, v in data.items()}


def save_state(state: dict[str, EpisodeState], state_file: Path) -> None:
    """Persist processing state to JSON file."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    data = {guid: es.model_dump(mode="json") for guid, es in state.items()}
    state_file.write_text(json.dumps(data, indent=2, default=str))


async def process_new_episodes(settings: Settings | None = None) -> None:
    """Main pipeline: check feed, process any unprocessed episodes."""
    if settings is None:
        settings = Settings()

    state_file = settings.data_dir / "state.json"
    state = load_state(state_file)

    episodes = await get_episodes(settings.feed_url)
    cutoff = settings.episodes_after
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    episodes = [e for e in episodes if e.published >= cutoff]
    logger.info("Found %d episodes after %s", len(episodes), settings.episodes_after.date())

    for episode in episodes:
        if episode.guid in state and state[episode.guid].status == "uploaded":
            continue

        try:
            episode_state = await process_episode(episode, settings, state.get(episode.guid))
            state[episode.guid] = episode_state
            save_state(state, state_file)
        except Exception:
            logger.exception("Failed to process episode: %s", episode.title)
            if episode.guid not in state:
                state[episode.guid] = EpisodeState(episode=episode, status="error")
            else:
                state[episode.guid].status = "error"
            save_state(state, state_file)


async def process_episode(
    episode: Episode,
    settings: Settings,
    existing_state: EpisodeState | None = None,
) -> EpisodeState:
    """Process a single episode through the full pipeline."""
    es = existing_state or EpisodeState(episode=episode)
    data_dir = settings.data_dir

    # Step 1: Download
    if es.status in ("new", "error"):
        safe_name = episode.guid.replace("/", "_").replace(":", "_")
        audio_path = data_dir / "episodes" / f"{safe_name}.mp3"
        await download_episode(episode.audio_url, audio_path)
        es.audio_path = str(audio_path)
        es.status = "downloaded"
        logger.info("Downloaded: %s", episode.title)

    # Step 2: Transcribe
    if es.status == "downloaded":
        transcript = await transcribe_episode(
            Path(es.audio_path),
            settings.gemini_api_key,
            episode.guid,
        )
        transcript_path = data_dir / "transcripts" / f"{safe_name}.json"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(transcript.model_dump_json(indent=2))
        es.transcript_path = str(transcript_path)
        es.status = "transcribed"
        logger.info("Transcribed: %s", episode.title)

    # Step 3: Edit
    if es.status == "transcribed":
        transcript = Transcript.model_validate_json(Path(es.transcript_path).read_text())
        edit_list = await generate_edit_list(
            transcript, settings.anthropic_api_key
        )
        warnings = validate_edit_list(edit_list, transcript)
        if warnings:
            logger.warning("Edit list warnings for %s: %s", episode.title, warnings)

        edit_path = data_dir / "edits" / f"{safe_name}.json"
        edit_path.parent.mkdir(parents=True, exist_ok=True)
        edit_path.write_text(edit_list.model_dump_json(indent=2))
        es.edit_list_path = str(edit_path)
        es.status = "edited"
        logger.info("Edited: %s", episode.title)

    # Step 4: Assemble audio
    if es.status == "edited":
        edit_list = EditList.model_validate_json(Path(es.edit_list_path).read_text())

        interjection_paths = await generate_all_interjections(
            edit_list.interjections,
            settings.elevenlabs_voice_id,
            settings.elevenlabs_api_key,
            data_dir / "interjections",
        )

        output_path = data_dir / "output" / f"{safe_name}.mp3"
        assemble_audio(
            Path(es.audio_path), edit_list, interjection_paths, output_path
        )
        es.output_path = str(output_path)
        es.status = "assembled"
        logger.info("Assembled: %s", episode.title)

    # Step 5: Upload to B2
    if es.status == "assembled":
        storage = B2Storage(
            settings.b2_key_id,
            settings.b2_application_key,
            settings.b2_bucket_name,
        )
        b2_url = storage.upload_episode(Path(es.output_path), episode.guid)
        es.b2_url = b2_url
        es.status = "uploaded"
        logger.info("Uploaded: %s -> %s", episode.title, b2_url)

    return es
