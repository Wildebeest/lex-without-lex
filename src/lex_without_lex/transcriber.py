from __future__ import annotations

import json
import logging
import mimetypes
import re
import tempfile
from pathlib import Path

import httpx

from .models import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)

GEMINI_UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB
TRANSCRIPTION_CHUNK_MS = 30 * 60 * 1000  # 30 minutes per chunk

TRANSCRIPTION_PROMPT = """\
You are a professional podcast transcription service. Transcribe the provided audio with detailed speaker diarization and precise timestamps.

Instructions:
1. Identify speakers. The host is Lex Fridman — label him as "lex". Label the guest as "guest". If there are multiple guests, label them as "guest1", "guest2", etc.
   Note: Lex Fridman always begins episodes with a solo introduction/monologue before the guest speaks, often followed by sponsor reads and personal reflections. Label ALL of this opening section as "lex" even though no other speaker is present.
2. Produce segments of continuous speech by one speaker. Each segment should be a natural utterance or turn in conversation.
3. Provide timestamps in milliseconds from the start of the audio.
4. Capture ALL speech accurately, including filler words, false starts, and interruptions.
5. Note any significant non-speech audio (laughter, long pauses) in the text field with brackets like [laughs] or [long pause].

Output ONLY valid JSON (no markdown fences) with this exact structure:
{
  "segments": [
    {
      "speaker": "lex",
      "text": "Welcome to the podcast...",
      "start_ms": 0,
      "end_ms": 5200
    },
    {
      "speaker": "guest",
      "text": "Thanks for having me...",
      "start_ms": 5200,
      "end_ms": 9800
    }
  ]
}
"""


async def transcribe_episode(
    audio_path: Path,
    api_key: str,
    episode_guid: str = "",
    client: httpx.AsyncClient | None = None,
) -> Transcript:
    """Upload audio to Gemini and get back a diarized transcript.

    For long episodes that exceed Gemini's output token limit, automatically
    splits the audio into chunks and transcribes each separately.
    """
    if client is None:
        async with httpx.AsyncClient(timeout=600.0) as c:
            return await _transcribe_with_chunking(c, audio_path, api_key, episode_guid)
    return await _transcribe_with_chunking(client, audio_path, api_key, episode_guid)


async def _transcribe_with_chunking(
    client: httpx.AsyncClient,
    audio_path: Path,
    api_key: str,
    episode_guid: str,
) -> Transcript:
    """Transcribe audio, splitting into chunks if longer than TRANSCRIPTION_CHUNK_MS."""
    from .audio import get_audio_duration_ms, split_audio

    duration_ms = get_audio_duration_ms(audio_path)
    if duration_ms <= TRANSCRIPTION_CHUNK_MS:
        logger.info("Audio is %d min, transcribing in one request", duration_ms // 60000)
        return await _do_transcribe(client, audio_path, api_key, episode_guid)

    # Episode too long — split into chunks and transcribe each
    logger.info(
        "Audio is %d min, splitting into %d-min chunks",
        duration_ms // 60000,
        TRANSCRIPTION_CHUNK_MS // 60000,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        chunks = split_audio(audio_path, TRANSCRIPTION_CHUNK_MS, Path(tmpdir))
        logger.info("Split into %d chunks", len(chunks))

        all_segments: list[TranscriptSegment] = []
        raw_responses: list[dict] = []

        for chunk_path, offset_ms in chunks:
            logger.info("Transcribing chunk at offset %d ms", offset_ms)
            transcript = await _do_transcribe(
                client, chunk_path, api_key, episode_guid
            )
            # Adjust timestamps to absolute positions
            for seg in transcript.segments:
                all_segments.append(TranscriptSegment(
                    speaker=seg.speaker,
                    text=seg.text,
                    start_ms=seg.start_ms + offset_ms,
                    end_ms=seg.end_ms + offset_ms,
                ))
            raw_responses.append(transcript.raw_response)

    all_segments.sort(key=lambda s: s.start_ms)
    return Transcript(
        episode_guid=episode_guid,
        segments=all_segments,
        raw_response={"chunked": True, "chunks": raw_responses},
    )


async def _do_transcribe(
    client: httpx.AsyncClient,
    audio_path: Path,
    api_key: str,
    episode_guid: str,
) -> Transcript:
    file_uri = await _upload_audio_to_gemini(audio_path, api_key, client)

    mime_type = mimetypes.guess_type(str(audio_path))[0] or "audio/mpeg"

    payload = {
        "contents": [
            {
                "parts": [
                    {"file_data": {"mime_type": mime_type, "file_uri": file_uri}},
                    {"text": TRANSCRIPTION_PROMPT},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 65536,
            "responseMimeType": "application/json",
        },
    }

    resp = await client.post(
        GEMINI_API_URL,
        params={"key": api_key},
        json=payload,
    )
    resp.raise_for_status()
    return parse_gemini_response(resp.json(), episode_guid)


async def _upload_audio_to_gemini(
    audio_path: Path,
    api_key: str,
    client: httpx.AsyncClient,
) -> str:
    """Upload audio file via Gemini Files API, return file URI."""
    mime_type = mimetypes.guess_type(str(audio_path))[0] or "audio/mpeg"
    file_size = audio_path.stat().st_size

    # Initiate resumable upload
    resp = await client.post(
        GEMINI_UPLOAD_URL,
        params={"key": api_key},
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
        },
        json={"file": {"display_name": audio_path.name}},
    )
    resp.raise_for_status()
    upload_url = resp.headers["X-Goog-Upload-URL"]

    # Upload file bytes in chunks to avoid loading entire file into memory
    offset = 0
    resp = None
    with open(audio_path, "rb") as f:
        while offset < file_size:
            chunk = f.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            is_last = offset + len(chunk) >= file_size
            command = "upload, finalize" if is_last else "upload"
            resp = await client.put(
                upload_url,
                headers={
                    "X-Goog-Upload-Command": command,
                    "X-Goog-Upload-Offset": str(offset),
                    "Content-Length": str(len(chunk)),
                    "Content-Type": mime_type,
                },
                content=chunk,
            )
            resp.raise_for_status()
            offset += len(chunk)

    result = resp.json()
    return result["file"]["uri"]


def parse_gemini_response(response: dict, episode_guid: str = "") -> Transcript:
    """Parse Gemini's response into our Transcript model."""
    # Navigate Gemini's response structure
    candidates = response.get("candidates", [])
    if not candidates:
        raise ValueError("No candidates in Gemini response")

    finish_reason = candidates[0].get("finishReason", "")
    if finish_reason == "MAX_TOKENS":
        raise ValueError(
            "Gemini response was truncated (MAX_TOKENS). "
            "The episode may be too long for a single transcription request."
        )

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    if not parts:
        raise ValueError("No parts in Gemini response")

    text = parts[0].get("text", "")

    # Strip markdown code fences if present
    text = _strip_code_fences(text)

    data = json.loads(text)
    segments_data = data.get("segments", data if isinstance(data, list) else [])

    segments = [
        TranscriptSegment(
            speaker=seg["speaker"],
            text=seg["text"],
            start_ms=seg["start_ms"],
            end_ms=seg["end_ms"],
        )
        for seg in segments_data
    ]

    return Transcript(
        episode_guid=episode_guid,
        segments=segments,
        raw_response=response,
    )


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from around JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line (```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


# --- Published transcript fetching ---

_TRANSCRIPT_URL_RE = re.compile(
    r"https?://lexfridman\.com/[\w-]+-transcript"
)

# Matches speaker labels like: Lex Fridman [(00:01:11)](url)
_SPEAKER_RE = re.compile(
    r"\*\*(.+?)\s*\[\((\d{1,2}:\d{2}(?::\d{2})?)\)\]"
)


def extract_transcript_url(description: str) -> str | None:
    """Extract the transcript page URL from an episode description."""
    m = _TRANSCRIPT_URL_RE.search(description)
    return m.group(0) if m else None


def _parse_timestamp_ms(ts: str) -> int:
    """Parse HH:MM:SS or MM:SS timestamp to milliseconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        return (int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])) * 1000
    if len(parts) == 2:
        return (int(parts[0]) * 60 + int(parts[1])) * 1000
    return 0


def _classify_speaker(name: str) -> str:
    """Map a display name like 'Lex Fridman' to 'lex' or 'guest'."""
    if "lex" in name.lower():
        return "lex"
    return "guest"


def parse_published_transcript(html: str, episode_guid: str = "") -> Transcript:
    """Parse the HTML/markdown of a published transcript page into a Transcript.

    The page uses a pattern like:
        **Lex Fridman [(00:00:00)](youtube_url)** transcript text here...
        **Jensen Huang [(00:01:11)](youtube_url)** response text...
    """
    segments: list[TranscriptSegment] = []

    # Find all speaker turns by their labeled timestamps
    matches = list(_SPEAKER_RE.finditer(html))

    for i, m in enumerate(matches):
        speaker_name = m.group(1).strip()
        timestamp = m.group(2)
        start_ms = _parse_timestamp_ms(timestamp)
        speaker = _classify_speaker(speaker_name)

        # Extract text: from after this match to the start of the next match
        text_start = m.end()
        # Skip past the closing **
        rest = html[text_start:]
        # Remove leading )](url)** pattern
        close = rest.find("**")
        if close != -1 and close < 200:
            rest = rest[close + 2:]
            text_start += close + 2

        if i + 1 < len(matches):
            text_end_abs = matches[i + 1].start()
            text = html[text_start:text_end_abs]
        else:
            text = rest

        # Clean up the text
        text = re.sub(r"\*\*", "", text)  # remove remaining bold markers
        text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # remove markdown links
        text = re.sub(r"<[^>]+>", "", text)  # remove HTML tags
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue

        # Estimate end_ms from next segment's start, or add 30s for last segment
        if i + 1 < len(matches):
            end_ms = _parse_timestamp_ms(matches[i + 1].group(2))
        else:
            end_ms = start_ms + 30000

        if end_ms <= start_ms:
            end_ms = start_ms + 5000

        segments.append(TranscriptSegment(
            speaker=speaker,
            text=text,
            start_ms=start_ms,
            end_ms=end_ms,
        ))

    return Transcript(
        episode_guid=episode_guid,
        segments=segments,
    )


async def fetch_published_transcript(
    description: str,
    episode_guid: str = "",
    client: httpx.AsyncClient | None = None,
) -> Transcript | None:
    """Fetch and parse a published transcript linked from the episode description.

    Returns None if no transcript link is found or fetching/parsing fails.
    """
    url = extract_transcript_url(description)
    if not url:
        logger.debug("No transcript URL found in description")
        return None

    logger.info("Fetching published transcript from %s", url)
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=60.0) as c:
                resp = await c.get(url)
        else:
            resp = await client.get(url)
        resp.raise_for_status()
        transcript = parse_published_transcript(resp.text, episode_guid)
        if transcript.segments:
            logger.info(
                "Parsed published transcript: %d segments, %s to %s",
                len(transcript.segments),
                transcript.segments[0].start_ms,
                transcript.segments[-1].end_ms,
            )
            return transcript
        logger.warning("Published transcript had no parseable segments")
        return None
    except Exception:
        logger.warning("Failed to fetch/parse published transcript", exc_info=True)
        return None
