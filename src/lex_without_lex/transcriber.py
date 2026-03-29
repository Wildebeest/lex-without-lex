from __future__ import annotations

import json
import mimetypes
from pathlib import Path

import httpx

from .models import Transcript, TranscriptSegment

GEMINI_UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB

TRANSCRIPTION_PROMPT = """\
You are a professional podcast transcription service. Transcribe the provided audio with detailed speaker diarization and precise timestamps.

Instructions:
1. Identify speakers. The host is Lex Fridman — label him as "lex". Label the guest as "guest". If there are multiple guests, label them as "guest1", "guest2", etc.
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
    """Upload audio to Gemini and get back a diarized transcript."""
    if client is None:
        async with httpx.AsyncClient(timeout=600.0) as c:
            return await _do_transcribe(c, audio_path, api_key, episode_guid)
    return await _do_transcribe(client, audio_path, api_key, episode_guid)


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
