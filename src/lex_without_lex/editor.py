from __future__ import annotations

import json
import re

import httpx

from .models import EditList, Interjection, SegmentAction, Transcript

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-opus-4-20250514"

EDITOR_SYSTEM_PROMPT = """\
You are an expert podcast editor. Your job is to edit a podcast transcript to remove the host (Lex Fridman) and keep only the guest's speech, creating a seamless listening experience.

Rules:
1. KEEP all guest speech segments. Mark them with action "keep".
2. CUT all of Lex's speech segments. Mark them with action "cut".
3. When cutting Lex's speech would make the guest's subsequent response confusing or lack context, create a SHORT interjection. The interjection should be a brief, neutral question or context-setting statement (e.g., "On the topic of consciousness..." or "When asked about the future of AI..."). Keep interjections under 15 words.
4. Do NOT create interjections for simple conversational exchanges where the guest's response is self-contained.
5. Interjections should be placed at the timestamp where Lex's cut segment ends (which is where the next guest segment begins).

Output ONLY valid JSON (no markdown fences) with this exact structure:
{
  "segments": [
    {"action": "keep", "start_ms": 6500, "end_ms": 9000, "speaker": "guest", "reason": "Guest introduction"},
    {"action": "cut", "start_ms": 0, "end_ms": 6500, "speaker": "lex", "reason": "Host introduction"}
  ],
  "interjections": [
    {"insert_after_ms": 12500, "text": "On the topic of consciousness...", "context": "Lex asked about consciousness, guest's answer needs context"}
  ],
  "summary": "Removed 3 Lex segments, kept 4 guest segments, added 1 interjection for context."
}
"""


async def generate_edit_list(
    transcript: Transcript,
    api_key: str,
    client: httpx.AsyncClient | None = None,
) -> EditList:
    """Send transcript to Claude Opus, get back edit decisions."""
    if client is None:
        async with httpx.AsyncClient(timeout=300.0) as c:
            return await _do_generate(c, transcript, api_key)
    return await _do_generate(client, transcript, api_key)


async def _do_generate(
    client: httpx.AsyncClient,
    transcript: Transcript,
    api_key: str,
) -> EditList:
    user_content = _build_user_prompt(transcript)

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 16384,
        "system": EDITOR_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}],
    }

    resp = await client.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
    )
    resp.raise_for_status()

    response_data = resp.json()
    response_text = response_data["content"][0]["text"]
    return parse_opus_response(response_text, transcript.episode_guid)


def _build_user_prompt(transcript: Transcript) -> str:
    """Build the user prompt containing the transcript."""
    lines = ["Here is the podcast transcript to edit:\n"]
    for seg in transcript.segments:
        lines.append(
            f"[{seg.start_ms}-{seg.end_ms}] {seg.speaker}: {seg.text}"
        )
    return "\n".join(lines)


def parse_opus_response(response_text: str, episode_guid: str = "") -> EditList:
    """Extract JSON from Opus response, validate into EditList."""
    text = _strip_code_fences(response_text)
    data = json.loads(text)

    segments = [
        SegmentAction(
            action=seg["action"],
            start_ms=seg["start_ms"],
            end_ms=seg["end_ms"],
            speaker=seg.get("speaker", ""),
            reason=seg.get("reason", ""),
        )
        for seg in data.get("segments", [])
    ]

    interjections = [
        Interjection(
            insert_after_ms=inj["insert_after_ms"],
            text=inj["text"],
            context=inj.get("context", ""),
        )
        for inj in data.get("interjections", [])
    ]

    return EditList(
        episode_guid=episode_guid,
        segments=segments,
        interjections=interjections,
        summary=data.get("summary", ""),
        raw_response=response_text,
    )


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from around JSON."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def validate_edit_list(edit_list: EditList, transcript: Transcript) -> list[str]:
    """Validate edit list against transcript. Returns list of warnings."""
    warnings: list[str] = []

    if not transcript.segments:
        return warnings

    episode_end = max(seg.end_ms for seg in transcript.segments)

    # Check segment bounds
    for seg in edit_list.segments:
        if seg.start_ms < 0:
            warnings.append(f"Segment starts before 0: {seg.start_ms}")
        if seg.end_ms > episode_end:
            warnings.append(
                f"Segment ends after episode ({seg.end_ms} > {episode_end})"
            )
        if seg.start_ms >= seg.end_ms:
            warnings.append(
                f"Segment has zero or negative duration: {seg.start_ms}-{seg.end_ms}"
            )

    # Check for overlapping segments
    sorted_segs = sorted(edit_list.segments, key=lambda s: s.start_ms)
    for i in range(1, len(sorted_segs)):
        if sorted_segs[i].start_ms < sorted_segs[i - 1].end_ms:
            warnings.append(
                f"Overlapping segments: [{sorted_segs[i-1].start_ms}-{sorted_segs[i-1].end_ms}] "
                f"and [{sorted_segs[i].start_ms}-{sorted_segs[i].end_ms}]"
            )

    # Check interjections are at valid points (not inside cut regions)
    cut_regions = [
        (seg.start_ms, seg.end_ms)
        for seg in edit_list.segments
        if seg.action == "cut"
    ]
    for inj in edit_list.interjections:
        for cut_start, cut_end in cut_regions:
            if cut_start < inj.insert_after_ms < cut_end:
                warnings.append(
                    f"Interjection at {inj.insert_after_ms}ms is inside cut region "
                    f"[{cut_start}-{cut_end}]"
                )
        if inj.insert_after_ms > episode_end:
            warnings.append(
                f"Interjection at {inj.insert_after_ms}ms is after episode end ({episode_end})"
            )

    return warnings
