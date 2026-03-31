from __future__ import annotations

import json
import re

import httpx

from .models import EditList, Interjection, SegmentAction, Transcript

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-opus-4-20250514"

EDITOR_SYSTEM_PROMPT = """\
You are an expert podcast editor. Your job is to edit a podcast transcript to remove the host (Lex Fridman) and keep only the guest's speech, creating a seamless listening experience.

Goal: MAXIMUM guest speech, MINIMUM synthetic interjections, ZERO host audio.

Rules:
1. KEEP all guest speech segments. Mark them with action "keep".
2. CUT all of Lex's speech segments. This includes:
   - Solo intro monologues and episode introductions
   - Sponsor reads and advertisements
   - Personal reflections, comments, and anecdotes by the host
   - ALL host speech before the guest's first appearance
   - Questions, follow-ups, and conversational turns by Lex during the interview
3. The edited audio MUST start with the guest's first utterance. Cut EVERYTHING before it — intro, sponsors, reflections, all of it.
4. If speaker labels seem wrong but content is clearly host speech (e.g., "Welcome to the Lex Fridman podcast", sponsor reads, first-person monologue before guest appears), treat it as Lex and CUT it.
5. If an episode outline is provided, use it to identify non-conversation sections. Sections labeled "Introduction", "Sponsors", "Comments", "Reflections", or similar should be cut entirely.
6. If a published transcript reference is provided, content present in the audio transcript but absent from the published transcript is likely sponsors or filler — cut it.
7. Create interjections ONLY when the guest's response would be completely incomprehensible without context. Most exchanges do NOT need interjections — guests often restate the topic naturally. When in doubt, do NOT create an interjection. Aim for fewer than 1 interjection per 15 minutes of kept audio. Keep interjections under 15 words.
8. Interjections should be placed at the timestamp where Lex's cut segment ends (which is where the next guest segment begins).

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
    episode_description: str = "",
    published_transcript: Transcript | None = None,
) -> EditList:
    """Send transcript to Claude Opus, get back edit decisions."""
    if client is None:
        async with httpx.AsyncClient(timeout=300.0) as c:
            return await _do_generate(
                c, transcript, api_key, episode_description, published_transcript
            )
    return await _do_generate(
        client, transcript, api_key, episode_description, published_transcript
    )


async def _do_generate(
    client: httpx.AsyncClient,
    transcript: Transcript,
    api_key: str,
    episode_description: str = "",
    published_transcript: Transcript | None = None,
) -> EditList:
    user_content = _build_user_prompt(
        transcript, episode_description, published_transcript
    )

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


def _extract_outline(description: str) -> str | None:
    """Extract the OUTLINE section from an episode description."""
    # Look for the outline section in the description
    outline_match = re.search(
        r"OUTLINE:?\s*</?\w[^>]*>?\s*(.*?)(?:</?p>|<b>|\Z)",
        description,
        re.DOTALL | re.IGNORECASE,
    )
    if not outline_match:
        return None
    raw = outline_match.group(1).strip()
    if not raw:
        return None
    # Clean HTML tags, decode entities
    raw = re.sub(r"<br\s*/?>", "\n", raw)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = raw.replace("&#8211;", "–").replace("&#8217;", "'").replace("&amp;", "&")
    lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
    return "\n".join(lines) if lines else None


def _build_user_prompt(
    transcript: Transcript,
    episode_description: str = "",
    published_transcript: Transcript | None = None,
) -> str:
    """Build the user prompt containing the transcript and optional context."""
    sections: list[str] = []

    # Include episode outline if available
    if episode_description:
        outline = _extract_outline(episode_description)
        if outline:
            sections.append(
                "EPISODE OUTLINE (use timestamps to identify intro/sponsor "
                "sections that should be CUT entirely):\n" + outline
            )

    # Include published transcript summary if available
    if published_transcript and published_transcript.segments:
        first = published_transcript.segments[0]
        sections.append(
            "PUBLISHED TRANSCRIPT REFERENCE:\n"
            f"The official published transcript starts at {first.start_ms}ms "
            f"with speaker '{first.speaker}'. Content present in the audio "
            "but absent from the published transcript (e.g., sponsor reads, "
            "personal reflections) should be CUT.\n"
            f"First line: [{first.start_ms}-{first.end_ms}] "
            f"{first.speaker}: {first.text[:200]}"
        )

    sections.append("AUDIO TRANSCRIPT TO EDIT:")
    for seg in transcript.segments:
        sections.append(
            f"[{seg.start_ms}-{seg.end_ms}] {seg.speaker}: {seg.text}"
        )
    return "\n\n".join(sections)


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

    # Semantic check: first kept segment should be from the guest
    kept = sorted(
        [s for s in edit_list.segments if s.action == "keep"],
        key=lambda s: s.start_ms,
    )
    if kept and kept[0].speaker == "lex":
        warnings.append(
            "First kept segment is from Lex — the edit should start with guest speech"
        )

    # Semantic check: no Lex segments should be kept
    lex_kept = [
        s for s in edit_list.segments if s.action == "keep" and s.speaker == "lex"
    ]
    if lex_kept:
        warnings.append(
            f"{len(lex_kept)} Lex segment(s) marked as 'keep' — all Lex speech should be cut"
        )

    # Semantic check: everything before guest's first transcript segment should be cut
    guest_transcript_segs = [
        s for s in transcript.segments if s.speaker == "guest"
    ]
    if guest_transcript_segs:
        first_guest_ms = guest_transcript_segs[0].start_ms
        intro_kept = [
            s for s in edit_list.segments
            if s.action == "keep" and s.end_ms <= first_guest_ms
        ]
        if intro_kept:
            warnings.append(
                f"Pre-guest intro not fully cut: {len(intro_kept)} segment(s) "
                f"kept before first guest speech at {first_guest_ms}ms"
            )

    # Semantic check: interjection count relative to kept segments
    if kept and len(edit_list.interjections) > len(kept) / 5:
        warnings.append(
            f"High interjection count ({len(edit_list.interjections)}) "
            f"relative to kept segments ({len(kept)})"
        )

    return warnings
