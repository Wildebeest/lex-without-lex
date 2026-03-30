"""Parse chapter timestamps from episode descriptions and remap to edited timeline."""

from __future__ import annotations

import json
import re

from .models import Chapter, EditList


def parse_chapters_from_description(description: str) -> list[Chapter]:
    """Extract chapter markers from Lex Fridman's episode descriptions.

    Matches patterns like:
        (0:00) – Introduction
        (01:23:45) – Deep topic
        (12:34) - Another topic
    """
    pattern = r"\((\d{1,2}:\d{2}(?::\d{2})?)\)\s*[-–—]\s*([^<\n]+)"
    chapters: list[Chapter] = []
    for m in re.finditer(pattern, description):
        ts = m.group(1).strip()
        title = m.group(2).strip()
        # Strip trailing HTML tags or whitespace
        title = re.sub(r"<[^>]+>$", "", title).strip()
        seconds = _parse_timestamp(ts)
        if seconds is not None:
            chapters.append(Chapter(start_seconds=seconds, title=title))
    return chapters


def _parse_timestamp(ts: str) -> float | None:
    """Parse HH:MM:SS or MM:SS timestamp to seconds."""
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        return None
    return None


def remap_chapters(
    chapters: list[Chapter], edit_list: EditList
) -> list[Chapter]:
    """Remap original chapter timestamps to the edited audio timeline.

    Walks through the edit list segments in order. For each "keep" segment,
    the output time advances by the segment duration. For "cut" segments,
    the output time stays the same but the source time advances.

    Each chapter is placed at the output time corresponding to its original
    source timestamp. Chapters that fall entirely within cut regions are
    mapped to the output time at the start of the next kept segment.
    """
    segments = sorted(edit_list.segments, key=lambda s: s.start_ms)

    # Build a mapping from source_ms -> output_ms
    # by walking through segments and tracking accumulated output time
    output_ms = 0
    source_to_output: list[tuple[int, int, int, int]] = []  # (src_start, src_end, out_start, out_end)

    for seg in segments:
        if seg.action == "keep":
            seg_duration = seg.end_ms - seg.start_ms
            source_to_output.append((seg.start_ms, seg.end_ms, output_ms, output_ms + seg_duration))
            output_ms += seg_duration

    remapped: list[Chapter] = []
    for ch in chapters:
        ch_ms = ch.start_seconds * 1000
        out = _map_source_to_output(ch_ms, source_to_output)
        if out is not None:
            remapped.append(Chapter(start_seconds=round(out / 1000, 1), title=ch.title))

    # Deduplicate chapters that map to the same output time
    seen_times: set[float] = set()
    deduped: list[Chapter] = []
    for ch in remapped:
        if ch.start_seconds not in seen_times:
            seen_times.add(ch.start_seconds)
            deduped.append(ch)

    return deduped


def _map_source_to_output(
    source_ms: float,
    source_to_output: list[tuple[int, int, int, int]],
) -> float | None:
    """Map a source timestamp to the corresponding output timestamp."""
    if not source_to_output:
        return None

    for src_start, src_end, out_start, out_end in source_to_output:
        if src_start <= source_ms < src_end:
            # Chapter falls within this kept segment
            offset = source_ms - src_start
            return out_start + offset

    # Chapter is in a cut region — map to the start of the next kept segment
    for src_start, _src_end, out_start, _out_end in source_to_output:
        if src_start > source_ms:
            return out_start

    # Past all segments — map to end of last segment
    if source_to_output:
        return source_to_output[-1][3]
    return None


def chapters_to_json(chapters: list[Chapter]) -> str:
    """Produce Podcast Namespace JSON Chapters format."""
    return json.dumps(
        {
            "version": "1.2.0",
            "chapters": [
                {"startTime": ch.start_seconds, "title": ch.title}
                for ch in chapters
            ],
        },
        indent=2,
    )
