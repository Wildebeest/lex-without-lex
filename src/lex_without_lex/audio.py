from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

from .models import EditList

logger = logging.getLogger(__name__)


async def assemble_audio(
    source_path: Path,
    edit_list: EditList,
    interjection_paths: dict[int, Path],
    output_path: Path,
) -> Path:
    """Reconstruct audio based on edit list + interjections.

    1. Extract each "keep" segment from source
    2. Interleave interjection audio at the right points
    3. Concatenate all pieces

    Runs ffmpeg in a thread pool to avoid blocking the async event loop.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keep_segments = sorted(
        [s for s in edit_list.segments if s.action == "keep"],
        key=lambda s: s.start_ms,
    )

    if not keep_segments:
        raise ValueError("No segments to keep in edit list")

    logger.info("Assembling %d keep segments with %d interjections",
                len(keep_segments), len(interjection_paths))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pieces: list[Path] = []

        for i, seg in enumerate(keep_segments):
            # Check if an interjection should be inserted before this segment
            if seg.start_ms in interjection_paths:
                inj_path = interjection_paths[seg.start_ms]
                normalized = tmp / f"interjection_{i}.mp3"
                await asyncio.to_thread(_normalize_audio, inj_path, normalized)
                pieces.append(normalized)

            seg_path = tmp / f"segment_{i}.mp3"
            await asyncio.to_thread(
                _extract_segment, source_path, seg.start_ms, seg.end_ms, seg_path
            )
            pieces.append(seg_path)

            if (i + 1) % 50 == 0:
                logger.info("Extracted %d/%d segments", i + 1, len(keep_segments))

        logger.info("Extracted all %d segments, concatenating...", len(pieces))
        await asyncio.to_thread(_concat_files, pieces, output_path)

    logger.info("Assembly complete: %s", output_path.name)
    return output_path


def _extract_segment(
    source: Path, start_ms: int, end_ms: int, output: Path
) -> Path:
    """Extract a time segment from source audio using ffmpeg."""
    start_sec = start_ms / 1000.0
    duration_sec = (end_ms - start_ms) / 1000.0

    _run_ffmpeg([
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(source),
        "-t", f"{duration_sec:.3f}",
        "-acodec", "libmp3lame",
        "-ar", "44100",
        "-ac", "1",
        str(output),
    ])
    return output


def _normalize_audio(source: Path, output: Path) -> Path:
    """Re-encode audio to consistent format (44.1kHz mono mp3)."""
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(source),
        "-acodec", "libmp3lame",
        "-ar", "44100",
        "-ac", "1",
        str(output),
    ])
    return output


def _concat_files(file_list: list[Path], output: Path) -> Path:
    """Concatenate audio files using ffmpeg concat demuxer."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_file = Path(f.name)
        for fp in file_list:
            f.write(f"file '{fp}'\n")

    try:
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-acodec", "libmp3lame",
            "-ar", "44100",
            "-ac", "1",
            str(output),
        ])
    finally:
        concat_file.unlink(missing_ok=True)

    return output


def build_concat_list(
    keep_segments: list[tuple[int, Path]],
    interjection_paths: dict[int, Path],
) -> list[Path]:
    """Order segments and interjections chronologically.

    keep_segments: list of (start_ms, file_path) tuples, sorted by start_ms
    interjection_paths: mapping of insert_after_ms -> audio file path
    """
    result: list[Path] = []
    for start_ms, seg_path in keep_segments:
        if start_ms in interjection_paths:
            result.append(interjection_paths[start_ms])
        result.append(seg_path)
    return result


def get_audio_duration_ms(source: Path) -> int:
    """Return duration of audio file in milliseconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return int(float(result.stdout.strip()) * 1000)


def split_audio(
    source: Path, chunk_duration_ms: int, output_dir: Path
) -> list[tuple[Path, int]]:
    """Split audio into chunks. Returns list of (chunk_path, offset_ms) pairs."""
    total_ms = get_audio_duration_ms(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[tuple[Path, int]] = []
    offset = 0
    i = 0
    while offset < total_ms:
        chunk_path = output_dir / f"chunk_{i}.mp3"
        duration = min(chunk_duration_ms, total_ms - offset)
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-ss", f"{offset / 1000.0:.3f}",
            "-i", str(source),
            "-t", f"{duration / 1000.0:.3f}",
            "-acodec", "libmp3lame",
            "-ar", "44100",
            "-ac", "1",
            str(chunk_path),
        ])
        chunks.append((chunk_path, offset))
        offset += chunk_duration_ms
        i += 1

    return chunks


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    """Run ffmpeg subprocess, raise on error."""
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    return result
