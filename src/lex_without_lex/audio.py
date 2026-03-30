from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

from .models import EditList, SegmentAction

logger = logging.getLogger(__name__)


def _build_filter_graph(
    keep_segments: list[SegmentAction],
    interjection_paths: dict[int, Path],
) -> tuple[list[Path], str, str]:
    """Build FFmpeg complex filter graph for single-pass assembly.

    Returns:
        extra_inputs: interjection file Paths in input order (source is input 0)
        filter_complex: the -filter_complex string
        output_label: the label to -map (always "[out]")
    """
    filters: list[str] = []
    concat_labels: list[str] = []
    extra_inputs: list[Path] = []
    # Map interjection start_ms -> input index (starting from 1, since 0 is source)
    inj_input_idx: dict[int, int] = {}

    # Assign input indices to interjections in chronological order
    for seg in keep_segments:
        if seg.start_ms in interjection_paths and seg.start_ms not in inj_input_idx:
            idx = len(extra_inputs) + 1  # input 0 is source
            inj_input_idx[seg.start_ms] = idx
            extra_inputs.append(interjection_paths[seg.start_ms])

    # Build filter chains
    inj_counter = 0
    for i, seg in enumerate(keep_segments):
        # Interjection before this segment
        if seg.start_ms in inj_input_idx:
            inp_idx = inj_input_idx[seg.start_ms]
            inj_label = f"inj{inj_counter}"
            filters.append(
                f"[{inp_idx}:a]aformat=sample_rates=44100:channel_layouts=mono[{inj_label}]"
            )
            concat_labels.append(f"[{inj_label}]")
            inj_counter += 1

        # Segment from source
        start_s = seg.start_ms / 1000.0
        end_s = seg.end_ms / 1000.0
        seg_label = f"seg{i}"
        filters.append(
            f"[0:a]atrim=start={start_s:.3f}:end={end_s:.3f},"
            f"asetpts=PTS-STARTPTS,"
            f"aformat=sample_rates=44100:channel_layouts=mono[{seg_label}]"
        )
        concat_labels.append(f"[{seg_label}]")

    n = len(concat_labels)
    concat_line = f"{''.join(concat_labels)}concat=n={n}:v=0:a=1[out]"
    filters.append(concat_line)

    filter_complex = ";\n".join(filters)
    return extra_inputs, filter_complex, "[out]"


def _build_ffmpeg_command(
    source_path: Path,
    extra_inputs: list[Path],
    filter_complex: str,
    output_label: str,
    output_path: Path,
) -> list[str]:
    """Build the full ffmpeg command for complex filter assembly."""
    cmd = ["ffmpeg", "-y", "-i", str(source_path)]
    for inp in extra_inputs:
        cmd.extend(["-i", str(inp)])
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", output_label,
        "-acodec", "libmp3lame",
        str(output_path),
    ])
    return cmd


async def assemble_audio(
    source_path: Path,
    edit_list: EditList,
    interjection_paths: dict[int, Path],
    output_path: Path,
) -> Path:
    """Reconstruct audio based on edit list + interjections.

    Uses a single FFmpeg complex filter graph to extract, normalize, and
    concatenate all segments in one pass — no intermediate files.

    Runs ffmpeg in a thread pool to avoid blocking the async event loop.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keep_segments = sorted(
        [s for s in edit_list.segments if s.action == "keep"],
        key=lambda s: s.start_ms,
    )

    if not keep_segments:
        raise ValueError("No segments to keep in edit list")

    extra_inputs, filter_complex, output_label = _build_filter_graph(
        keep_segments, interjection_paths,
    )
    cmd = _build_ffmpeg_command(
        source_path, extra_inputs, filter_complex, output_label, output_path,
    )

    logger.info(
        "Assembling %d segments + %d interjections in single FFmpeg pass",
        len(keep_segments), len(interjection_paths),
    )
    await asyncio.to_thread(_run_ffmpeg, cmd)
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
