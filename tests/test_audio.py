import shutil
import subprocess
from pathlib import Path

import pytest

from lex_without_lex.audio import (
    _concat_files,
    _extract_segment,
    assemble_audio,
    build_concat_list,
)
from lex_without_lex.models import EditList, SegmentAction

has_ffmpeg = shutil.which("ffmpeg") is not None
skip_no_ffmpeg = pytest.mark.skipif(not has_ffmpeg, reason="ffmpeg not installed")


def _make_sine_wav(path: Path, duration: float = 2.0, freq: int = 440) -> Path:
    """Generate a sine wave WAV file using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency={freq}:duration={duration}",
            "-ar", "44100",
            "-ac", "1",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


def _get_duration(path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


class TestBuildConcatList:
    def test_ordering(self):
        segments = [
            (0, Path("seg0.mp3")),
            (5000, Path("seg1.mp3")),
            (15000, Path("seg2.mp3")),
        ]
        interjections = {
            5000: Path("inj0.mp3"),
            15000: Path("inj1.mp3"),
        }
        result = build_concat_list(segments, interjections)
        assert result == [
            Path("seg0.mp3"),
            Path("inj0.mp3"),
            Path("seg1.mp3"),
            Path("inj1.mp3"),
            Path("seg2.mp3"),
        ]

    def test_no_interjections(self):
        segments = [(0, Path("seg0.mp3")), (5000, Path("seg1.mp3"))]
        result = build_concat_list(segments, {})
        assert result == [Path("seg0.mp3"), Path("seg1.mp3")]


@skip_no_ffmpeg
class TestExtractSegment:
    def test_extracts_middle(self, tmp_path):
        source = _make_sine_wav(tmp_path / "source.wav", duration=5.0)
        output = tmp_path / "segment.mp3"
        _extract_segment(source, 1000, 3000, output)
        assert output.exists()
        duration = _get_duration(output)
        assert abs(duration - 2.0) < 0.2  # ~2 seconds with encoding tolerance


@skip_no_ffmpeg
class TestConcatFiles:
    def test_concatenates(self, tmp_path):
        f1 = _make_sine_wav(tmp_path / "a.wav", duration=2.0, freq=440)
        f2 = _make_sine_wav(tmp_path / "b.wav", duration=3.0, freq=880)
        output = tmp_path / "concat.mp3"
        _concat_files([f1, f2], output)
        assert output.exists()
        duration = _get_duration(output)
        assert abs(duration - 5.0) < 0.5


@skip_no_ffmpeg
class TestAssembleAudio:
    def test_full_assembly(self, tmp_path):
        # Create a 10-second source
        source = _make_sine_wav(tmp_path / "source.wav", duration=10.0)
        # Create a short interjection
        interjection = _make_sine_wav(tmp_path / "interjection.wav", duration=1.0, freq=880)

        edit_list = EditList(
            episode_guid="test",
            segments=[
                SegmentAction(action="cut", start_ms=0, end_ms=2000, speaker="lex"),
                SegmentAction(action="keep", start_ms=2000, end_ms=5000, speaker="guest"),
                SegmentAction(action="cut", start_ms=5000, end_ms=7000, speaker="lex"),
                SegmentAction(action="keep", start_ms=7000, end_ms=10000, speaker="guest"),
            ],
            interjections=[],
        )

        output = tmp_path / "output.mp3"
        result = assemble_audio(source, edit_list, {7000: interjection}, output)
        assert result.exists()
        # Should be ~3s (first keep) + ~1s (interjection) + ~3s (second keep) ≈ 7s
        duration = _get_duration(result)
        assert 5.5 < duration < 8.5

    def test_raises_on_empty_keep(self, tmp_path):
        source = _make_sine_wav(tmp_path / "source.wav", duration=5.0)
        edit_list = EditList(
            episode_guid="test",
            segments=[
                SegmentAction(action="cut", start_ms=0, end_ms=5000),
            ],
            interjections=[],
        )
        with pytest.raises(ValueError, match="No segments to keep"):
            assemble_audio(source, edit_list, {}, tmp_path / "out.mp3")
