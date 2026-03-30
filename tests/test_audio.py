import shutil
import subprocess
from pathlib import Path

import pytest

from lex_without_lex.audio import (
    _build_ffmpeg_command,
    _build_filter_graph,
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


class TestBuildFilterGraph:
    def test_single_segment_no_interjections(self):
        segments = [SegmentAction(action="keep", start_ms=2000, end_ms=5000)]
        extra_inputs, filter_complex, output_label = _build_filter_graph(segments, {})

        assert extra_inputs == []
        assert "atrim=start=2.000:end=5.000" in filter_complex
        assert "asetpts=PTS-STARTPTS" in filter_complex
        assert "concat=n=1:v=0:a=1" in filter_complex
        assert output_label == "[out]"

    def test_multiple_segments_no_interjections(self):
        segments = [
            SegmentAction(action="keep", start_ms=0, end_ms=2000),
            SegmentAction(action="keep", start_ms=5000, end_ms=8000),
            SegmentAction(action="keep", start_ms=10000, end_ms=12000),
        ]
        extra_inputs, filter_complex, output_label = _build_filter_graph(segments, {})

        assert extra_inputs == []
        assert "atrim=start=0.000:end=2.000" in filter_complex
        assert "atrim=start=5.000:end=8.000" in filter_complex
        assert "atrim=start=10.000:end=12.000" in filter_complex
        assert "concat=n=3:v=0:a=1" in filter_complex
        # Segments should appear in order in concat
        assert "[seg0][seg1][seg2]concat" in filter_complex

    def test_segments_with_interjections(self):
        segments = [
            SegmentAction(action="keep", start_ms=2000, end_ms=5000),
            SegmentAction(action="keep", start_ms=7000, end_ms=10000),
        ]
        inj_path = Path("/tmp/inj.mp3")
        extra_inputs, filter_complex, output_label = _build_filter_graph(
            segments, {7000: inj_path},
        )

        assert extra_inputs == [inj_path]
        # Interjection uses input index 1
        assert "[1:a]" in filter_complex
        # Concat order: seg0, then interjection before seg1
        assert "[seg0][inj0][seg1]concat=n=3:v=0:a=1" in filter_complex

    def test_multiple_interjections_input_ordering(self):
        segments = [
            SegmentAction(action="keep", start_ms=0, end_ms=3000),
            SegmentAction(action="keep", start_ms=5000, end_ms=8000),
            SegmentAction(action="keep", start_ms=10000, end_ms=13000),
        ]
        inj0 = Path("/tmp/inj_a.mp3")
        inj1 = Path("/tmp/inj_b.mp3")
        extra_inputs, filter_complex, output_label = _build_filter_graph(
            segments, {0: inj0, 10000: inj1},
        )

        # Two interjection inputs, in chronological order
        assert extra_inputs == [inj0, inj1]
        assert "[1:a]" in filter_complex
        assert "[2:a]" in filter_complex
        # Concat: inj0 before seg0, seg1 standalone, inj1 before seg2
        assert "[inj0][seg0][seg1][inj1][seg2]concat=n=5:v=0:a=1" in filter_complex

    def test_millisecond_precision(self):
        segments = [SegmentAction(action="keep", start_ms=1500, end_ms=3750)]
        _, filter_complex, _ = _build_filter_graph(segments, {})

        assert "atrim=start=1.500:end=3.750" in filter_complex


class TestBuildFfmpegCommand:
    def test_command_structure(self):
        source = Path("/tmp/source.mp3")
        extra = [Path("/tmp/inj0.mp3"), Path("/tmp/inj1.mp3")]
        cmd = _build_ffmpeg_command(
            source, extra, "FILTER", "[out]", Path("/tmp/out.mp3"),
        )

        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd
        # Source is first input
        idx_first_i = cmd.index("-i")
        assert cmd[idx_first_i + 1] == str(source)
        # Extra inputs follow
        assert str(extra[0]) in cmd
        assert str(extra[1]) in cmd
        # Filter complex
        fc_idx = cmd.index("-filter_complex")
        assert cmd[fc_idx + 1] == "FILTER"
        # Map and codec
        map_idx = cmd.index("-map")
        assert cmd[map_idx + 1] == "[out]"
        assert "-acodec" in cmd
        assert "libmp3lame" in cmd
        assert cmd[-1] == "/tmp/out.mp3"

    def test_no_extra_inputs(self):
        cmd = _build_ffmpeg_command(
            Path("/tmp/src.mp3"), [], "F", "[out]", Path("/tmp/o.mp3"),
        )
        # Only one -i flag
        assert cmd.count("-i") == 1


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
    async def test_full_assembly(self, tmp_path):
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
        result = await assemble_audio(source, edit_list, {7000: interjection}, output)
        assert result.exists()
        # Should be ~3s (first keep) + ~1s (interjection) + ~3s (second keep) ≈ 7s
        duration = _get_duration(result)
        assert 5.5 < duration < 8.5

    async def test_raises_on_empty_keep(self, tmp_path):
        source = _make_sine_wav(tmp_path / "source.wav", duration=5.0)
        edit_list = EditList(
            episode_guid="test",
            segments=[
                SegmentAction(action="cut", start_ms=0, end_ms=5000),
            ],
            interjections=[],
        )
        with pytest.raises(ValueError, match="No segments to keep"):
            await assemble_audio(source, edit_list, {}, tmp_path / "out.mp3")

    async def test_single_segment_no_interjections(self, tmp_path):
        source = _make_sine_wav(tmp_path / "source.wav", duration=5.0)
        edit_list = EditList(
            episode_guid="test",
            segments=[
                SegmentAction(action="keep", start_ms=1000, end_ms=4000),
            ],
            interjections=[],
        )
        output = tmp_path / "output.mp3"
        result = await assemble_audio(source, edit_list, {}, output)
        assert result.exists()
        duration = _get_duration(result)
        assert abs(duration - 3.0) < 0.5

    async def test_many_segments(self, tmp_path):
        source = _make_sine_wav(tmp_path / "source.wav", duration=10.0)
        segments = [
            SegmentAction(action="keep", start_ms=i * 2000, end_ms=i * 2000 + 1000)
            for i in range(5)
        ]
        edit_list = EditList(
            episode_guid="test", segments=segments, interjections=[],
        )
        output = tmp_path / "output.mp3"
        result = await assemble_audio(source, edit_list, {}, output)
        assert result.exists()
        # 5 segments of ~1s each
        duration = _get_duration(result)
        assert 3.5 < duration < 6.5

    async def test_interjection_at_first_segment(self, tmp_path):
        source = _make_sine_wav(tmp_path / "source.wav", duration=5.0)
        interjection = _make_sine_wav(tmp_path / "inj.wav", duration=1.0, freq=880)
        edit_list = EditList(
            episode_guid="test",
            segments=[
                SegmentAction(action="keep", start_ms=0, end_ms=3000),
                SegmentAction(action="keep", start_ms=3000, end_ms=5000),
            ],
            interjections=[],
        )
        output = tmp_path / "output.mp3"
        result = await assemble_audio(source, edit_list, {0: interjection}, output)
        assert result.exists()
        # ~1s (interjection) + ~3s + ~2s = ~6s
        duration = _get_duration(result)
        assert 4.5 < duration < 7.5
