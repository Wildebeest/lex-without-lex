import json

from lex_without_lex.chapters import (
    chapters_to_json,
    parse_chapters_from_description,
    remap_chapters,
)
from lex_without_lex.models import Chapter, EditList, SegmentAction


class TestParseChaptersFromDescription:
    def test_parses_mm_ss_format(self):
        desc = "(0:00) – Introduction\n(12:34) – Topic one\n(45:00) – Wrap up"
        chapters = parse_chapters_from_description(desc)
        assert len(chapters) == 3
        assert chapters[0].title == "Introduction"
        assert chapters[0].start_seconds == 0
        assert chapters[1].start_seconds == 12 * 60 + 34
        assert chapters[2].start_seconds == 45 * 60

    def test_parses_hh_mm_ss_format(self):
        desc = "(1:05:20) – Deep topic\n(2:30:00) – Closing"
        chapters = parse_chapters_from_description(desc)
        assert len(chapters) == 2
        assert chapters[0].start_seconds == 3920  # 1*3600 + 5*60 + 20
        assert chapters[1].start_seconds == 9000  # 2*3600 + 30*60

    def test_handles_dash_variants(self):
        desc = "(0:00) - Intro\n(5:00) – Middle\n(10:00) — End"
        chapters = parse_chapters_from_description(desc)
        assert len(chapters) == 3

    def test_no_chapters_returns_empty(self):
        desc = "Just a regular description with no timestamps."
        chapters = parse_chapters_from_description(desc)
        assert chapters == []

    def test_real_lex_format(self):
        desc = """<p>Jensen Huang is co-founder and CEO of NVIDIA.</p>
<p>OUTLINE:</p>
<p>(0:00) – Introduction<br/>
(02:24) – Sponsors, Comments, and Reflections<br/>
(10:47) – Early games: Pac-Man, Zork, Doom, Quake<br/>
(1:23:45) – AI and the future</p>"""
        chapters = parse_chapters_from_description(desc)
        assert len(chapters) == 4
        assert chapters[0].title == "Introduction"
        assert chapters[1].start_seconds == 144  # 2*60+24
        assert chapters[2].title == "Early games: Pac-Man, Zork, Doom, Quake"
        assert chapters[3].start_seconds == 1 * 3600 + 23 * 60 + 45

    def test_strips_trailing_html(self):
        desc = "(0:00) – Introduction<br/>\n(5:00) – Topic<br/>"
        chapters = parse_chapters_from_description(desc)
        assert chapters[0].title == "Introduction"
        assert chapters[1].title == "Topic"


class TestRemapChapters:
    def _make_edit_list(self, segments):
        return EditList(
            episode_guid="test",
            segments=[
                SegmentAction(action=a, start_ms=s, end_ms=e)
                for a, s, e in segments
            ],
            interjections=[],
        )

    def test_identity_remap_all_kept(self):
        """When all segments are kept, chapters stay at original timestamps."""
        edit_list = self._make_edit_list([
            ("keep", 0, 60000),
        ])
        chapters = [Chapter(start_seconds=0, title="Start"), Chapter(start_seconds=30, title="Middle")]
        result = remap_chapters(chapters, edit_list)
        assert len(result) == 2
        assert result[0].start_seconds == 0
        assert result[1].start_seconds == 30

    def test_chapter_after_cut(self):
        """Chapter after a cut region should shift earlier."""
        edit_list = self._make_edit_list([
            ("keep", 0, 10000),      # 0-10s kept
            ("cut", 10000, 30000),   # 10-30s cut (20s removed)
            ("keep", 30000, 60000),  # 30-60s kept
        ])
        # Chapter at 40s (in second keep segment) should map to 20s output
        # (10s from first keep + 10s into second keep)
        chapters = [Chapter(start_seconds=0, title="Start"), Chapter(start_seconds=40, title="After cut")]
        result = remap_chapters(chapters, edit_list)
        assert result[0].start_seconds == 0
        assert result[1].start_seconds == 20  # 10 + (40-30)

    def test_chapter_in_cut_region(self):
        """Chapter in a cut region maps to start of next kept segment."""
        edit_list = self._make_edit_list([
            ("keep", 0, 10000),
            ("cut", 10000, 30000),
            ("keep", 30000, 60000),
        ])
        chapters = [Chapter(start_seconds=20, title="In cut")]
        result = remap_chapters(chapters, edit_list)
        assert len(result) == 1
        assert result[0].start_seconds == 10  # maps to start of second keep

    def test_empty_chapters(self):
        edit_list = self._make_edit_list([("keep", 0, 60000)])
        result = remap_chapters([], edit_list)
        assert result == []

    def test_deduplicates_same_output_time(self):
        """Two chapters mapping to same output time should be deduplicated."""
        edit_list = self._make_edit_list([
            ("keep", 0, 5000),
            ("cut", 5000, 30000),
            ("keep", 30000, 60000),
        ])
        # Both at 10s and 20s are in cut region, both map to 5s
        chapters = [
            Chapter(start_seconds=10, title="First"),
            Chapter(start_seconds=20, title="Second"),
        ]
        result = remap_chapters(chapters, edit_list)
        assert len(result) == 1


class TestChaptersToJson:
    def test_produces_valid_json(self):
        chapters = [
            Chapter(start_seconds=0, title="Intro"),
            Chapter(start_seconds=120.5, title="Topic"),
        ]
        result = chapters_to_json(chapters)
        data = json.loads(result)
        assert data["version"] == "1.2.0"
        assert len(data["chapters"]) == 2
        assert data["chapters"][0]["startTime"] == 0
        assert data["chapters"][0]["title"] == "Intro"
        assert data["chapters"][1]["startTime"] == 120.5

    def test_empty_chapters(self):
        result = chapters_to_json([])
        data = json.loads(result)
        assert data["chapters"] == []
