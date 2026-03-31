import json

import httpx
import pytest
import respx

from lex_without_lex.editor import (
    ANTHROPIC_API_URL,
    _build_user_prompt,
    _extract_outline,
    _strip_code_fences,
    generate_edit_list,
    parse_opus_response,
    validate_edit_list,
)
from lex_without_lex.models import (
    EditList,
    Interjection,
    SegmentAction,
    Transcript,
    TranscriptSegment,
)


@pytest.fixture
def sample_edit_json(fixtures_dir):
    return (fixtures_dir / "sample_edit_list.json").read_text()


@pytest.fixture
def sample_transcript():
    return Transcript(
        episode_guid="ep-001",
        segments=[
            TranscriptSegment(speaker="lex", text="Welcome.", start_ms=0, end_ms=6500),
            TranscriptSegment(speaker="guest", text="Thanks.", start_ms=6500, end_ms=9000),
            TranscriptSegment(speaker="lex", text="What is consciousness?", start_ms=9000, end_ms=12500),
            TranscriptSegment(speaker="guest", text="It's about information.", start_ms=12500, end_ms=25000),
            TranscriptSegment(speaker="lex", text="Conscious AI?", start_ms=25000, end_ms=28000),
            TranscriptSegment(speaker="guest", text="Possible but careful.", start_ms=28000, end_ms=48000),
            TranscriptSegment(speaker="lex", text="Your transformer paper?", start_ms=48000, end_ms=53000),
            TranscriptSegment(speaker="guest", text="Attention is routing.", start_ms=53000, end_ms=70000),
        ],
    )


class TestParseOpusResponse:
    def test_basic_parsing(self, sample_edit_json):
        edit_list = parse_opus_response(sample_edit_json, "ep-001")
        assert edit_list.episode_guid == "ep-001"
        assert len(edit_list.segments) == 8
        assert len(edit_list.interjections) == 3

    def test_segment_actions(self, sample_edit_json):
        edit_list = parse_opus_response(sample_edit_json)
        actions = [s.action for s in edit_list.segments]
        assert actions.count("keep") == 4
        assert actions.count("cut") == 4

    def test_interjection_text(self, sample_edit_json):
        edit_list = parse_opus_response(sample_edit_json)
        texts = [i.text for i in edit_list.interjections]
        assert any("consciousness" in t.lower() for t in texts)

    def test_preserves_raw_response(self, sample_edit_json):
        edit_list = parse_opus_response(sample_edit_json)
        assert edit_list.raw_response == sample_edit_json

    def test_with_code_fences(self, sample_edit_json):
        fenced = f"```json\n{sample_edit_json}\n```"
        edit_list = parse_opus_response(fenced, "ep-001")
        assert len(edit_list.segments) == 8

    def test_summary(self, sample_edit_json):
        edit_list = parse_opus_response(sample_edit_json)
        assert "Removed" in edit_list.summary


class TestStripCodeFences:
    def test_strips_json(self):
        assert _strip_code_fences('```json\n{"a":1}\n```') == '{"a":1}'

    def test_no_fences(self):
        assert _strip_code_fences('{"a":1}') == '{"a":1}'


class TestValidateEditList:
    def test_valid_edit_list_structural(self, sample_edit_json, sample_transcript):
        """The sample edit list passes structural checks (bounds, overlaps, etc)."""
        edit_list = parse_opus_response(sample_edit_json, "ep-001")
        warnings = validate_edit_list(edit_list, sample_transcript)
        # The sample fixture has 3 interjections for 4 kept segments, which
        # triggers the high-interjection warning. Filter to structural only.
        structural = [
            w for w in warnings
            if not w.startswith("High interjection")
        ]
        assert structural == []

    def test_overlapping_segments(self, sample_transcript):
        edit_list = EditList(
            episode_guid="ep-001",
            segments=[
                SegmentAction(action="keep", start_ms=0, end_ms=5000),
                SegmentAction(action="keep", start_ms=4000, end_ms=8000),
            ],
            interjections=[],
        )
        warnings = validate_edit_list(edit_list, sample_transcript)
        assert any("Overlapping" in w for w in warnings)

    def test_out_of_bounds(self, sample_transcript):
        edit_list = EditList(
            episode_guid="ep-001",
            segments=[
                SegmentAction(action="keep", start_ms=0, end_ms=100000),
            ],
            interjections=[],
        )
        warnings = validate_edit_list(edit_list, sample_transcript)
        assert any("after episode" in w for w in warnings)

    def test_zero_duration_segment(self, sample_transcript):
        edit_list = EditList(
            episode_guid="ep-001",
            segments=[
                SegmentAction(action="keep", start_ms=5000, end_ms=5000),
            ],
            interjections=[],
        )
        warnings = validate_edit_list(edit_list, sample_transcript)
        assert any("zero or negative" in w for w in warnings)

    def test_interjection_in_cut_region(self, sample_transcript):
        edit_list = EditList(
            episode_guid="ep-001",
            segments=[
                SegmentAction(action="cut", start_ms=0, end_ms=10000),
                SegmentAction(action="keep", start_ms=10000, end_ms=70000),
            ],
            interjections=[
                Interjection(insert_after_ms=5000, text="Test"),
            ],
        )
        warnings = validate_edit_list(edit_list, sample_transcript)
        assert any("inside cut region" in w for w in warnings)

    def test_interjection_after_episode(self, sample_transcript):
        edit_list = EditList(
            episode_guid="ep-001",
            segments=[
                SegmentAction(action="keep", start_ms=0, end_ms=70000),
            ],
            interjections=[
                Interjection(insert_after_ms=80000, text="Test"),
            ],
        )
        warnings = validate_edit_list(edit_list, sample_transcript)
        assert any("after episode end" in w for w in warnings)


class TestSemanticValidation:
    """Tests for the new semantic edit quality checks."""

    def test_first_kept_is_lex_warns(self, sample_transcript):
        edit_list = EditList(
            episode_guid="ep-001",
            segments=[
                SegmentAction(action="keep", start_ms=0, end_ms=6500, speaker="lex"),
                SegmentAction(action="keep", start_ms=6500, end_ms=9000, speaker="guest"),
            ],
            interjections=[],
        )
        warnings = validate_edit_list(edit_list, sample_transcript)
        assert any("First kept segment is from Lex" in w for w in warnings)

    def test_lex_segment_kept_warns(self, sample_transcript):
        edit_list = EditList(
            episode_guid="ep-001",
            segments=[
                SegmentAction(action="cut", start_ms=0, end_ms=6500, speaker="lex"),
                SegmentAction(action="keep", start_ms=6500, end_ms=9000, speaker="guest"),
                SegmentAction(action="keep", start_ms=9000, end_ms=12500, speaker="lex"),
                SegmentAction(action="keep", start_ms=12500, end_ms=70000, speaker="guest"),
            ],
            interjections=[],
        )
        warnings = validate_edit_list(edit_list, sample_transcript)
        assert any("Lex segment(s) marked as 'keep'" in w for w in warnings)

    def test_pre_guest_intro_kept_warns(self, sample_transcript):
        edit_list = EditList(
            episode_guid="ep-001",
            segments=[
                SegmentAction(action="keep", start_ms=0, end_ms=6000, speaker="lex"),
                SegmentAction(action="keep", start_ms=6500, end_ms=70000, speaker="guest"),
            ],
            interjections=[],
        )
        warnings = validate_edit_list(edit_list, sample_transcript)
        assert any("Pre-guest intro not fully cut" in w for w in warnings)

    def test_excessive_interjections_warns(self, sample_transcript):
        edit_list = EditList(
            episode_guid="ep-001",
            segments=[
                SegmentAction(action="keep", start_ms=6500, end_ms=9000, speaker="guest"),
                SegmentAction(action="keep", start_ms=12500, end_ms=25000, speaker="guest"),
            ],
            interjections=[
                Interjection(insert_after_ms=6500, text="Test 1"),
                Interjection(insert_after_ms=9000, text="Test 2"),
                Interjection(insert_after_ms=12500, text="Test 3"),
            ],
        )
        warnings = validate_edit_list(edit_list, sample_transcript)
        assert any("High interjection count" in w for w in warnings)

    def test_clean_edit_no_semantic_warnings(self, sample_transcript):
        """A properly edited list with guest-first, no kept Lex, low interjections."""
        edit_list = EditList(
            episode_guid="ep-001",
            segments=[
                SegmentAction(action="cut", start_ms=0, end_ms=6500, speaker="lex"),
                SegmentAction(action="keep", start_ms=6500, end_ms=9000, speaker="guest"),
                SegmentAction(action="cut", start_ms=9000, end_ms=12500, speaker="lex"),
                SegmentAction(action="keep", start_ms=12500, end_ms=25000, speaker="guest"),
                SegmentAction(action="cut", start_ms=25000, end_ms=28000, speaker="lex"),
                SegmentAction(action="keep", start_ms=28000, end_ms=48000, speaker="guest"),
                SegmentAction(action="cut", start_ms=48000, end_ms=53000, speaker="lex"),
                SegmentAction(action="keep", start_ms=53000, end_ms=70000, speaker="guest"),
            ],
            interjections=[],
        )
        warnings = validate_edit_list(edit_list, sample_transcript)
        assert warnings == []


class TestExtractOutline:
    def test_extracts_jensen_style_outline(self):
        desc = """<p><b>OUTLINE:</b><br/>
(00:00) &#8211; Introduction<br/>
(00:26) &#8211; Sponsors, Comments, and Reflections<br/>
(06:34) &#8211; Extreme co-design and rack-scale engineering<br/>
(09:20) &#8211; How Jensen runs NVIDIA</p>
<p><b>PODCAST LINKS:</b></p>"""
        outline = _extract_outline(desc)
        assert outline is not None
        assert "Introduction" in outline
        assert "Sponsors" in outline
        assert "(06:34)" in outline

    def test_returns_none_when_no_outline(self):
        assert _extract_outline("Just a plain description") is None


class TestBuildUserPrompt:
    def test_contains_transcript(self, sample_transcript):
        prompt = _build_user_prompt(sample_transcript)
        assert "Welcome." in prompt
        assert "[0-6500] lex:" in prompt

    def test_includes_outline_when_provided(self, sample_transcript):
        desc = """<p><b>OUTLINE:</b><br/>
(00:00) &#8211; Introduction<br/>
(06:34) &#8211; Topic</p>"""
        prompt = _build_user_prompt(sample_transcript, episode_description=desc)
        assert "EPISODE OUTLINE" in prompt
        assert "Introduction" in prompt

    def test_includes_published_transcript_ref(self, sample_transcript):
        published = Transcript(
            episode_guid="ep-001",
            segments=[
                TranscriptSegment(
                    speaker="lex", text="You've propelled NVIDIA...",
                    start_ms=33000, end_ms=42000,
                ),
            ],
        )
        prompt = _build_user_prompt(
            sample_transcript, published_transcript=published
        )
        assert "PUBLISHED TRANSCRIPT REFERENCE" in prompt
        assert "33000ms" in prompt

    def test_plain_prompt_without_context(self, sample_transcript):
        prompt = _build_user_prompt(sample_transcript)
        assert "AUDIO TRANSCRIPT TO EDIT:" in prompt
        assert "EPISODE OUTLINE" not in prompt
        assert "PUBLISHED TRANSCRIPT" not in prompt


class TestGenerateEditList:
    @respx.mock
    @pytest.mark.asyncio
    async def test_calls_anthropic_api(self, sample_transcript, sample_edit_json):
        respx.post(ANTHROPIC_API_URL).respond(
            200,
            json={
                "content": [{"type": "text", "text": sample_edit_json}],
                "role": "assistant",
            },
        )

        async with httpx.AsyncClient() as client:
            edit_list = await generate_edit_list(
                sample_transcript, "fake-key", client=client
            )

        assert len(edit_list.segments) == 8
        assert edit_list.episode_guid == "ep-001"

        # Verify API was called with correct headers
        request = respx.calls[0].request
        assert request.headers["x-api-key"] == "fake-key"
        assert request.headers["anthropic-version"] == "2023-06-01"

    @respx.mock
    @pytest.mark.asyncio
    async def test_raises_on_api_error(self, sample_transcript):
        respx.post(ANTHROPIC_API_URL).respond(500)

        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await generate_edit_list(sample_transcript, "fake-key", client=client)

    @respx.mock
    @pytest.mark.asyncio
    async def test_passes_episode_description(self, sample_transcript, sample_edit_json):
        respx.post(ANTHROPIC_API_URL).respond(
            200,
            json={
                "content": [{"type": "text", "text": sample_edit_json}],
                "role": "assistant",
            },
        )

        desc = """<p><b>OUTLINE:</b><br/>
(00:00) &#8211; Introduction<br/>
(06:34) &#8211; Topic</p>"""

        async with httpx.AsyncClient() as client:
            edit_list = await generate_edit_list(
                sample_transcript, "fake-key", client=client,
                episode_description=desc,
            )

        assert len(edit_list.segments) == 8
        # Verify the outline was included in the API request
        request = respx.calls[0].request
        body = json.loads(request.content)
        user_msg = body["messages"][0]["content"]
        assert "EPISODE OUTLINE" in user_msg
