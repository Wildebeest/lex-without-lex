"""LLM-judge tests for edit quality evaluation.

These tests use Gemini Flash (via OpenRouter or direct API) to evaluate
whether edit lists achieve the editing objective: maximum guest speech,
minimum interjections, zero host audio.

Run with: pytest tests/test_edit_quality.py -m llm_judge
Requires: LWL_OPENROUTER_API_KEY or LWL_GEMINI_API_KEY environment variable
"""

from __future__ import annotations

import json

import pytest

from lex_without_lex.editor import parse_opus_response, validate_edit_list
from lex_without_lex.models import (
    EditList,
    Interjection,
    SegmentAction,
    Transcript,
    TranscriptSegment,
)

from .llm_judge import judge_edit_quality


def _load_scenario(fixtures_dir, name: str) -> Transcript:
    data = json.loads((fixtures_dir / f"scenario_{name}.json").read_text())
    return Transcript(
        episode_guid=data.get("episode_guid", f"test-{name}"),
        segments=[TranscriptSegment(**s) for s in data["segments"]],
    )


def _load_golden_edit(fixtures_dir, name: str) -> EditList:
    data = json.loads((fixtures_dir / f"golden_edit_{name}.json").read_text())
    return EditList(
        episode_guid=f"test-{name}",
        segments=[SegmentAction(**s) for s in data["segments"]],
        interjections=[Interjection(**i) for i in data.get("interjections", [])],
        summary=data.get("summary", ""),
    )


@pytest.mark.llm_judge
class TestIntroAndSponsorsCut:
    """Verify a Jensen-like episode with 6-min intro/sponsors is handled correctly."""

    def test_structural_checks(self, fixtures_dir):
        transcript = _load_scenario(fixtures_dir, "jensen_intro")
        edit_list = _load_golden_edit(fixtures_dir, "jensen_intro")

        # First kept segment should be guest
        kept = sorted(
            [s for s in edit_list.segments if s.action == "keep"],
            key=lambda s: s.start_ms,
        )
        assert kept[0].speaker == "guest"
        assert kept[0].start_ms == 394000, "First guest speech should start after sponsors"

        # All intro/sponsor segments (0-394000ms) should be cut
        intro_kept = [
            s for s in edit_list.segments
            if s.action == "keep" and s.start_ms < 394000
        ]
        assert intro_kept == [], "All pre-guest segments must be cut"

        # No interjections
        assert len(edit_list.interjections) == 0

        # Validation should pass cleanly
        warnings = validate_edit_list(edit_list, transcript)
        assert warnings == []

    @pytest.mark.asyncio
    async def test_gemini_judge_approves(self, fixtures_dir, gemini_api_key, judge_backend):
        transcript = _load_scenario(fixtures_dir, "jensen_intro")
        edit_list = _load_golden_edit(fixtures_dir, "jensen_intro")

        result = await judge_edit_quality(
            transcript, edit_list, gemini_api_key, backend=judge_backend
        )
        assert result.scores.get("INTRO_HANDLING", 0) >= 4, (
            f"INTRO_HANDLING score too low: {result.scores}. Issues: {result.issues}"
        )
        assert result.scores.get("HOST_REMOVAL", 0) >= 4
        assert result.passed, f"Judge did not pass. Issues: {result.issues}"


@pytest.mark.llm_judge
class TestHostFullyRemoved:
    """Standard alternating conversation — all host segments should be cut."""

    @pytest.fixture
    def scenario(self):
        return Transcript(
            episode_guid="test-host-removal",
            segments=[
                TranscriptSegment(speaker="lex", text="Welcome to the show.", start_ms=0, end_ms=5000),
                TranscriptSegment(speaker="guest", text="Great to be here. I've been working on quantum computing for twenty years now.", start_ms=5000, end_ms=30000),
                TranscriptSegment(speaker="lex", text="Tell me about quantum error correction.", start_ms=30000, end_ms=35000),
                TranscriptSegment(speaker="guest", text="Error correction is the key challenge. We need logical qubits that are fault-tolerant. The surface code approach shows promise but requires thousands of physical qubits per logical qubit.", start_ms=35000, end_ms=90000),
                TranscriptSegment(speaker="lex", text="When will we have useful quantum computers?", start_ms=90000, end_ms=95000),
                TranscriptSegment(speaker="guest", text="I think within ten years we'll see quantum advantage for specific problems like drug discovery and materials science. But general-purpose quantum computing is further out.", start_ms=95000, end_ms=150000),
            ],
        )

    @pytest.fixture
    def golden_edit(self):
        return EditList(
            episode_guid="test-host-removal",
            segments=[
                SegmentAction(action="cut", start_ms=0, end_ms=5000, speaker="lex", reason="Host intro"),
                SegmentAction(action="keep", start_ms=5000, end_ms=30000, speaker="guest", reason="Guest on quantum computing career"),
                SegmentAction(action="cut", start_ms=30000, end_ms=35000, speaker="lex", reason="Host question"),
                SegmentAction(action="keep", start_ms=35000, end_ms=90000, speaker="guest", reason="Guest on error correction"),
                SegmentAction(action="cut", start_ms=90000, end_ms=95000, speaker="lex", reason="Host question"),
                SegmentAction(action="keep", start_ms=95000, end_ms=150000, speaker="guest", reason="Guest on timeline"),
            ],
            interjections=[],
            summary="Removed 3 Lex segments, kept 3 guest segments, zero interjections.",
        )

    def test_structural(self, scenario, golden_edit):
        kept = [s for s in golden_edit.segments if s.action == "keep"]
        assert all(s.speaker == "guest" for s in kept)
        assert len(golden_edit.interjections) == 0
        warnings = validate_edit_list(golden_edit, scenario)
        assert warnings == []

    @pytest.mark.asyncio
    async def test_gemini_judge(self, scenario, golden_edit, gemini_api_key, judge_backend):
        result = await judge_edit_quality(
            scenario, golden_edit, gemini_api_key, backend=judge_backend
        )
        assert result.scores.get("HOST_REMOVAL", 0) >= 5
        assert result.scores.get("GUEST_PRESERVATION", 0) >= 5
        assert result.passed, f"Issues: {result.issues}"


@pytest.mark.llm_judge
class TestGuestFullyPreserved:
    """Verify no guest speech is accidentally cut."""

    @pytest.fixture
    def scenario(self):
        return Transcript(
            episode_guid="test-guest-preserved",
            segments=[
                TranscriptSegment(speaker="lex", text="Let's dive in.", start_ms=0, end_ms=3000),
                TranscriptSegment(speaker="guest", text="So the key insight of our paper is that attention mechanisms can be viewed as learned information routing.", start_ms=3000, end_ms=40000),
                TranscriptSegment(speaker="guest", text="Building on that, we showed that multi-head attention creates parallel routing channels, each specializing in different types of information flow.", start_ms=40000, end_ms=80000),
                TranscriptSegment(speaker="lex", text="Fascinating.", start_ms=80000, end_ms=82000),
                TranscriptSegment(speaker="guest", text="And the implications for scaling are significant. Larger models don't just memorize more — they develop more sophisticated routing strategies.", start_ms=82000, end_ms=120000),
            ],
        )

    @pytest.fixture
    def golden_edit(self):
        return EditList(
            episode_guid="test-guest-preserved",
            segments=[
                SegmentAction(action="cut", start_ms=0, end_ms=3000, speaker="lex", reason="Host intro"),
                SegmentAction(action="keep", start_ms=3000, end_ms=40000, speaker="guest", reason="Guest on attention routing"),
                SegmentAction(action="keep", start_ms=40000, end_ms=80000, speaker="guest", reason="Guest on multi-head attention"),
                SegmentAction(action="cut", start_ms=80000, end_ms=82000, speaker="lex", reason="Host reaction"),
                SegmentAction(action="keep", start_ms=82000, end_ms=120000, speaker="guest", reason="Guest on scaling implications"),
            ],
            interjections=[],
            summary="Removed 2 Lex segments, kept 3 guest segments.",
        )

    @pytest.mark.asyncio
    async def test_gemini_judge(self, scenario, golden_edit, gemini_api_key, judge_backend):
        result = await judge_edit_quality(
            scenario, golden_edit, gemini_api_key, backend=judge_backend
        )
        assert result.scores.get("GUEST_PRESERVATION", 0) >= 5
        assert result.passed, f"Issues: {result.issues}"


@pytest.mark.llm_judge
class TestInterjectionsMinimal:
    """Conversation where the guest restates topics — should need few/no interjections."""

    @pytest.fixture
    def scenario(self):
        return Transcript(
            episode_guid="test-interjections",
            segments=[
                TranscriptSegment(speaker="lex", text="Hello.", start_ms=0, end_ms=2000),
                TranscriptSegment(speaker="guest", text="Thanks for having me.", start_ms=2000, end_ms=5000),
                TranscriptSegment(speaker="lex", text="Tell me about your childhood.", start_ms=5000, end_ms=8000),
                TranscriptSegment(speaker="guest", text="I grew up in Taiwan and moved to the US when I was nine. My parents wanted me to have better educational opportunities.", start_ms=8000, end_ms=40000),
                TranscriptSegment(speaker="lex", text="What was that transition like?", start_ms=40000, end_ms=43000),
                TranscriptSegment(speaker="guest", text="The transition from Taiwan to the US was incredibly difficult. I didn't speak English. I was bullied. But it taught me resilience.", start_ms=43000, end_ms=90000),
                TranscriptSegment(speaker="lex", text="How did you start NVIDIA?", start_ms=90000, end_ms=94000),
                TranscriptSegment(speaker="guest", text="Starting NVIDIA was about seeing an opportunity in 3D graphics. Chris Malachowsky, Curtis Priem, and I founded it in 1993 at a Denny's restaurant.", start_ms=94000, end_ms=150000),
            ],
        )

    @pytest.fixture
    def golden_edit(self):
        return EditList(
            episode_guid="test-interjections",
            segments=[
                SegmentAction(action="cut", start_ms=0, end_ms=2000, speaker="lex", reason="Greeting"),
                SegmentAction(action="keep", start_ms=2000, end_ms=5000, speaker="guest", reason="Guest greeting"),
                SegmentAction(action="cut", start_ms=5000, end_ms=8000, speaker="lex", reason="Host question"),
                SegmentAction(action="keep", start_ms=8000, end_ms=40000, speaker="guest", reason="Guest on childhood"),
                SegmentAction(action="cut", start_ms=40000, end_ms=43000, speaker="lex", reason="Host follow-up"),
                SegmentAction(action="keep", start_ms=43000, end_ms=90000, speaker="guest", reason="Guest on transition"),
                SegmentAction(action="cut", start_ms=90000, end_ms=94000, speaker="lex", reason="Host question"),
                SegmentAction(action="keep", start_ms=94000, end_ms=150000, speaker="guest", reason="Guest on founding NVIDIA"),
            ],
            interjections=[],
            summary="Removed 4 Lex segments, kept 4 guest segments, zero interjections. Guest restates topics naturally.",
        )

    def test_structural(self, scenario, golden_edit):
        assert len(golden_edit.interjections) == 0
        warnings = validate_edit_list(golden_edit, scenario)
        assert warnings == []

    @pytest.mark.asyncio
    async def test_gemini_judge(self, scenario, golden_edit, gemini_api_key, judge_backend):
        result = await judge_edit_quality(
            scenario, golden_edit, gemini_api_key, backend=judge_backend
        )
        assert result.scores.get("INTERJECTION_ECONOMY", 0) >= 4
        assert result.passed, f"Issues: {result.issues}"


@pytest.mark.llm_judge
class TestSponsorReadMidEpisode:
    """Mid-episode sponsor read by Lex should be cut."""

    @pytest.fixture
    def scenario(self):
        return Transcript(
            episode_guid="test-mid-sponsor",
            segments=[
                TranscriptSegment(speaker="guest", text="And that's how we solved the memory bandwidth problem.", start_ms=0, end_ms=30000),
                TranscriptSegment(speaker="lex", text="Let me take a quick break. This episode is brought to you by ExpressVPN. Go to expressvpn.com/lex.", start_ms=30000, end_ms=60000),
                TranscriptSegment(speaker="lex", text="Also brought to you by Athletic Greens. I drink it every morning.", start_ms=60000, end_ms=85000),
                TranscriptSegment(speaker="lex", text="OK, back to our conversation. What happened next with the architecture?", start_ms=85000, end_ms=92000),
                TranscriptSegment(speaker="guest", text="So the next step was to integrate the networking stack directly into the silicon. This was radical at the time.", start_ms=92000, end_ms=140000),
            ],
        )

    @pytest.fixture
    def golden_edit(self):
        return EditList(
            episode_guid="test-mid-sponsor",
            segments=[
                SegmentAction(action="keep", start_ms=0, end_ms=30000, speaker="guest", reason="Guest on memory bandwidth"),
                SegmentAction(action="cut", start_ms=30000, end_ms=60000, speaker="lex", reason="Mid-episode sponsor (ExpressVPN)"),
                SegmentAction(action="cut", start_ms=60000, end_ms=85000, speaker="lex", reason="Mid-episode sponsor (Athletic Greens)"),
                SegmentAction(action="cut", start_ms=85000, end_ms=92000, speaker="lex", reason="Host transition back from sponsors"),
                SegmentAction(action="keep", start_ms=92000, end_ms=140000, speaker="guest", reason="Guest on networking integration"),
            ],
            interjections=[],
            summary="Cut 3 Lex segments including mid-episode sponsors, kept 2 guest segments.",
        )

    def test_structural(self, scenario, golden_edit):
        warnings = validate_edit_list(golden_edit, scenario)
        assert warnings == []
        cut_segments = [s for s in golden_edit.segments if s.action == "cut"]
        assert any("sponsor" in s.reason.lower() for s in cut_segments)

    @pytest.mark.asyncio
    async def test_gemini_judge(self, scenario, golden_edit, gemini_api_key, judge_backend):
        result = await judge_edit_quality(
            scenario, golden_edit, gemini_api_key, backend=judge_backend
        )
        assert result.scores.get("HOST_REMOVAL", 0) >= 4
        assert result.scores.get("FLOW", 0) >= 4
        assert result.passed, f"Issues: {result.issues}"
