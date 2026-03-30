"""LLM-based judge for evaluating podcast edit quality using Gemini."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx

from lex_without_lex.models import EditList, Transcript

logger = logging.getLogger(__name__)

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

JUDGE_SYSTEM_PROMPT = """\
You are an expert podcast editor evaluating the quality of an automated edit.
You will be given:
1. A podcast transcript with speaker labels and timestamps
2. An edit list specifying which segments to keep/cut and any interjections

The goal of the edit is to produce a guest-only version of the podcast:
maximum guest speech, zero host audio, minimal synthetic interjections.

Evaluate the edit on these criteria and score each 1-5:

1. HOST_REMOVAL: Are ALL host (Lex) segments cut? (5 = all cut, 1 = many kept)
2. INTRO_HANDLING: Is the host's solo intro, sponsors, and pre-guest content fully removed? Does the edit start with the guest's first utterance? (5 = fully removed, 1 = intro kept)
3. GUEST_PRESERVATION: Are ALL guest segments kept? (5 = all kept, 1 = many cut)
4. INTERJECTION_ECONOMY: Are interjections minimal — only used when the guest's response would be incomprehensible without context? (5 = zero unnecessary interjections, 1 = excessive interjections)
5. FLOW: Would the edited audio flow naturally for a listener? No jarring transitions? (5 = seamless, 1 = jarring)

Output ONLY valid JSON:
{
  "scores": {"HOST_REMOVAL": 5, "INTRO_HANDLING": 5, "GUEST_PRESERVATION": 5, "INTERJECTION_ECONOMY": 4, "FLOW": 4},
  "overall": 4.6,
  "issues": ["description of any issues found"],
  "pass": true
}

Set "pass" to true only if ALL individual scores are >= 4 and overall >= 4.0.
"""

CRITERIA = [
    "HOST_REMOVAL",
    "INTRO_HANDLING",
    "GUEST_PRESERVATION",
    "INTERJECTION_ECONOMY",
    "FLOW",
]


@dataclass
class JudgeResult:
    scores: dict[str, int]
    overall: float
    issues: list[str]
    passed: bool
    raw_response: str = ""


def _build_judge_prompt(transcript: Transcript, edit_list: EditList) -> str:
    """Build the evaluation prompt with transcript and edit list."""
    lines = ["TRANSCRIPT:\n"]
    for seg in transcript.segments:
        lines.append(f"[{seg.start_ms}-{seg.end_ms}] {seg.speaker}: {seg.text}")

    lines.append("\n\nEDIT LIST:\n")
    lines.append("Segments:")
    for seg in edit_list.segments:
        lines.append(
            f"  {seg.action} [{seg.start_ms}-{seg.end_ms}] "
            f"speaker={seg.speaker} reason={seg.reason}"
        )

    if edit_list.interjections:
        lines.append("\nInterjections:")
        for inj in edit_list.interjections:
            lines.append(
                f"  at {inj.insert_after_ms}ms: \"{inj.text}\" "
                f"(context: {inj.context})"
            )
    else:
        lines.append("\nInterjections: none")

    lines.append(f"\nSummary: {edit_list.summary}")
    return "\n".join(lines)


def _parse_judge_response(text: str) -> JudgeResult:
    """Parse Gemini's judge response into a JudgeResult."""
    # Strip markdown fences if present
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean = "\n".join(lines)

    data = json.loads(clean)
    scores = data.get("scores", {})
    return JudgeResult(
        scores=scores,
        overall=float(data.get("overall", 0)),
        issues=data.get("issues", []),
        passed=bool(data.get("pass", False)),
        raw_response=text,
    )


async def judge_edit_quality(
    transcript: Transcript,
    edit_list: EditList,
    gemini_api_key: str,
    client: httpx.AsyncClient | None = None,
) -> JudgeResult:
    """Use Gemini Flash to evaluate an edit list against a transcript.

    Returns a JudgeResult with scores, overall rating, issues, and pass/fail.
    """
    user_content = _build_judge_prompt(transcript, edit_list)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": JUDGE_SYSTEM_PROMPT + "\n\n" + user_content}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
        },
    }

    async def _call(c: httpx.AsyncClient) -> JudgeResult:
        resp = await c.post(
            GEMINI_API_URL,
            params={"key": gemini_api_key},
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        response_data = resp.json()
        text = (
            response_data["candidates"][0]["content"]["parts"][0]["text"]
        )
        return _parse_judge_response(text)

    if client is None:
        async with httpx.AsyncClient() as c:
            return await _call(c)
    return await _call(client)
