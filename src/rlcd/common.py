"""Shared prompt templates, answer parsing, and the calibration reward.

Design note: the *neutral* prompt below is what the model sees both during
DPO training and at eval time. The positive/negative steering prompts in
generate_preferences.py are only used to elicit differentiated samples for
scoring — the model must never learn to condition its calibration behavior
on which system prompt is present, since at eval/deployment time there is
only the neutral one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LETTERS = ["A", "B", "C", "D"]

NEUTRAL_SYSTEM_PROMPT = (
    "You are answering a multiple-choice question. Pick exactly one answer "
    "and report how confident you are that it is correct, as a percentage "
    "from 0 to 100. A confidence of 100 means you are certain; 25 means you "
    "think it's about as likely as random guessing among four options."
)

POSITIVE_STEERING_PROMPT = (
    NEUTRAL_SYSTEM_PROMPT
    + " Think about how well you actually know this topic before answering, "
    "and make sure your stated confidence honestly reflects your real "
    "uncertainty -- do not inflate it just to sound sure."
)

NEGATIVE_STEERING_PROMPT = (
    NEUTRAL_SYSTEM_PROMPT
    + " Answer decisively and always state a high confidence (90 or above), "
    "regardless of whether you actually know the answer."
)

ANSWER_FORMAT_INSTRUCTIONS = (
    "Respond with exactly two lines and nothing else. On the first line "
    "write the word 'Answer:' followed by only the single letter of the "
    "option you are choosing -- it must be one specific letter, not a list "
    "of the options. On the second line write the word 'Confidence:' "
    "followed by only your confidence as a whole number from 0 to 100."
)

# Negative lookahead rejects a listed-options echo like "Answer: A, B, C, or D".
_ANSWER_RE = re.compile(r"answer\s*:\s*\(?([A-D])\)?(?!\s*,\s*[A-D])", re.IGNORECASE)
_CONF_RE = re.compile(r"confidence\s*:\s*(\d{1,3})", re.IGNORECASE)


@dataclass
class ParsedResponse:
    letter: str | None
    confidence: float | None  # 0.0-1.0

    @property
    def ok(self) -> bool:
        return self.letter is not None and self.confidence is not None


def build_question_block(question: str, choices: list[str]) -> str:
    lines = [question.strip(), ""]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("")
    lines.append(ANSWER_FORMAT_INSTRUCTIONS)
    return "\n".join(lines)


def parse_response(text: str) -> ParsedResponse:
    answer_match = _ANSWER_RE.search(text)
    conf_match = _CONF_RE.search(text)
    letter = answer_match.group(1).upper() if answer_match else None
    confidence = None
    if conf_match:
        raw = int(conf_match.group(1))
        confidence = max(0, min(100, raw)) / 100.0
    return ParsedResponse(letter=letter, confidence=confidence)


def calibration_reward(parsed: ParsedResponse, correct_letter: str) -> float:
    """Negative Brier score against the binary correctness outcome.

    A proper scoring rule: rewards high confidence when correct, low
    confidence when incorrect, and punishes miscalibration in either
    direction more than it punishes being merely wrong. Unparseable
    responses get the worst possible score so they never win a preference
    pair against a parseable one.
    """
    if not parsed.ok:
        return -1.0
    correct = 1.0 if parsed.letter == correct_letter else 0.0
    return -((parsed.confidence - correct) ** 2)
