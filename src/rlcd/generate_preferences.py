"""RLCD-style label-free preference generation.

For each training question, sample one completion under a positive
("be honestly calibrated") steering prompt and one under a negative
("be overconfident") steering prompt. Score both with the calibration
reward (rlcd.common.calibration_reward) and keep whichever scored higher as
`chosen`, the other as `rejected`. The stored `prompt` uses the neutral
system prompt -- the steering prompts only shape which sample gets
generated, they are never seen by the trained model, so it can't just learn
to key off "am I being told to be honest or overconfident."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlcd.common import (
    NEGATIVE_STEERING_PROMPT,
    NEUTRAL_SYSTEM_PROMPT,
    POSITIVE_STEERING_PROMPT,
    build_question_block,
    calibration_reward,
    parse_response,
)
from rlcd.data import get_train_pool
from rlcd.model_utils import DEFAULT_MODEL, generate_batch, load_model_and_tokenizer

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--out", default="preferences.jsonl")
    args = parser.parse_args()

    pool = get_train_pool()[: args.n]
    model, tokenizer = load_model_and_tokenizer(args.model)

    user_blocks = [build_question_block(q.question, q.choices) for q in pool]

    pos_conversations = [
        [{"role": "system", "content": POSITIVE_STEERING_PROMPT}, {"role": "user", "content": ub}]
        for ub in user_blocks
    ]
    neg_conversations = [
        [{"role": "system", "content": NEGATIVE_STEERING_PROMPT}, {"role": "user", "content": ub}]
        for ub in user_blocks
    ]

    pos_completions = generate_batch(
        model, tokenizer, pos_conversations, max_new_tokens=30, do_sample=True, temperature=args.temperature
    )
    neg_completions = generate_batch(
        model, tokenizer, neg_conversations, max_new_tokens=30, do_sample=True, temperature=args.temperature
    )

    OUTPUTS_DIR.mkdir(exist_ok=True)
    out_path = OUTPUTS_DIR / args.out

    kept, skipped_ties, skipped_both_unparseable = 0, 0, 0
    with out_path.open("w") as f:
        for q, ub, pos_text, neg_text in zip(pool, user_blocks, pos_completions, neg_completions):
            pos_parsed = parse_response(pos_text)
            neg_parsed = parse_response(neg_text)

            if not pos_parsed.ok and not neg_parsed.ok:
                skipped_both_unparseable += 1
                continue

            pos_reward = calibration_reward(pos_parsed, q.answer_letter)
            neg_reward = calibration_reward(neg_parsed, q.answer_letter)

            if pos_reward == neg_reward:
                skipped_ties += 1
                continue

            chosen, rejected = (pos_text, neg_text) if pos_reward > neg_reward else (neg_text, pos_text)

            record = {
                "prompt": [
                    {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
                    {"role": "user", "content": ub},
                ],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "subject": q.subject,
                "pos_reward": pos_reward,
                "neg_reward": neg_reward,
            }
            f.write(json.dumps(record) + "\n")
            kept += 1

    print(
        f"wrote {kept} preference pairs to {out_path} "
        f"(skipped {skipped_ties} ties, {skipped_both_unparseable} both-unparseable)"
    )


if __name__ == "__main__":
    main()
