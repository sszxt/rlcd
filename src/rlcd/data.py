"""MMLU loading: a disjoint train pool (for preference generation) and eval
pool (for calibration measurement), sampled with a fixed seed so re-running
any script draws the same split.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from datasets import load_dataset

from rlcd.common import LETTERS

SEED = 0
TRAIN_POOL_SIZE = 3000
EVAL_POOL_SIZE = 500


@dataclass
class Question:
    question: str
    choices: list[str]
    answer_letter: str
    subject: str


def _to_question(row: dict) -> Question:
    return Question(
        question=row["question"],
        choices=list(row["choices"]),
        answer_letter=LETTERS[row["answer"]],
        subject=row.get("subject", ""),
    )


def _load_pools() -> tuple[list[Question], list[Question]]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    indices = list(range(len(ds)))
    random.Random(SEED).shuffle(indices)

    eval_idx = indices[:EVAL_POOL_SIZE]
    train_idx = indices[EVAL_POOL_SIZE : EVAL_POOL_SIZE + TRAIN_POOL_SIZE]

    eval_pool = [_to_question(ds[i]) for i in eval_idx]
    train_pool = [_to_question(ds[i]) for i in train_idx]
    return train_pool, eval_pool


def get_train_pool() -> list[Question]:
    train_pool, _ = _load_pools()
    return train_pool


def get_eval_pool() -> list[Question]:
    _, eval_pool = _load_pools()
    return eval_pool
