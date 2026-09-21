"""Measure calibration: run the model over the held-out eval pool, parse
(answer, confidence, correctness), compute ECE + Brier score, and plot a
reliability diagram. Run this once before training and once after (against
the LoRA adapter) to compare.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rlcd.common import NEUTRAL_SYSTEM_PROMPT, build_question_block, parse_response
from rlcd.data import get_eval_pool
from rlcd.model_utils import DEFAULT_MODEL, generate_batch, load_model_and_tokenizer

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


def expected_calibration_error(confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = corrects[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def reliability_diagram(confidences: np.ndarray, corrects: np.ndarray, title: str, out_path: Path, n_bins: int = 10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_accs, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_accs.append(corrects[mask].mean())
        bin_counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.bar(bin_centers, bin_accs, width=1 / n_bins * 0.9, alpha=0.7, edgecolor="black", label="observed accuracy")
    for c, a, n in zip(bin_centers, bin_accs, bin_counts):
        ax.annotate(str(n), (c, a), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_eval(model_name: str, adapter_path: str | None, n: int, out_prefix: str, temperature: float, do_sample: bool):
    pool = get_eval_pool()[:n]
    model, tokenizer = load_model_and_tokenizer(model_name, adapter_path)

    conversations = [
        [
            {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
            {"role": "user", "content": build_question_block(q.question, q.choices)},
        ]
        for q in pool
    ]

    completions = generate_batch(
        model, tokenizer, conversations, max_new_tokens=30, do_sample=do_sample, temperature=temperature
    )

    records = []
    for q, completion in zip(pool, completions):
        parsed = parse_response(completion)
        correct = parsed.letter == q.answer_letter if parsed.ok else False
        records.append(
            {
                "subject": q.subject,
                "answer_letter": q.answer_letter,
                "raw_completion": completion,
                "parsed_letter": parsed.letter,
                "parsed_confidence": parsed.confidence,
                "parsed_ok": parsed.ok,
                "correct": correct,
            }
        )

    OUTPUTS_DIR.mkdir(exist_ok=True)
    records_path = OUTPUTS_DIR / f"{out_prefix}_records.json"
    records_path.write_text(json.dumps(records, indent=2))

    parseable = [r for r in records if r["parsed_ok"]]
    parse_failure_rate = 1 - len(parseable) / len(records)
    confidences = np.array([r["parsed_confidence"] for r in parseable])
    corrects = np.array([1.0 if r["correct"] else 0.0 for r in parseable])

    accuracy = float(corrects.mean()) if len(corrects) else float("nan")
    brier = float(np.mean((confidences - corrects) ** 2)) if len(corrects) else float("nan")
    ece = expected_calibration_error(confidences, corrects) if len(corrects) else float("nan")

    summary = {
        "model": model_name,
        "adapter": adapter_path,
        "n": len(records),
        "parse_failure_rate": parse_failure_rate,
        "accuracy": accuracy,
        "brier_score": brier,
        "ece": ece,
    }
    summary_path = OUTPUTS_DIR / f"{out_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    if len(corrects):
        reliability_diagram(
            confidences,
            corrects,
            title=f"{out_prefix}: ECE={ece:.3f}, Brier={brier:.3f}, acc={accuracy:.2f}",
            out_path=OUTPUTS_DIR / f"{out_prefix}_reliability.png",
        )

    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None, help="path to a trained LoRA adapter")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--out-prefix", default="baseline")
    parser.add_argument("--sample", action="store_true", help="sample instead of greedy decode")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    run_eval(args.model, args.adapter, args.n, args.out_prefix, args.temperature, args.sample)


if __name__ == "__main__":
    main()
