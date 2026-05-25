"""Analyze materialized synthetic datasets against paper figure criteria.

Writes:
  data_cache/analysis/*_solution_hist.csv
  data_cache/analysis/distribution_report.json
  data_cache/analysis/distribution_report.txt
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch


def histogram(values):
    return dict(sorted(Counter(int(v) for v in values).items()))


def write_hist_csv(path: Path, hist: dict[int, int]):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_solutions", "count"])
        for k, v in hist.items():
            w.writerow([k, v])


def summarize_file(path: Path, out_dir: Path):
    payload = torch.load(path, map_location="cpu")
    task = payload["task"]
    n = payload["n"]
    if task == "nqueens":
        train_counts = [len(payload["completions"][tuple(x.tolist())]) for x in payload["train_inputs_unique"]]
        test_counts = [len(payload["completions"][tuple(x.tolist())]) for x in payload["test_inputs_unique"]]
        prefix = f"nqueens_n{n}"
        size = {
            "train_examples": int(payload["train_x"].shape[0]),
            "test_unique_inputs": int(payload["test_inputs_unique"].shape[0]),
            "input_seq_len": int(payload["train_x"].shape[1]),
            "target_seq_len": int(payload["train_y"].shape[1]),
        }
    else:
        train_counts = payload["train_solution_counts"].tolist()
        test_counts = payload["test_solution_counts"].tolist()
        prefix = f"graphcolor_n{n}"
        size = {
            "train_examples": int(payload["train_x"].shape[0]),
            "test_examples": int(payload["test_x"].shape[0]),
            "input_seq_len": int(payload["train_x"].shape[1]),
            "target_seq_len": int(payload["train_y_first_canonical"].shape[1]),
        }

    train_hist = histogram(train_counts)
    test_hist = histogram(test_counts)
    write_hist_csv(out_dir / f"{prefix}_train_solution_hist.csv", train_hist)
    write_hist_csv(out_dir / f"{prefix}_test_solution_hist.csv", test_hist)

    return {
        "file": str(path),
        "task": task,
        "n": n,
        **size,
        "train_solution_min": min(train_counts),
        "train_solution_max": max(train_counts),
        "test_solution_min": min(test_counts),
        "test_solution_max": max(test_counts),
        "train_solution_hist": train_hist,
        "test_solution_hist": test_hist,
        "metadata": payload.get("metadata", {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data_cache"))
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.data_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for path in sorted(args.data_dir.glob("*.pt")):
        reports.append(summarize_file(path, out_dir))

    json_path = out_dir / "distribution_report.json"
    json_path.write_text(json.dumps(reports, indent=2, sort_keys=True))

    lines = []
    for r in reports:
        prefix = "nqueens" if r["task"] == "nqueens" else "graphcolor"
        train_csv = out_dir / f"{prefix}_n{r['n']}_train_solution_hist.csv"
        test_csv = out_dir / f"{prefix}_n{r['n']}_test_solution_hist.csv"
        lines.append(f"{r['task']} n={r['n']}")
        lines.append(f"  file: {r['file']}")
        lines.append(f"  train/test solution range: {r['train_solution_min']}..{r['train_solution_max']} / {r['test_solution_min']}..{r['test_solution_max']}")
        lines.append(f"  train hist csv: {train_csv}")
        lines.append(f"  test hist csv : {test_csv}")
    text_path = out_dir / "distribution_report.txt"
    text_path.write_text("\n".join(lines) + "\n")
    print(text_path.read_text())
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
