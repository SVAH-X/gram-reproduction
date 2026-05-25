"""Materialize the four paper-style synthetic datasets for inspection.

Outputs are written under data_cache/ by default:
  * nqueens_n8_seed0.pt
  * nqueens_n10_seed0.pt
  * graphcolor_n8_p0.4_seed0.pt
  * graphcolor_n10_p0.4_seed0.pt
  * metadata.json
  * summary.txt

The .pt files contain plain tensors/lists/dicts so they can be inspected with
torch.load without importing training code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from data_nqueens import (
    EMPTY,
    PAD as NQ_PAD,
    QUEEN,
    VOCAB_SIZE as NQ_VOCAB_SIZE,
    NQueensTrainDataset,
    build_nqueens,
    steps_per_epoch,
)
from data_graph_coloring import (
    COLOR_BASE,
    EDGE,
    NO_EDGE,
    NUM_COLORS,
    PAD as GC_PAD,
    VOCAB_SIZE as GC_VOCAB_SIZE,
    build_graph_coloring,
    default_split_sizes,
    graph_to_tensors,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_with_hash(payload: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return sha256_file(path)


def tensorize_boards(items):
    return torch.tensor(items, dtype=torch.long)


def materialize_nqueens(n: int, seed: int, global_batch: int, out_dir: Path) -> dict[str, Any]:
    build = build_nqueens(n, seed=seed)
    train_ds = NQueensTrainDataset(build)
    train_x = tensorize_boards([x for x, _ in train_ds.pairs])
    train_y = tensorize_boards([y for _, y in train_ds.pairs])
    test_inputs = tensorize_boards(build.test_inputs)
    train_inputs = tensorize_boards(build.train_inputs)

    payload = {
        "task": "nqueens",
        "n": n,
        "train_x": train_x,
        "train_y": train_y,
        "train_inputs_unique": train_inputs,
        "test_inputs_unique": test_inputs,
        "completions": build.completions,
        "token_map": {"pad": NQ_PAD, "empty": EMPTY, "queen": QUEEN},
        "metadata": {
            "seed": seed,
            "train_fraction": 0.85,
            "removed_counts": (5, 6, 7) if n == 8 else (7, 8, 9),
            "raw_generated_pairs": build.raw_pairs,
            "unique_train_inputs": len(build.train_inputs),
            "unique_test_inputs": len(build.test_inputs),
            "train_target_pairs": build.train_pairs,
            "test_target_pairs": build.test_pairs,
            "vocab_size": NQ_VOCAB_SIZE,
            "seq_len": n * n,
            "global_batch": global_batch,
            "dataset_batches_per_pass": steps_per_epoch(len(train_ds), global_batch),
            "paper_trajectory_batches": 3000 if n == 8 else 1000,
            "segment_updates_per_trajectory": 16,
            "paper_training_steps": (3000 if n == 8 else 1000) * 16,
        },
    }
    path = out_dir / f"nqueens_n{n}_seed{seed}.pt"
    digest = save_with_hash(payload, path)
    payload["metadata"]["path"] = str(path)
    payload["metadata"]["sha256"] = digest
    return payload["metadata"]


def materialize_graphcolor(n: int, p_edge: float, seed: int, global_batch: int, out_dir: Path) -> dict[str, Any]:
    train_size, test_size = default_split_sizes(n)
    # Disable the internal cache here; this script is the explicit materializer.
    build = build_graph_coloring(
        n,
        p_edge=p_edge,
        train_size=train_size,
        test_size=test_size,
        seed=seed,
        cache_dir=None,
    )

    train_x = []
    train_y = []
    train_solution_counts = []
    for adj_flat in build.train_graphs:
        sols = build.colorings[adj_flat]
        x, y = graph_to_tensors(adj_flat, sols[0], n)
        train_x.append(x)
        train_y.append(y)
        train_solution_counts.append(len(sols))

    test_x = []
    test_solution_counts = []
    for adj_flat in build.test_graphs:
        sols = build.colorings[adj_flat]
        test_x.append(torch.tensor([EDGE if e else NO_EDGE for e in adj_flat], dtype=torch.long))
        test_solution_counts.append(len(sols))

    payload = {
        "task": "graph_coloring",
        "n": n,
        "train_x": torch.stack(train_x, dim=0),
        "train_y_first_canonical": torch.stack(train_y, dim=0),
        "test_x": torch.stack(test_x, dim=0),
        "train_graphs": build.train_graphs,
        "test_graphs": build.test_graphs,
        "colorings": build.colorings,
        "train_solution_counts": torch.tensor(train_solution_counts, dtype=torch.long),
        "test_solution_counts": torch.tensor(test_solution_counts, dtype=torch.long),
        "token_map": {
            "pad": GC_PAD,
            "no_edge": NO_EDGE,
            "edge": EDGE,
            "color_base": COLOR_BASE,
            "colors": [COLOR_BASE + i for i in range(NUM_COLORS)],
        },
        "metadata": {
            "seed": seed,
            "p_edge": p_edge,
            "train_graphs": len(build.train_graphs),
            "test_graphs": len(build.test_graphs),
            "vocab_size": GC_VOCAB_SIZE,
            "input_seq_len": n * (n - 1) // 2,
            "target_seq_len": n,
            "global_batch": global_batch,
            "dataset_batches_per_pass": steps_per_epoch(len(build.train_graphs), global_batch),
            "paper_trajectory_batches": 5000,
            "segment_updates_per_trajectory": 16,
            "paper_training_steps": 5000 * 16,
            "train_solution_count_min": int(min(train_solution_counts)),
            "train_solution_count_max": int(max(train_solution_counts)),
            "test_solution_count_min": int(min(test_solution_counts)),
            "test_solution_count_max": int(max(test_solution_counts)),
        },
    }
    path = out_dir / f"graphcolor_n{n}_p{p_edge:g}_seed{seed}.pt"
    digest = save_with_hash(payload, path)
    payload["metadata"]["path"] = str(path)
    payload["metadata"]["sha256"] = digest
    return payload["metadata"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data_cache"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-edge", type=float, default=0.4)
    ap.add_argument("--global-batch", type=int, default=768)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    summaries.append(materialize_nqueens(8, args.seed, args.global_batch, args.out_dir))
    summaries.append(materialize_nqueens(10, args.seed, args.global_batch, args.out_dir))
    summaries.append(materialize_graphcolor(8, args.p_edge, args.seed, args.global_batch, args.out_dir))
    summaries.append(materialize_graphcolor(10, args.p_edge, args.seed, args.global_batch, args.out_dir))

    metadata_path = args.out_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)

    lines = []
    for meta in summaries:
        task = meta.get("path", "")
        lines.append(task)
        for key in (
            "seq_len",
            "input_seq_len",
            "target_seq_len",
            "unique_train_inputs",
            "unique_test_inputs",
            "train_target_pairs",
            "test_target_pairs",
            "train_graphs",
            "test_graphs",
            "steps_per_epoch",
            "dataset_batches_per_pass",
            "paper_trajectory_batches",
            "segment_updates_per_trajectory",
            "paper_training_steps",
            "sha256",
        ):
            if key in meta:
                lines.append(f"  {key}: {meta[key]}")
    summary_path = args.out_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")

    print(summary_path.read_text())
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
