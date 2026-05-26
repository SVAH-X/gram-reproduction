"""Paper-style evaluation for N-Queens and Graph Coloring.

Reports N=1/5/10/20 sampling coverage and buckets metrics by the number of
valid solutions, matching the analysis behind Figures 4 and 15.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from gram_model import GRAM, GRAMConfig
from data_nqueens import QUEEN, nqueens_accuracy, nqueens_coverage, nqueens_diagnostics
from data_graph_coloring import (
    COLOR_BASE,
    NUM_COLORS,
    VOCAB_SIZE as GC_VOCAB_SIZE,
    color_logits_to_tokens,
    conflict_edges,
    graph_coverage,
    upper_to_adj,
)


SAMPLE_COUNTS = (1, 5, 10, 20)


def _fallback_config(args):
    if args.task == "nqueens":
        cfg = GRAMConfig(
            vocab_size=3,
            seq_len=args.n * args.n,
            d_model=512,
            n_heads=8,
            ffn_hidden=512,
            n_layers=2,
            K=4,
            T=3,
            N_sup=16,
            use_attn=True,
            use_rope=True,
            use_halt=True,
        )
    else:
        cfg = GRAMConfig(
            vocab_size=GC_VOCAB_SIZE,
            seq_len=args.n * (args.n - 1) // 2,
            target_seq_len=args.n,
            d_model=512,
            n_heads=8,
            ffn_hidden=512,
            n_layers=2,
            K=4,
            T=3,
            N_sup=16,
            use_attn=True,
            use_rope=True,
            use_halt=True,
        )
    return cfg


def load_model(args, payload):
    ckpt = torch.load(args.checkpoint, map_location="cpu") if args.checkpoint else None
    if ckpt is not None and "cfg" in ckpt:
        cfg = GRAMConfig(**ckpt["cfg"])
    else:
        cfg = _fallback_config(args)
    model = GRAM(cfg)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"], strict=True)
        state = ckpt.get("ema", {}).get("shadow") if args.use_ema and "ema" in ckpt else None
        if state:
            merged = model.state_dict()
            merged.update(state)
            model.load_state_dict(merged, strict=True)
    model.to(args.device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(model, x, max_samples):
    preds = []
    for _ in range(max_samples):
        preds.append(model(x).argmax(-1).cpu())
    return torch.stack(preds, dim=1)


@torch.no_grad()
def eval_nqueens(args, payload):
    model = load_model(args, payload)
    x_all = payload["test_inputs_unique"]
    if args.max_examples:
        x_all = x_all[:args.max_examples]
    completions_all = [payload["completions"][tuple(x.tolist())] for x in x_all]
    rows = []
    bucket_rows = defaultdict(lambda: {f"coverage@{n}": [] for n in SAMPLE_COUNTS} | {"acc@1": []})
    queen_sum = 0.0
    keep_sum = 0.0
    given_sum = 0.0
    diag_sum = {"exact_n": 0.0, "rows_ok": 0.0, "cols_ok": 0.0, "diag_ok": 0.0, "valid_tokens": 0.0}

    for start in range(0, x_all.shape[0], args.batch_size):
        x = x_all[start : start + args.batch_size].to(args.device)
        completions = completions_all[start : start + args.batch_size]
        samples = sample_model(model, x, max(SAMPLE_COUNTS))
        pred1 = samples[:, 0]
        x_cpu = x.cpu()
        acc1 = nqueens_accuracy(pred1, x_cpu, args.n)
        diag = nqueens_diagnostics(pred1, x_cpu, args.n)
        for key, value in diag.items():
            diag_sum[key] += value * x.shape[0]
        queen_sum += (pred1 == QUEEN).sum().item()
        keep_sum += ((x_cpu == QUEEN) & (pred1 == QUEEN)).sum().item()
        given_sum += (x_cpu == QUEEN).sum().item()
        metrics = {"acc@1": acc1}
        for n_samples in SAMPLE_COUNTS:
            metrics[f"coverage@{n_samples}"] = nqueens_coverage(samples[:, :n_samples], x_cpu, completions, args.n)
        rows.append(metrics | {"count": x.shape[0]})

        for i, valid in enumerate(completions):
            b = len(valid)
            xi = x_cpu[i : i + 1]
            si = samples[i : i + 1]
            bucket_rows[b]["acc@1"].append(nqueens_accuracy(si[:, 0], xi, args.n))
            for n_samples in SAMPLE_COUNTS:
                bucket_rows[b][f"coverage@{n_samples}"].append(nqueens_coverage(si[:, :n_samples], xi, [valid], args.n))

    total = sum(r["count"] for r in rows)
    overall = {"task": "nqueens", "n": args.n, "num_examples": total}
    for key in ["acc@1"] + [f"coverage@{n}" for n in SAMPLE_COUNTS]:
        overall[key] = sum(r[key] * r["count"] for r in rows) / max(total, 1)
    overall["avg_queens@1"] = queen_sum / max(total, 1)
    overall["given_keep@1"] = keep_sum / max(given_sum, 1)
    for key, value in diag_sum.items():
        overall[f"{key}@1"] = value / max(total, 1)

    buckets = []
    for b, vals in sorted(bucket_rows.items()):
        row = {"num_solutions": b, "count": len(vals["acc@1"])}
        for key, xs in vals.items():
            row[key] = sum(xs) / max(len(xs), 1)
        buckets.append(row)
    return overall, buckets


@torch.no_grad()
def eval_graphcolor(args, payload):
    model = load_model(args, payload)
    x_all = payload["test_x"]
    graphs = payload["test_graphs"]
    if args.max_examples:
        x_all = x_all[:args.max_examples]
        graphs = graphs[:args.max_examples]
    colorings_all = [payload["colorings"][g] for g in graphs]
    adjs = torch.stack([upper_to_adj(g, args.n) for g in graphs], dim=0)
    rows = []
    bucket_rows = defaultdict(lambda: {f"coverage@{n}": [] for n in SAMPLE_COUNTS} | {"conflicts@1": []})

    for start in range(0, x_all.shape[0], args.batch_size):
        x = x_all[start : start + args.batch_size].to(args.device)
        adj = adjs[start : start + args.batch_size].to(args.device)
        colorings = colorings_all[start : start + args.batch_size]
        sample_tokens = []
        for _ in range(max(SAMPLE_COUNTS)):
            logits = model(x)
            sample_tokens.append(color_logits_to_tokens(logits, args.n).cpu())
        samples = torch.stack(sample_tokens, dim=1)
        conflicts = conflict_edges(samples[:, 0].to(args.device), adj).cpu()
        metrics = {"conflicts@1": conflicts.mean().item()}
        for n_samples in SAMPLE_COUNTS:
            metrics[f"coverage@{n_samples}"] = graph_coverage(samples[:, :n_samples], colorings, args.n)
        rows.append(metrics | {"count": x.shape[0]})

        for i, valid in enumerate(colorings):
            b = len(valid)
            bucket_rows[b]["conflicts@1"].append(float(conflicts[i].item()))
            for n_samples in SAMPLE_COUNTS:
                bucket_rows[b][f"coverage@{n_samples}"].append(graph_coverage(samples[i : i + 1, :n_samples], [valid], args.n))

    total = sum(r["count"] for r in rows)
    overall = {"task": "graph_coloring", "n": args.n, "num_examples": total}
    for key in ["conflicts@1"] + [f"coverage@{n}" for n in SAMPLE_COUNTS]:
        overall[key] = sum(r[key] * r["count"] for r in rows) / max(total, 1)

    buckets = []
    for b, vals in sorted(bucket_rows.items()):
        row = {"num_solutions": b, "count": len(vals["conflicts@1"])}
        for key, xs in vals.items():
            row[key] = sum(xs) / max(len(xs), 1)
        buckets.append(row)
    return overall, buckets


def write_outputs(overall, buckets, out_prefix: Path):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".buckets.csv")
    json_path.write_text(json.dumps({"overall": overall, "buckets": buckets}, indent=2, sort_keys=True))
    with csv_path.open("w", newline="") as f:
        if buckets:
            w = csv.DictWriter(f, fieldnames=list(buckets[0].keys()))
            w.writeheader()
            w.writerows(buckets)
    print(json.dumps(overall, indent=2, sort_keys=True))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["nqueens", "graphcolor"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--data-cache", type=Path, default=Path("data_cache"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-prefix", type=Path, default=None)
    args = ap.parse_args()

    if args.task == "nqueens":
        payload = torch.load(args.data_cache / f"nqueens_n{args.n}_seed0.pt", map_location="cpu")
        overall, buckets = eval_nqueens(args, payload)
        prefix = args.out_prefix or (args.data_cache / "analysis" / f"eval_nqueens_n{args.n}")
    else:
        matches = sorted(args.data_cache.glob(f"graphcolor_n{args.n}_p*_seed0.pt"))
        if not matches:
            raise FileNotFoundError(f"no graphcolor cache for n={args.n} under {args.data_cache}")
        payload = torch.load(matches[0], map_location="cpu")
        overall, buckets = eval_graphcolor(args, payload)
        prefix = args.out_prefix or (args.data_cache / "analysis" / f"eval_graphcolor_n{args.n}")
    write_outputs(overall, buckets, prefix)


if __name__ == "__main__":
    main()
