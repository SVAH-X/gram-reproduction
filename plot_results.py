"""Plotting helpers for the GRAM reproduction.

Uses SciencePlots styles ("science", "ieee") for publication-quality figures.

Produces three families of plots:

  1. Training curves from a *.log file (loss / recon / KL / acc / coverage).
       python plot_results.py curves --log gram_nqueens_n8.log

  2. Solution-count histograms (Figure 10 / 12 from the GRAM paper).
       python plot_results.py hist --data-dir data_cache

  3. Coverage vs. number of samples (Figure 15) from an eval_paper.py
     buckets.csv. Each row is a (num_solutions, count, acc@1, coverage@N) tuple.
       python plot_results.py coverage \\
              --buckets data_cache/analysis/eval_nqueens_n8.buckets.csv

All output figures are written under data_cache/analysis/figs/ unless
--out-dir is set.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401  (registers styles via side-effect)
    _STYLES = ["science", "ieee", "no-latex"]
except ImportError:
    print("[plot_results] SciencePlots not installed; falling back to default style.")
    _STYLES = []


def apply_style() -> None:
    if _STYLES:
        plt.style.use(_STYLES)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})


# --------------------------------------------------------------- training log

_LOG_LINE_OLD = re.compile(
    r"step\s+(?P<step>\d+)/\d+\s*\|\s*"
    r"loss\s+(?P<loss>[-\d.eE]+)\s+"
    r"recon\s+(?P<recon>[-\d.eE]+)\s+"
    r"kl\s+(?P<kl>[-\d.eE]+)\s+"
    r"halt\s+(?P<halt>[-\d.eE]+)\s+"
    r"lprm\s+(?P<lprm>[-\d.eE]+)\s+"
    r"r\s+(?P<r>[-\d.eE]+)\s+"
    r"acc\s+(?P<acc>[-\d.eE]+)\s+"
    r"gn\s+(?P<gn>[-\d.eE]+)"
)
_LOG_LINE_NEW = re.compile(
    r"step\s+(?P<step>\d+)/\d+\s*\|\s*"
    r"loss\s+(?P<loss>[-\d.eE]+)\s+"
    r"rp\s+(?P<recon>[-\d.eE]+)\s+"
    r"rq\s+(?P<recon_q>[-\d.eE]+)\s+"
    r"kl\s+(?P<kl>[-\d.eE]+).*?"
    r"halt\s+(?P<halt>[-\d.eE]+)\s+"
    r"lprm\s+(?P<lprm>[-\d.eE]+)\s+"
    r"r\s+(?P<r>[-\d.eE]+)\s+"
    r"acc_p\s+(?P<acc>[-\d.eE]+)\s+"
    r"acc_q\s+(?P<acc_q>[-\d.eE]+)\s+"
    r"gn\s+(?P<gn>[-\d.eE]+)"
)
_EVAL_LINE = re.compile(
    r">>\s*(?P<tag>raw|EMA)\s+test\s+n=(?P<n>\d+)\s+"
    r"(?:acc\s+(?P<acc>[-\d.eE]+)|conflicts\s+(?P<conflict>[-\d.eE]+))\s+"
    r"coverage@(?P<k>\d+)\s+(?P<cov>[-\d.eE]+)"
)


def parse_log(path: Path):
    train = defaultdict(list)
    eval_raw = defaultdict(list)
    eval_ema = defaultdict(list)
    last_step = 0
    with path.open() as f:
        for line in f:
            m = _LOG_LINE_NEW.search(line) or _LOG_LINE_OLD.search(line)
            if m:
                step = int(m["step"])
                last_step = step
                for key in ("loss", "recon", "kl", "halt", "lprm", "r", "acc", "gn"):
                    train[key].append((step, float(m[key])))
                if "recon_q" in m.groupdict() and m["recon_q"] is not None:
                    train["recon_q"].append((step, float(m["recon_q"])))
                if "acc_q" in m.groupdict() and m["acc_q"] is not None:
                    train["acc_q"].append((step, float(m["acc_q"])))
                continue
            e = _EVAL_LINE.search(line)
            if e:
                tag = e["tag"]
                bucket = eval_raw if tag == "raw" else eval_ema
                bucket["step"].append(last_step)
                if e["acc"] is not None:
                    bucket["acc"].append(float(e["acc"]))
                if e["conflict"] is not None:
                    bucket["conflict"].append(float(e["conflict"]))
                bucket["cov"].append(float(e["cov"]))
    return train, eval_raw, eval_ema


def plot_curves(log_path: Path, out_dir: Path) -> None:
    train, eval_raw, eval_ema = parse_log(log_path)
    if not train:
        raise SystemExit(f"no training-step lines parsed out of {log_path}")
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4))
    panels = [
        ("loss",  "loss"),
        ("recon", "reconstruction NLL"),
        ("kl",    "KL(q || p)"),
        ("acc",   "train per-position accuracy"),
    ]
    for ax, (key, ylab) in zip(axes.flat, panels):
        xs, ys = zip(*train[key])
        ax.plot(xs, ys, linewidth=0.8)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylab)
        if key in ("loss", "recon", "kl"):
            ax.set_yscale("log")
    fig.suptitle(log_path.stem)
    fig.tight_layout()
    out = out_dir / f"{log_path.stem}_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    if eval_raw.get("step"):
        fig, ax = plt.subplots(figsize=(4.0, 2.6))
        if eval_raw.get("acc"):
            ax.plot(eval_raw["step"], eval_raw["acc"], "o-", label="raw acc", markersize=2)
            ax.plot(eval_ema["step"], eval_ema["acc"], "s-", label="EMA acc", markersize=2)
            ax.set_ylabel("test accuracy")
        else:
            ax.plot(eval_raw["step"], eval_raw["conflict"], "o-", label="raw", markersize=2)
            ax.plot(eval_ema["step"], eval_ema["conflict"], "s-", label="EMA", markersize=2)
            ax.set_ylabel("test conflicts (lower is better)")
        ax2 = ax.twinx()
        ax2.plot(eval_raw["step"], eval_raw["cov"], "x--", color="C2", label="raw cov", markersize=3)
        ax2.plot(eval_ema["step"], eval_ema["cov"], "+--", color="C3", label="EMA cov", markersize=3)
        ax2.set_ylabel("coverage@20")
        ax.set_xlabel("training step")
        ax.legend(loc="upper left", fontsize=6)
        ax2.legend(loc="lower right", fontsize=6)
        fig.tight_layout()
        out = out_dir / f"{log_path.stem}_eval.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


# --------------------------------------------------------------- histograms

def plot_solution_histograms(data_dir: Path, out_dir: Path) -> None:
    analysis = data_dir / "analysis"
    if not analysis.exists():
        raise SystemExit("Run `python analyze_data.py --data-dir data_cache` first.")
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = sorted(analysis.glob("*_train_solution_hist.csv"))
    if not pairs:
        raise SystemExit(f"no *_train_solution_hist.csv under {analysis}")
    n_per = 2
    fig, axes = plt.subplots(1, len(pairs), figsize=(2.6 * len(pairs), 2.4), sharey=False)
    if len(pairs) == 1:
        axes = [axes]
    for ax, train_csv in zip(axes, pairs):
        prefix = train_csv.name.replace("_train_solution_hist.csv", "")
        test_csv = analysis / f"{prefix}_test_solution_hist.csv"
        for csv_path, label, color in (
            (train_csv, "train", "C0"),
            (test_csv, "test", "C1"),
        ):
            xs, ys = [], []
            with csv_path.open() as f:
                r = csv.reader(f); next(r)
                for k, v in r:
                    xs.append(int(k)); ys.append(int(v))
            if xs:
                ax.bar(xs, ys, alpha=0.6, label=label, color=color, width=0.9)
        ax.set_xlabel("# solutions per input")
        ax.set_ylabel("count")
        ax.set_title(prefix)
        ax.legend(fontsize=6)
    fig.tight_layout()
    out = out_dir / "solution_histograms.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------- coverage

def plot_coverage_by_bucket(buckets_csv: Path, out_dir: Path) -> None:
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with buckets_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        raise SystemExit(f"empty buckets file {buckets_csv}")

    cov_keys = sorted(
        k for k in rows[0].keys() if k.startswith("coverage@")
    )
    if not cov_keys:
        raise SystemExit("no coverage@N columns in buckets file")

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    xs = [int(row["num_solutions"]) for row in rows]
    for k in cov_keys:
        ys = [float(row[k]) for row in rows]
        n = k.split("@")[1]
        ax.plot(xs, ys, "o-", label=f"N={n}", markersize=2.5, linewidth=0.8)
    ax.set_xlabel("# ground-truth solutions per input")
    ax.set_ylabel("coverage")
    ax.set_title(buckets_csv.stem)
    ax.legend(fontsize=6)
    fig.tight_layout()
    out = out_dir / f"{buckets_csv.stem}_coverage.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------------------------------- entry

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_curves = sub.add_parser("curves")
    p_curves.add_argument("--log", type=Path, required=True)
    p_curves.add_argument("--out-dir", type=Path, default=Path("data_cache/analysis/figs"))

    p_hist = sub.add_parser("hist")
    p_hist.add_argument("--data-dir", type=Path, default=Path("data_cache"))
    p_hist.add_argument("--out-dir", type=Path, default=Path("data_cache/analysis/figs"))

    p_cov = sub.add_parser("coverage")
    p_cov.add_argument("--buckets", type=Path, required=True)
    p_cov.add_argument("--out-dir", type=Path, default=Path("data_cache/analysis/figs"))

    args = ap.parse_args()
    if args.cmd == "curves":
        plot_curves(args.log, args.out_dir)
    elif args.cmd == "hist":
        plot_solution_histograms(args.data_dir, args.out_dir)
    elif args.cmd == "coverage":
        plot_coverage_by_bucket(args.buckets, args.out_dir)


if __name__ == "__main__":
    main()
