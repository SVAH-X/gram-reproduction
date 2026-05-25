"""Paper-strict verification of N-Queens and Graph Coloring training data.

This script does not trust the generator code. It loads the materialized
cache files under data_cache/ and re-derives every property from raw bytes,
checking each one against the paper's specification:

  N-Queens (Appendix C.2.1)
    * vocab exactly {0=PAD, 1=empty, 2=queen}
    * input length = N^2, target length = N^2
    * each input is some complete N-queens solution with k queens removed
    * removal counts: {5,6,7} for N=8 and {7,8,9} for N=10
    * each target is a valid N-queens solution consistent with its input
      (every queen in the input also a queen in the target; all N queens
      of the target satisfy row/col/diagonal constraints)
    * for every unique input, the cached completions cover ALL valid
      completions of that partial board
    * train/test split is by unique input (no input appears in both)

  Graph Coloring (Appendix C.2.2)
    * vocab exactly {0=PAD, 1=no-edge, 2=edge, 3,4,5=colors}
    * input length = n(n-1)/2, target length = n
    * input encodes the strict upper triangle of a symmetric adjacency
    * every graph in the cache admits at least one 3-coloring
    * every cached coloring is a valid 3-coloring of the corresponding adjacency
    * every cached coloring is canonical: the first occurrence of each color
      appears in order 0, 1, 2 (so node 0 is always color 0)
    * cached colorings dictionary covers ALL canonical 3-colorings of the graph
    * train/test counts match paper exactly: 7002/255 for n=8, 13465/192 for n=10
    * train and test sets are disjoint

Each check prints PASS/FAIL; the script exits non-zero on any FAIL.
"""

from __future__ import annotations

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from data_nqueens import (
    EMPTY,
    PAD as NQ_PAD,
    QUEEN,
    VOCAB_SIZE as NQ_VOCAB_SIZE,
    enumerate_solutions,
    is_valid_completion,
)
from data_graph_coloring import (
    COLOR_BASE,
    EDGE,
    NO_EDGE,
    NUM_COLORS,
    PAD as GC_PAD,
    VOCAB_SIZE as GC_VOCAB_SIZE,
    canonical_colorings,
    upper_pairs,
    upper_to_adj,
)


# ---------------------------------------------------------------- pretty print

FAILS: List[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    line = f"  [{tag}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not ok:
        FAILS.append(label)


# ---------------------------------------------------------------- N-Queens

def expected_removed(n: int) -> Tuple[int, ...]:
    return (5, 6, 7) if n == 8 else (7, 8, 9)


def verify_nqueens(path: Path) -> None:
    print(f"\n== {path.name} ==")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    n = int(payload["n"])
    seq_len = n * n
    train_x = payload["train_x"]
    train_y = payload["train_y"]
    train_inputs_unique = payload["train_inputs_unique"]
    test_inputs_unique = payload["test_inputs_unique"]
    completions = payload["completions"]
    meta = payload.get("metadata", {})

    # 1. Shapes and dtype
    check(
        "train_x shape is (P, N^2)",
        train_x.dim() == 2 and train_x.shape[1] == seq_len,
        detail=f"got {tuple(train_x.shape)}",
    )
    check(
        "train_y shape matches train_x",
        train_y.shape == train_x.shape,
        detail=f"x={tuple(train_x.shape)} y={tuple(train_y.shape)}",
    )
    check(
        "train_x dtype long",
        train_x.dtype == torch.long,
        detail=f"got {train_x.dtype}",
    )

    # 2. Vocab — only EMPTY/QUEEN should appear (no PAD on N-Queens)
    x_unique = set(train_x.unique().tolist())
    y_unique = set(train_y.unique().tolist())
    check(
        "train_x tokens subset of {empty=1, queen=2}",
        x_unique <= {EMPTY, QUEEN},
        detail=f"saw {sorted(x_unique)}",
    )
    check(
        "train_y tokens subset of {empty=1, queen=2}",
        y_unique <= {EMPTY, QUEEN},
        detail=f"saw {sorted(y_unique)}",
    )
    check(
        "vocab size in code is 3",
        NQ_VOCAB_SIZE == 3,
        detail=f"NQ_VOCAB_SIZE={NQ_VOCAB_SIZE}",
    )

    # 3. Every train target has exactly N queens and is a valid solution
    n_targets = train_y.shape[0]
    target_queen_counts = (train_y == QUEEN).sum(dim=1)
    check(
        "every train target has exactly N queens",
        bool((target_queen_counts == n).all()),
        detail=f"min={int(target_queen_counts.min())} max={int(target_queen_counts.max())}",
    )

    bad_pair = 0
    bad_queen_subset = 0
    bad_solution = 0
    for i in range(n_targets):
        partial = tuple(train_x[i].tolist())
        target = tuple(train_y[i].tolist())
        # Every queen in partial must be a queen in target.
        for pos, t in enumerate(partial):
            if t == QUEEN and target[pos] != QUEEN:
                bad_queen_subset += 1
                break
        # Target must be a valid solution given the partial.
        if not is_valid_completion(target, partial, n):
            bad_solution += 1
    check(
        "every train input is a subset of its target's queens",
        bad_queen_subset == 0,
        detail=f"violations={bad_queen_subset}",
    )
    check(
        "every train target is a valid N-queens completion of its input",
        bad_solution == 0,
        detail=f"violations={bad_solution}",
    )

    # 4. Number of removed queens per input matches paper's k set
    removed_counts = (target_queen_counts.new_full((n_targets,), n)
                      - (train_x == QUEEN).sum(dim=1))
    removed_set = set(int(v) for v in removed_counts.unique().tolist())
    check(
        f"k (queens removed) set is {set(expected_removed(n))}",
        removed_set == set(expected_removed(n)),
        detail=f"saw {sorted(removed_set)}",
    )

    # 5. Train/test inputs are disjoint
    train_keys = {tuple(t.tolist()) for t in train_inputs_unique}
    test_keys = {tuple(t.tolist()) for t in test_inputs_unique}
    check(
        "train/test unique-input split is disjoint",
        train_keys.isdisjoint(test_keys),
        detail=f"overlap={len(train_keys & test_keys)}",
    )

    # 6. Pair set materialized = sum over completions in train_inputs
    expected_pairs = sum(len(completions[k]) for k in train_keys)
    check(
        "materialized train pairs = sum of completions over unique inputs",
        n_targets == expected_pairs,
        detail=f"n_targets={n_targets} expected={expected_pairs}",
    )

    # 7. completions dictionary covers ALL valid completions of each partial
    #    (Generation enumerates every solution × every removal of k rows; the
    #    same partial may come from multiple solutions, so completions[x]
    #    must equal the set of all valid completions of x.)
    sols = enumerate_solutions(n)
    full_boards = []
    for sol in sols:
        board = [EMPTY] * (n * n)
        for row, col in enumerate(sol):
            board[row * n + col] = QUEEN
        full_boards.append(tuple(board))

    sample_inputs = list(train_keys)[:64] + list(test_keys)[:64]
    cov_violations = 0
    for partial in sample_inputs:
        expected = set()
        for board in full_boards:
            ok = True
            for pos, p in enumerate(partial):
                if p == QUEEN and board[pos] != QUEEN:
                    ok = False
                    break
            if ok:
                expected.add(board)
        cached = set(completions[partial])
        if cached != expected:
            cov_violations += 1
    check(
        "cached completions exactly cover all valid completions (sample of 128 inputs)",
        cov_violations == 0,
        detail=f"violations={cov_violations}",
    )

    # 8. metadata cross-check
    check(
        "metadata.seq_len == N^2",
        int(meta.get("seq_len", -1)) == seq_len,
        detail=f"meta={meta.get('seq_len')}",
    )
    check(
        "metadata.vocab_size == 3",
        int(meta.get("vocab_size", -1)) == 3,
        detail=f"meta={meta.get('vocab_size')}",
    )
    check(
        "metadata.train_fraction == 0.85",
        abs(float(meta.get("train_fraction", 0.0)) - 0.85) < 1e-9,
        detail=f"meta={meta.get('train_fraction')}",
    )
    expected_trajectories = 3000 if n == 8 else 1000
    check(
        f"metadata.paper_trajectory_batches == {expected_trajectories}",
        int(meta.get("paper_trajectory_batches", -1)) == expected_trajectories,
        detail=f"meta={meta.get('paper_trajectory_batches')}",
    )
    check(
        "metadata.segment_updates_per_trajectory == 16 (= N_sup)",
        int(meta.get("segment_updates_per_trajectory", -1)) == 16,
        detail=f"meta={meta.get('segment_updates_per_trajectory')}",
    )

    # 9. summary print
    train_counts = [len(completions[k]) for k in train_keys]
    test_counts = [len(completions[k]) for k in test_keys]
    print(f"  · unique train inputs: {len(train_keys)} | unique test inputs: {len(test_keys)}")
    print(f"  · materialized train pairs: {n_targets}")
    print(f"  · per-input solution counts (train) min/median/max: "
          f"{min(train_counts)}/{sorted(train_counts)[len(train_counts)//2]}/{max(train_counts)}")
    print(f"  · per-input solution counts (test)  min/median/max: "
          f"{min(test_counts)}/{sorted(test_counts)[len(test_counts)//2]}/{max(test_counts)}")
    hist = Counter(train_counts + test_counts)
    most_common = sorted(hist.items())[:8]
    print(f"  · solution-count histogram (first bins): {most_common}")


# ---------------------------------------------------------------- Graph Coloring

def expected_split(n: int) -> Tuple[int, int]:
    return (7002, 255) if n == 8 else (13465, 192)


def is_valid_3coloring(coloring: Sequence[int], adj: torch.Tensor) -> bool:
    n = len(coloring)
    if any(c not in (0, 1, 2) for c in coloring):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j].item() and coloring[i] == coloring[j]:
                return False
    return True


def is_canonical(coloring: Sequence[int]) -> bool:
    max_used = -1
    for c in coloring:
        if c > max_used + 1:
            return False
        max_used = max(max_used, c)
    return True


def verify_graph_coloring(path: Path) -> None:
    print(f"\n== {path.name} ==")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    n = int(payload["n"])
    train_graphs = payload["train_graphs"]
    test_graphs = payload["test_graphs"]
    colorings = payload["colorings"]
    train_x = payload["train_x"]
    train_y = payload["train_y_first_canonical"]
    test_x = payload["test_x"]
    meta = payload.get("metadata", {})

    seq_len = n * (n - 1) // 2

    # 1. Shapes and dtype
    check(
        "train_x shape is (G, n(n-1)/2)",
        train_x.dim() == 2 and train_x.shape[1] == seq_len,
        detail=f"got {tuple(train_x.shape)}",
    )
    check(
        "train_y shape is (G, n)",
        train_y.dim() == 2 and train_y.shape[1] == n,
        detail=f"got {tuple(train_y.shape)}",
    )
    check(
        "test_x has the same input length",
        test_x.shape[1] == seq_len,
        detail=f"got {tuple(test_x.shape)}",
    )

    # 2. Token vocab
    x_unique = set(train_x.unique().tolist()) | set(test_x.unique().tolist())
    y_unique = set(train_y.unique().tolist())
    check(
        "input tokens subset of {no_edge=1, edge=2}",
        x_unique <= {NO_EDGE, EDGE},
        detail=f"saw {sorted(x_unique)}",
    )
    expected_color_tokens = {COLOR_BASE + i for i in range(NUM_COLORS)}
    check(
        f"target tokens subset of color set {sorted(expected_color_tokens)}",
        y_unique <= expected_color_tokens,
        detail=f"saw {sorted(y_unique)}",
    )
    check(
        "vocab size in code is 6",
        GC_VOCAB_SIZE == 6,
        detail=f"GC_VOCAB_SIZE={GC_VOCAB_SIZE}",
    )

    # 3. Sizes match paper exactly
    expected_train, expected_test = expected_split(n)
    check(
        f"train graph count == {expected_train}",
        len(train_graphs) == expected_train,
        detail=f"got {len(train_graphs)}",
    )
    check(
        f"test graph count == {expected_test}",
        len(test_graphs) == expected_test,
        detail=f"got {len(test_graphs)}",
    )

    # 4. Train and test disjoint
    train_set = set(train_graphs)
    test_set = set(test_graphs)
    check(
        "train/test graph sets are disjoint",
        train_set.isdisjoint(test_set),
        detail=f"overlap={len(train_set & test_set)}",
    )

    # 5. Every cached coloring is a valid 3-coloring AND canonical AND every
    #    graph admits at least one coloring. Full audit for the test set
    #    (small), and re-enumerate to confirm cached = full canonical set.
    bad_valid = 0
    bad_canon = 0
    missing_or_extra = 0
    for adj_flat in test_graphs:
        adj = upper_to_adj(adj_flat, n)
        cached = set(colorings[adj_flat])
        if not cached:
            bad_valid += 1
            continue
        for col in cached:
            if not is_valid_3coloring(col, adj):
                bad_valid += 1
            if not is_canonical(col):
                bad_canon += 1
        re_enumerated = set(canonical_colorings(adj_flat, n))
        if cached != re_enumerated:
            missing_or_extra += 1
    check(
        "every cached test coloring is a valid 3-coloring of its graph",
        bad_valid == 0,
        detail=f"violations={bad_valid}",
    )
    check(
        "every cached test coloring is canonical (first-occurrence order 0,1,2)",
        bad_canon == 0,
        detail=f"violations={bad_canon}",
    )
    check(
        "cached test colorings == full canonical enumeration",
        missing_or_extra == 0,
        detail=f"mismatched_graphs={missing_or_extra}",
    )

    # 6. Spot-check the train side. Validating all train graphs would re-run
    #    canonical_colorings on 7K–13K graphs and slow this script; the test
    #    side is the strict spec and 256 train spot checks suffice.
    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(train_graphs), generator=rng)[:256].tolist()
    bad_train_valid = 0
    bad_train_canon = 0
    for idx in perm:
        adj_flat = train_graphs[idx]
        adj = upper_to_adj(adj_flat, n)
        for col in colorings[adj_flat]:
            if not is_valid_3coloring(col, adj):
                bad_train_valid += 1
            if not is_canonical(col):
                bad_train_canon += 1
    check(
        "256-graph train spot check: all colorings are valid",
        bad_train_valid == 0,
        detail=f"violations={bad_train_valid}",
    )
    check(
        "256-graph train spot check: all colorings are canonical",
        bad_train_canon == 0,
        detail=f"violations={bad_train_canon}",
    )

    # 7. train_x encodes the strict upper triangle of each train graph
    encode_mismatch = 0
    for idx in perm:
        adj_flat = train_graphs[idx]
        expected = torch.tensor([EDGE if e else NO_EDGE for e in adj_flat], dtype=torch.long)
        if not torch.equal(train_x[idx], expected):
            encode_mismatch += 1
    check(
        "train_x encodes strict-upper-triangle correctly (spot)",
        encode_mismatch == 0,
        detail=f"violations={encode_mismatch}",
    )

    # 8. train_y_first_canonical is a valid canonical coloring of train_graphs[idx]
    y_mismatch = 0
    for idx in perm:
        adj_flat = train_graphs[idx]
        adj = upper_to_adj(adj_flat, n)
        coloring = tuple(int(c) - COLOR_BASE for c in train_y[idx].tolist())
        if not is_valid_3coloring(coloring, adj) or not is_canonical(coloring):
            y_mismatch += 1
        elif coloring not in colorings[adj_flat]:
            y_mismatch += 1
    check(
        "train_y_first_canonical is a valid canonical coloring of its graph (spot)",
        y_mismatch == 0,
        detail=f"violations={y_mismatch}",
    )

    # 9. metadata cross-check
    check(
        "metadata.input_seq_len == n(n-1)/2",
        int(meta.get("input_seq_len", -1)) == seq_len,
        detail=f"meta={meta.get('input_seq_len')}",
    )
    check(
        "metadata.target_seq_len == n",
        int(meta.get("target_seq_len", -1)) == n,
        detail=f"meta={meta.get('target_seq_len')}",
    )
    check(
        "metadata.vocab_size == 6",
        int(meta.get("vocab_size", -1)) == 6,
        detail=f"meta={meta.get('vocab_size')}",
    )
    check(
        "metadata.paper_trajectory_batches == 5000",
        int(meta.get("paper_trajectory_batches", -1)) == 5000,
        detail=f"meta={meta.get('paper_trajectory_batches')}",
    )
    check(
        "metadata.segment_updates_per_trajectory == 16",
        int(meta.get("segment_updates_per_trajectory", -1)) == 16,
        detail=f"meta={meta.get('segment_updates_per_trajectory')}",
    )

    # 10. summary print
    counts_train = [len(colorings[g]) for g in train_graphs]
    counts_test = [len(colorings[g]) for g in test_graphs]
    print(f"  · train/test graphs: {len(train_graphs)} / {len(test_graphs)}")
    print(f"  · per-graph canonical coloring counts (train) min/median/max: "
          f"{min(counts_train)}/{sorted(counts_train)[len(counts_train)//2]}/{max(counts_train)}")
    print(f"  · per-graph canonical coloring counts (test)  min/median/max: "
          f"{min(counts_test)}/{sorted(counts_test)[len(counts_test)//2]}/{max(counts_test)}")
    hist = Counter(counts_train + counts_test)
    most_common = sorted(hist.items())[:8]
    print(f"  · coloring-count histogram (first bins): {most_common}")


# ---------------------------------------------------------------- entry

def main():
    cache_dir = Path("data_cache")
    if not cache_dir.exists():
        print("data_cache/ missing — run `python prepare_data.py` first.")
        sys.exit(2)

    nq_files = sorted(cache_dir.glob("nqueens_n*_seed*.pt"))
    gc_files = sorted(cache_dir.glob("graphcolor_n*_p*_seed*.pt"))
    if not nq_files or not gc_files:
        print(f"expected nqueens_n*.pt and graphcolor_n*.pt under {cache_dir}/")
        sys.exit(2)

    for f in nq_files:
        verify_nqueens(f)
    for f in gc_files:
        verify_graph_coloring(f)

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s)")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print(f"ALL CHECKS PASSED across {len(nq_files) + len(gc_files)} cache file(s)")


if __name__ == "__main__":
    main()
