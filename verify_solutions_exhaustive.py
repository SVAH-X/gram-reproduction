"""Exhaustive solution-validity check across all four cache files.

Unlike verify_datasets.py (which spot-checks 256 train graphs), this script
runs the constraint validator over every cached (input, target) pair in
both splits and prints exact counts. If any cached target is not a true
solution, this script will say so.

It also dumps three concrete examples per task so you can read the boards /
graphs by eye.
"""

from __future__ import annotations

from pathlib import Path

import torch

from data_nqueens import (
    EMPTY,
    QUEEN,
    is_valid_completion,
)
from data_graph_coloring import (
    COLOR_BASE,
    EDGE,
    canonical_colorings,
    upper_pairs,
    upper_to_adj,
)


def verify_nqueens_exhaustive(path: Path) -> None:
    print(f"\n== {path.name} ==")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    n = int(payload["n"])
    train_x = payload["train_x"]
    train_y = payload["train_y"]
    completions = payload["completions"]
    test_inputs = payload["test_inputs_unique"]

    # 1. Check every materialized (input, target) pair.
    n_pairs = train_x.shape[0]
    bad_solution = 0
    bad_subset = 0
    bad_queens = 0
    for i in range(n_pairs):
        partial = tuple(train_x[i].tolist())
        target = tuple(train_y[i].tolist())
        n_target_queens = target.count(QUEEN)
        if n_target_queens != n:
            bad_queens += 1
            continue
        for pos, t in enumerate(partial):
            if t == QUEEN and target[pos] != QUEEN:
                bad_subset += 1
                break
        if not is_valid_completion(target, partial, n):
            bad_solution += 1
    print(f"  {n_pairs:>7d} train (input,target) pairs")
    print(f"    targets with != N queens : {bad_queens}")
    print(f"    targets violating input  : {bad_subset}")
    print(f"    targets failing constraint check : {bad_solution}")

    # 2. Check every cached completion (across the FULL completions dict
    #    that covers both train and test partials).
    total_completions = 0
    bad_in_dict = 0
    for partial, comps in completions.items():
        for c in comps:
            total_completions += 1
            if not is_valid_completion(c, partial, n):
                bad_in_dict += 1
    print(f"  {total_completions:>7d} cached completions across the full dict")
    print(f"    failing constraint check : {bad_in_dict}")

    # 3. Concrete eyeball example.
    show_example_nqueens(payload, n)


def show_example_nqueens(payload, n: int) -> None:
    train_x = payload["train_x"]
    train_y = payload["train_y"]
    idx = 0
    partial = train_x[idx].tolist()
    target = train_y[idx].tolist()
    print(f"  example pair (idx={idx}):")
    print_board("    input  ", partial, n)
    print_board("    target ", target, n)
    rows, cols, d1, d2 = set(), set(), set(), set()
    coords = []
    for pos, t in enumerate(target):
        if t == QUEEN:
            r, c = pos // n, pos % n
            coords.append((r, c))
            rows.add(r); cols.add(c); d1.add(r - c); d2.add(r + c)
    print(f"    queens at: {coords}")
    print(f"    rows distinct: {len(rows) == n}, cols distinct: {len(cols) == n}, "
          f"diag- distinct: {len(d1) == n}, diag+ distinct: {len(d2) == n}")
    for r, c in coords:
        in_partial = partial[r * n + c] == QUEEN
        if in_partial:
            print(f"      ({r},{c}) was already in input")


def print_board(label: str, board, n: int) -> None:
    print(f"{label} ({n}x{n}):")
    for r in range(n):
        row = "".join(
            "Q" if board[r * n + c] == QUEEN else "."
            for c in range(n)
        )
        print(f"      {row}")


def is_valid_3coloring(coloring, adj) -> bool:
    n = len(coloring)
    if any(c < 0 or c >= 3 for c in coloring):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j].item() and coloring[i] == coloring[j]:
                return False
    return True


def is_canonical(coloring) -> bool:
    m = -1
    for c in coloring:
        if c > m + 1:
            return False
        m = max(m, c)
    return True


def verify_graphcolor_exhaustive(path: Path) -> None:
    print(f"\n== {path.name} ==")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    n = int(payload["n"])
    train_graphs = payload["train_graphs"]
    test_graphs = payload["test_graphs"]
    colorings = payload["colorings"]

    total = 0
    bad_valid = 0
    bad_canon = 0
    for g in train_graphs + test_graphs:
        adj = upper_to_adj(g, n)
        for c in colorings[g]:
            total += 1
            if not is_valid_3coloring(c, adj):
                bad_valid += 1
            if not is_canonical(c):
                bad_canon += 1
    print(f"  {len(train_graphs)} train + {len(test_graphs)} test graphs "
          f"= {total} total cached colorings")
    print(f"    not a valid 3-coloring of its adjacency : {bad_valid}")
    print(f"    not canonical (first-occurrence order)  : {bad_canon}")

    # Independent re-enumeration on 200 random graphs: confirm cached set ==
    # complete canonical set (no missing colorings either).
    rng = torch.Generator().manual_seed(0)
    pool = train_graphs + test_graphs
    pick = torch.randperm(len(pool), generator=rng)[: min(200, len(pool))].tolist()
    set_mismatch = 0
    for idx in pick:
        g = pool[idx]
        cached = set(colorings[g])
        enumerated = set(canonical_colorings(g, n))
        if cached != enumerated:
            set_mismatch += 1
    print(f"  cached set == re-enumerated set on 200 graphs : "
          f"{200 - set_mismatch}/200 match")

    show_example_graphcolor(payload, n)


def show_example_graphcolor(payload, n: int) -> None:
    train_graphs = payload["train_graphs"]
    colorings = payload["colorings"]
    g = train_graphs[0]
    adj = upper_to_adj(g, n)
    edges = [(i, j) for (i, j), e in zip(upper_pairs(n), g) if e]
    sols = colorings[g]
    sol = sols[0]
    print(f"  example graph (n={n}, idx=0):")
    print(f"    {len(edges)} edges: {edges[:10]}{' ...' if len(edges) > 10 else ''}")
    print(f"    one valid canonical coloring: {sol}")
    color_names = {0: "R", 1: "B", 2: "G"}
    for u, v in edges:
        cu, cv = sol[u], sol[v]
        assert cu != cv, f"edge ({u},{v}) has matching colors!"
    print(f"    -> verified: all {len(edges)} edges are between differently-colored nodes")
    # Show node-color mapping.
    print(f"    nodes -> colors: ", end="")
    print(", ".join(f"{i}:{color_names[c]}" for i, c in enumerate(sol)))
    print(f"    total canonical 3-colorings of this graph: {len(sols)}")


def main():
    cache = Path("data_cache")
    for f in sorted(cache.glob("nqueens_n*_seed*.pt")):
        verify_nqueens_exhaustive(f)
    for f in sorted(cache.glob("graphcolor_n*_p*_seed*.pt")):
        verify_graphcolor_exhaustive(f)


if __name__ == "__main__":
    main()
