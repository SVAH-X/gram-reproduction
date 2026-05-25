"""Paper-aligned Graph Coloring data and metrics for GRAM.

Paper task definition:
  * 3-coloring with n in {8, 10}.
  * Graphs are Erdős-Rényi and only 3-colorable graphs are kept.
  * Inputs encode the strict upper triangle of the adjacency matrix.
  * Outputs are length-n node-color sequences, not padded adjacency-length
    targets.
  * Tokens: 0=PAD, 1=no-edge, 2=edge, 3/4/5=colors.
  * All valid 3-colorings are enumerated and canonicalized to remove color
    permutation duplicates.
  * Report conflict edges and coverage over unique valid colorings.

The paper gives final graph counts: 7002 train / 255 test for n=8 and
13465 train / 192 test for n=10. This file reproduces those counts locally.
The exact edge probability used by the paper is not stated; keep --p-edge
fixed across runs and materialize the generated split explicitly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


PAD = 0
NO_EDGE = 1
EDGE = 2
COLOR_BASE = 3
NUM_COLORS = 3
VOCAB_SIZE = 6


AdjFlat = Tuple[int, ...]
Coloring = Tuple[int, ...]


def default_split_sizes(n: int) -> Tuple[int, int]:
    if n == 8:
        return 7002, 255
    if n == 10:
        return 13465, 192
    raise ValueError("The paper specifies Graph Coloring only for n=8 and n=10.")


def upper_pairs(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def random_graph_upper(n: int, p_edge: float, gen: torch.Generator) -> AdjFlat:
    vals = (torch.rand(len(upper_pairs(n)), generator=gen) < p_edge).long()
    return tuple(int(v.item()) for v in vals)


def upper_to_adj(adj_flat: Sequence[int], n: int) -> torch.Tensor:
    adj = torch.zeros(n, n, dtype=torch.long)
    for value, (i, j) in zip(adj_flat, upper_pairs(n)):
        if value:
            adj[i, j] = 1
            adj[j, i] = 1
    return adj


def canonical_colorings(adj_flat: Sequence[int], n: int) -> Tuple[Coloring, ...]:
    """Enumerate canonical 3-colorings.

    Canonicalization removes color-permutation duplicates by allowing a new
    color label only after all lower labels have appeared. The first node is
    therefore always color 0 when n > 0.
    """
    adj = upper_to_adj(adj_flat, n)
    nbrs = [[j for j in range(n) if adj[i, j].item()] for i in range(n)]
    colors = [-1] * n
    out: List[Coloring] = []

    def backtrack(node: int, max_used: int) -> None:
        if node == n:
            out.append(tuple(colors))
            return
        limit = min(NUM_COLORS - 1, max_used + 1)
        for color in range(limit + 1):
            ok = True
            for nb in nbrs[node]:
                if colors[nb] == color:
                    ok = False
                    break
            if not ok:
                continue
            colors[node] = color
            backtrack(node + 1, max(max_used, color))
            colors[node] = -1

    backtrack(0, -1)
    return tuple(out)


@dataclass(frozen=True)
class GraphColoringBuild:
    n: int
    train_graphs: List[AdjFlat]
    test_graphs: List[AdjFlat]
    colorings: Dict[AdjFlat, Tuple[Coloring, ...]]
    p_edge: float

    @property
    def train_size(self) -> int:
        return len(self.train_graphs)

    @property
    def test_size(self) -> int:
        return len(self.test_graphs)


def build_graph_coloring(
    n: int,
    *,
    p_edge: float = 0.4,
    train_size: int | None = None,
    test_size: int | None = None,
    seed: int = 0,
    cache_dir: str | Path | None = None,
) -> GraphColoringBuild:
    default_train, default_test = default_split_sizes(n)
    train_size = default_train if train_size is None else train_size
    test_size = default_test if test_size is None else test_size
    total = train_size + test_size
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"graphcolor_n{n}_p{p_edge:g}_seed{seed}_total{total}.pt"
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            return GraphColoringBuild(
                n=payload["n"],
                train_graphs=payload["train_graphs"],
                test_graphs=payload["test_graphs"],
                colorings=payload["colorings"],
                p_edge=payload["p_edge"],
            )

    gen = torch.Generator().manual_seed(seed)
    seen = set()
    graphs: List[AdjFlat] = []
    colorings: Dict[AdjFlat, Tuple[Coloring, ...]] = {}

    while len(graphs) < total:
        adj_flat = random_graph_upper(n, p_edge, gen)
        if adj_flat in seen:
            continue
        seen.add(adj_flat)
        sols = canonical_colorings(adj_flat, n)
        if not sols:
            continue
        graphs.append(adj_flat)
        colorings[adj_flat] = sols

    rng = random.Random(seed)
    rng.shuffle(graphs)
    train_graphs = graphs[:train_size]
    test_graphs = graphs[train_size:]
    build = GraphColoringBuild(
        n=n,
        train_graphs=train_graphs,
        test_graphs=test_graphs,
        colorings=colorings,
        p_edge=p_edge,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "n": n,
                "train_graphs": train_graphs,
                "test_graphs": test_graphs,
                "colorings": colorings,
                "p_edge": p_edge,
            },
            cache_path,
        )
    return build


def _int_tuple(value) -> Tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple(int(v) for v in value)


def load_graph_coloring_cache(path: str | Path) -> GraphColoringBuild:
    """Load an explicitly materialized Graph Coloring dataset from prepare_data.py."""
    payload = torch.load(Path(path), map_location="cpu")
    if payload.get("task") != "graph_coloring":
        raise ValueError(f"Expected a graph_coloring cache file, got task={payload.get('task')!r}.")

    colorings = {
        _int_tuple(adj_flat): tuple(_int_tuple(coloring) for coloring in sols)
        for adj_flat, sols in payload["colorings"].items()
    }
    metadata = payload.get("metadata", {})
    return GraphColoringBuild(
        n=int(payload["n"]),
        train_graphs=[_int_tuple(g) for g in payload["train_graphs"]],
        test_graphs=[_int_tuple(g) for g in payload["test_graphs"]],
        colorings=colorings,
        p_edge=float(metadata.get("p_edge", payload.get("p_edge", 0.4))),
    )


def graph_to_tensors(adj_flat: AdjFlat, coloring: Coloring, n: int):
    seq_len = n * (n - 1) // 2
    x = torch.empty(seq_len, dtype=torch.long)
    for i, edge in enumerate(adj_flat):
        x[i] = EDGE if edge else NO_EDGE

    y = torch.tensor(coloring, dtype=torch.long) + COLOR_BASE
    return x, y


class GraphColoringTrainDataset(Dataset):
    """One graph per item; a canonical valid coloring is sampled per access."""

    def __init__(self, build: GraphColoringBuild, seed: int = 0):
        self.n = build.n
        self.graphs = build.train_graphs
        self.colorings = build.colorings
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        adj_flat = self.graphs[idx]
        sols = self.colorings[adj_flat]
        coloring = sols[self.rng.randrange(len(sols))]
        return graph_to_tensors(adj_flat, coloring, self.n)


class GraphColoringEvalSet:
    def __init__(self, build: GraphColoringBuild, split: str = "test"):
        self.n = build.n
        graphs = build.test_graphs if split == "test" else build.train_graphs
        self.examples = [(g, build.colorings[g]) for g in graphs]

    def __len__(self) -> int:
        return len(self.examples)

    def batches(self, batch_size: int, max_examples: int | None = None):
        examples = self.examples[:max_examples] if max_examples else self.examples
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            xs = []
            adjs = []
            colorings = []
            for adj_flat, sols in chunk:
                xs.append(torch.tensor([EDGE if e else NO_EDGE for e in adj_flat], dtype=torch.long))
                adjs.append(upper_to_adj(adj_flat, self.n))
                colorings.append(sols)
            yield torch.stack(xs, dim=0), torch.stack(adjs, dim=0), colorings


def steps_per_epoch(num_examples: int, global_batch_size: int) -> int:
    return math.ceil(num_examples / global_batch_size)


def color_logits_to_tokens(logits: torch.Tensor, n: int) -> torch.Tensor:
    """Restrict graph-color output positions to the three color tokens."""
    if logits.shape[1] != n:
        raise ValueError(f"expected graph-color logits with target length {n}, got {logits.shape[1]}")
    colors = logits[:, :, COLOR_BASE : COLOR_BASE + NUM_COLORS].argmax(-1)
    return colors + COLOR_BASE


def conflict_edges(pred_color_tokens: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    colors = pred_color_tokens - COLOR_BASE
    same = colors.unsqueeze(2) == colors.unsqueeze(1)
    conflict = same & adj.bool()
    n = adj.shape[-1]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=adj.device), diagonal=1)
    return (conflict & upper).float().sum(dim=(1, 2))


def graph_coverage(samples: torch.Tensor, colorings: List[Tuple[Coloring, ...]], n: int) -> float:
    """samples: (B, S, n) color tokens."""
    samples_l = (samples.detach().cpu() - COLOR_BASE).tolist()
    total = 0.0
    for preds, valid_sols in zip(samples_l, colorings):
        valid_set = set(valid_sols)
        found = {tuple(pred) for pred in preds if tuple(pred) in valid_set}
        total += len(found) / max(len(valid_set), 1)
    return total / max(len(colorings), 1)
