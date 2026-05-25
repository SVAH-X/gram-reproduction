"""Graph 3-coloring data utilities for GRAM.

Sequence layout:
    [N*N adjacency entries] + [N node colors]   = N*N + N tokens

Vocab (size 6):
    0 = no edge       (adjacency positions only)
    1 = edge          (adjacency positions only)
    2 = color 1
    3 = color 2
    4 = color 3
    5 = mask          (input only — replaces an unknown color)

Graph distribution: Erdős-Rényi G(N, p_edge), undirected, no self-loops.
We pre-cache only graphs that admit a valid 3-coloring (found by
backtracking) so supervision targets always exist. Hard graphs (chromatic
number > 3) are skipped at cache time.

Mask: per-node independently, p_mask ∈ p_mask_range, sampled per example.
The model is asked to recover the FULL coloring (not just masked nodes),
so loss is computed only on the N color positions — adjacency is input.
"""
from typing import List, Optional, Tuple
import torch

NO_EDGE = 0
EDGE    = 1
COLOR_BASE  = 2     # colors live at tokens 2, 3, 4
NUM_COLORS  = 3
MASK    = 5
VOCAB_SIZE = 6


def random_graph(n: int, p_edge: float, generator=None) -> torch.Tensor:
    """Symmetric n x n long adjacency tensor, no self-loops."""
    upper = (torch.rand(n, n, generator=generator) < p_edge).triu(diagonal=1)
    adj = upper | upper.T
    return adj.long()


def find_3coloring(adj: torch.Tensor) -> Optional[torch.Tensor]:
    """Backtracking 3-coloring. Returns (n,) long in {0,1,2} or None."""
    n = adj.shape[0]
    adj_list = [[j for j in range(n) if adj[i, j].item()] for i in range(n)]
    colors = [-1] * n

    def back(i: int) -> bool:
        if i == n:
            return True
        for c in range(NUM_COLORS):
            if all(colors[j] != c for j in adj_list[i]):
                colors[i] = c
                if back(i + 1):
                    return True
                colors[i] = -1
        return False

    return torch.tensor(colors, dtype=torch.long) if back(0) else None


class GraphColoringBatcher:
    """Pre-caches a fixed set of 3-colorable Erdős-Rényi graphs and at
    sample-time draws one + a fresh random mask. Each call returns
    (x, y, y_mask, adj) — the bool y_mask marks the N color positions
    (last N tokens of length N²+N) so train_step can ignore adjacency."""

    MASK = MASK
    NUM_COLORS = NUM_COLORS
    COLOR_BASE = COLOR_BASE

    def __init__(self, n: int = 8, p_edge: float = 0.4,
                 p_mask_range: Tuple[float, float] = (0.3, 0.8),
                 cache_size: int = 1024,
                 max_tries_factor: int = 50,
                 seed: int = 0):
        self.n = n
        self.p_edge = p_edge
        self.p_mask_range = p_mask_range

        gen = torch.Generator().manual_seed(seed)
        graphs:    List[torch.Tensor] = []
        colorings: List[torch.Tensor] = []
        tries = 0
        max_tries = cache_size * max_tries_factor
        while len(graphs) < cache_size and tries < max_tries:
            tries += 1
            adj = random_graph(n, p_edge, generator=gen)
            col = find_3coloring(adj)
            if col is not None:
                graphs.append(adj)
                colorings.append(col)
        if len(graphs) < cache_size:
            raise RuntimeError(
                f"only {len(graphs)} 3-colorable graphs after {tries} tries "
                f"at n={n}, p_edge={p_edge}; lower p_edge or raise max_tries.")

        self.graphs    = torch.stack(graphs, dim=0)            # (C, n, n)
        self.colorings = torch.stack(colorings, dim=0)         # (C, n)
        self.cache_size = cache_size
        # pre-flatten adjacency once
        self.adj_flat = self.graphs.reshape(cache_size, n * n)

    def num_graphs(self) -> int:
        return self.cache_size

    def sample(self, batch_size: int):
        n = self.n
        seq_len = n * n + n
        idx = torch.randint(0, self.cache_size, (batch_size,))

        adj_flat  = self.adj_flat[idx]                         # (B, n*n)
        col_token = self.colorings[idx] + COLOR_BASE           # (B, n) -> tokens 2..4

        x = torch.empty((batch_size, seq_len), dtype=torch.long)
        y = torch.empty((batch_size, seq_len), dtype=torch.long)
        x[:, :n * n] = adj_flat
        y[:, :n * n] = adj_flat                                # adjacency identical in y
        y[:, n * n:] = col_token                               # ground-truth colors

        # mask color positions in input
        p_lo, p_hi = self.p_mask_range
        p = torch.empty(batch_size, 1).uniform_(p_lo, p_hi)
        m = torch.rand(batch_size, n) < p
        x[:, n * n:] = torch.where(m, torch.full_like(col_token, MASK), col_token)

        # y_mask: True only at color positions
        y_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        y_mask[:, n * n:] = True

        return x, y, y_mask, self.graphs[idx]                  # also return adj for eval


def conflict_edges(pred_colors: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """Number of edges whose endpoints share a color, per example.
    pred_colors: (B, n) long in 0..NUM_COLORS-1 (already shifted off COLOR_BASE)
    adj:         (B, n, n) long
    Returns:     (B,) float — count of conflicting edges (each edge once).
    """
    same = pred_colors.unsqueeze(2) == pred_colors.unsqueeze(1)        # (B, n, n)
    conflict = same & adj.bool()
    n = adj.shape[-1]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=adj.device),
                       diagonal=1)
    return (conflict & upper).float().sum(dim=(1, 2))                  # (B,)
