"""N-Queens 8x8 data utilities for GRAM Phase 3 validation.

Token vocabulary (size 3):
    0 = empty cell
    1 = queen
    2 = unknown / mask  (only appears in the input; never in the target)

There are exactly 92 valid 8-queens solutions. We enumerate them once
(~milliseconds), then at each training step sample a random solution and
a random mask rate. The model's task: given a partial board, predict the
full board.
"""
from typing import List, Tuple
import torch


def enumerate_solutions(n: int = 8) -> List[List[int]]:
    """All n-queens solutions, each as a list of column-per-row.
    Backtracking; for n=8 there are 92 solutions."""
    sols: List[List[int]] = []
    cols = [-1] * n

    def back(r: int) -> None:
        if r == n:
            sols.append(cols.copy())
            return
        for c in range(n):
            ok = True
            for r2 in range(r):
                c2 = cols[r2]
                if c2 == c or abs(c2 - c) == r - r2:
                    ok = False
                    break
            if ok:
                cols[r] = c
                back(r + 1)
    back(0)
    return sols


def solutions_to_grids(sols: List[List[int]], n: int = 8) -> torch.Tensor:
    """(N_sols, n*n) int64 0/1 grid tensor."""
    g = torch.zeros((len(sols), n, n), dtype=torch.long)
    for i, sol in enumerate(sols):
        for r, c in enumerate(sol):
            g[i, r, c] = 1
    return g.reshape(len(sols), n * n)


class NQueensBatcher:
    """Sample (input, target) pairs.
        target: 0/1 flattened grid (length n*n)
        input:  target with p_mask fraction of cells replaced by MASK=2
    p_mask is sampled per-example uniformly in p_mask_range to expose the
    model to varied difficulty levels."""

    MASK = 2

    def __init__(self, n: int = 8, p_mask_range: Tuple[float, float] = (0.3, 0.8)):
        self.n = n
        self.grids = solutions_to_grids(enumerate_solutions(n), n)
        self.p_mask_range = p_mask_range

    def num_solutions(self) -> int:
        return len(self.grids)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, len(self.grids), (batch_size,))
        y = self.grids[idx]                                              # (B, n*n)
        p_lo, p_hi = self.p_mask_range
        p = torch.empty(batch_size, 1).uniform_(p_lo, p_hi)
        mask = torch.rand(y.shape) < p
        x = torch.where(mask, torch.full_like(y, self.MASK), y)
        return x, y
