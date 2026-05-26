"""Paper-aligned N-Queens data and metrics for GRAM.

Paper task definition:
  * Tokens: 0=PAD, 1=empty, 2=queen.
  * Build all complete N-Queens solutions.
  * For N=8 remove k in {5, 6, 7} queens; for N=10 remove k in {7, 8, 9}.
  * The remaining partial queen configuration is the input, and a complete
    board is the target.
  * Train/test split is by unique input configuration, not by target pair.

The training dataset below keeps one item per input-target pair after the
unique-input split. Evaluation is done per unique input and uses all valid
completions to compute constraint accuracy and sample coverage.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


PAD = 0
EMPTY = 1
QUEEN = 2
VOCAB_SIZE = 3


Board = Tuple[int, ...]


def enumerate_solutions(n: int) -> List[Tuple[int, ...]]:
    """Return all N-Queens solutions as column-per-row tuples."""
    sols: List[Tuple[int, ...]] = []
    cols = [-1] * n

    def backtrack(row: int) -> None:
        if row == n:
            sols.append(tuple(cols))
            return
        for col in range(n):
            ok = True
            for prev_row in range(row):
                prev_col = cols[prev_row]
                if prev_col == col or abs(prev_col - col) == row - prev_row:
                    ok = False
                    break
            if ok:
                cols[row] = col
                backtrack(row + 1)
                cols[row] = -1

    backtrack(0)
    return sols


def solution_to_board(sol: Sequence[int]) -> Board:
    n = len(sol)
    board = [EMPTY] * (n * n)
    for row, col in enumerate(sol):
        board[row * n + col] = QUEEN
    return tuple(board)


def partial_from_solution(sol: Sequence[int], removed_rows: Iterable[int]) -> Board:
    n = len(sol)
    removed = set(removed_rows)
    board = [EMPTY] * (n * n)
    for row, col in enumerate(sol):
        if row not in removed:
            board[row * n + col] = QUEEN
    return tuple(board)


def default_removed_counts(n: int) -> Tuple[int, ...]:
    if n == 8:
        return (5, 6, 7)
    if n == 10:
        return (7, 8, 9)
    raise ValueError("The paper specifies N-Queens only for n=8 and n=10.")


@dataclass(frozen=True)
class NQueensBuild:
    n: int
    train_inputs: List[Board]
    test_inputs: List[Board]
    completions: Dict[Board, Tuple[Board, ...]]
    raw_pairs: int

    @property
    def train_pairs(self) -> int:
        return sum(len(self.completions[x]) for x in self.train_inputs)

    @property
    def test_pairs(self) -> int:
        return sum(len(self.completions[x]) for x in self.test_inputs)


def build_nqueens(
    n: int,
    *,
    seed: int = 0,
    train_fraction: float = 0.85,
    removed_counts: Sequence[int] | None = None,
) -> NQueensBuild:
    """Generate paper-style N-Queens inputs and split by unique input."""
    removed_counts = tuple(removed_counts or default_removed_counts(n))
    sols = enumerate_solutions(n)
    completions_mut: Dict[Board, set[Board]] = {}
    raw_pairs = 0

    for sol in sols:
        target = solution_to_board(sol)
        for k in removed_counts:
            for removed_rows in combinations(range(n), k):
                x = partial_from_solution(sol, removed_rows)
                completions_mut.setdefault(x, set()).add(target)
                raw_pairs += 1

    keys = list(completions_mut)
    random.Random(seed).shuffle(keys)
    cut = int(len(keys) * train_fraction)
    train_inputs = keys[:cut]
    test_inputs = keys[cut:]
    completions = {
        x: tuple(sorted(targets))
        for x, targets in completions_mut.items()
    }
    return NQueensBuild(
        n=n,
        train_inputs=train_inputs,
        test_inputs=test_inputs,
        completions=completions,
        raw_pairs=raw_pairs,
    )


def _board_tuple(value) -> Board:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple(int(v) for v in value)


def load_nqueens_cache(path: str | Path) -> NQueensBuild:
    """Load an explicitly materialized N-Queens dataset from prepare_data.py."""
    payload = torch.load(Path(path), map_location="cpu")
    if payload.get("task") != "nqueens":
        raise ValueError(f"Expected an nqueens cache file, got task={payload.get('task')!r}.")

    completions = {
        _board_tuple(x): tuple(_board_tuple(y) for y in targets)
        for x, targets in payload["completions"].items()
    }
    metadata = payload.get("metadata", {})
    return NQueensBuild(
        n=int(payload["n"]),
        train_inputs=[_board_tuple(x) for x in payload["train_inputs_unique"]],
        test_inputs=[_board_tuple(x) for x in payload["test_inputs_unique"]],
        completions=completions,
        raw_pairs=int(metadata.get("raw_generated_pairs", 0)),
    )


class NQueensTrainDataset(Dataset):
    """One item per input-target pair after splitting by unique input."""

    def __init__(self, build: NQueensBuild):
        self.n = build.n
        pairs: List[Tuple[Board, Board]] = []
        for x in build.train_inputs:
            for y in build.completions[x]:
                pairs.append((x, y))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        x, y = self.pairs[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class NQueensEvalSet:
    """Unique-input evaluation set with all valid completions per input."""

    def __init__(self, build: NQueensBuild, split: str = "test"):
        self.n = build.n
        inputs = build.test_inputs if split == "test" else build.train_inputs
        self.examples = [(x, build.completions[x]) for x in inputs]

    def __len__(self) -> int:
        return len(self.examples)

    def batches(self, batch_size: int, max_examples: int | None = None):
        examples = self.examples[:max_examples] if max_examples else self.examples
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            x = torch.tensor([e[0] for e in chunk], dtype=torch.long)
            completions = [e[1] for e in chunk]
            yield x, completions


def steps_per_epoch(num_examples: int, global_batch_size: int) -> int:
    return math.ceil(num_examples / global_batch_size)


def is_valid_completion(pred: Sequence[int], partial: Sequence[int], n: int) -> bool:
    if len(pred) != n * n:
        return False
    queens: List[Tuple[int, int]] = []
    for idx, token in enumerate(pred):
        if token not in (EMPTY, QUEEN):
            return False
        if partial[idx] == QUEEN and token != QUEEN:
            return False
        if token == QUEEN:
            queens.append((idx // n, idx % n))
    if len(queens) != n:
        return False
    rows = {r for r, _ in queens}
    cols = {c for _, c in queens}
    diag1 = {r - c for r, c in queens}
    diag2 = {r + c for r, c in queens}
    return len(rows) == len(cols) == len(diag1) == len(diag2) == n


def nqueens_accuracy(pred: torch.Tensor, x: torch.Tensor, n: int) -> float:
    correct = 0
    pred_l = pred.detach().cpu().tolist()
    x_l = x.detach().cpu().tolist()
    for p, partial in zip(pred_l, x_l):
        correct += int(is_valid_completion(p, partial, n))
    return correct / max(len(pred_l), 1)


def nqueens_diagnostics(pred: torch.Tensor, x: torch.Tensor, n: int) -> Dict[str, float]:
    """Full-prior N-Queens validity breakdown for debugging zero accuracy."""
    pred_l = pred.detach().cpu().tolist()
    x_l = x.detach().cpu().tolist()
    total = max(len(pred_l), 1)
    out = {
        "exact_n": 0,
        "rows_ok": 0,
        "cols_ok": 0,
        "diag_ok": 0,
        "valid_tokens": 0,
    }
    for p, partial in zip(pred_l, x_l):
        if len(p) != n * n:
            continue
        valid_tokens = True
        queens: List[Tuple[int, int]] = []
        for idx, token in enumerate(p):
            if token not in (EMPTY, QUEEN):
                valid_tokens = False
                break
            if partial[idx] == QUEEN and token != QUEEN:
                valid_tokens = False
                break
            if token == QUEEN:
                queens.append((idx // n, idx % n))
        if not valid_tokens:
            continue
        out["valid_tokens"] += 1
        rows = {r for r, _ in queens}
        cols = {c for _, c in queens}
        diag1 = {r - c for r, c in queens}
        diag2 = {r + c for r, c in queens}
        exact_n = len(queens) == n
        out["exact_n"] += int(exact_n)
        out["rows_ok"] += int(exact_n and len(rows) == n)
        out["cols_ok"] += int(exact_n and len(cols) == n)
        out["diag_ok"] += int(exact_n and len(diag1) == n and len(diag2) == n)
    return {k: v / total for k, v in out.items()}


def nqueens_coverage(samples: torch.Tensor, x: torch.Tensor, completions: List[Tuple[Board, ...]], n: int) -> float:
    """samples: (B, S, L) predicted token ids."""
    total = 0.0
    samples_l = samples.detach().cpu().tolist()
    x_l = x.detach().cpu().tolist()
    for preds, partial, valid_targets in zip(samples_l, x_l, completions):
        target_set = set(valid_targets)
        found = set()
        for pred in preds:
            pred_t = tuple(pred)
            if pred_t in target_set and is_valid_completion(pred_t, partial, n):
                found.add(pred_t)
        total += len(found) / max(len(target_set), 1)
    return total / max(len(completions), 1)
