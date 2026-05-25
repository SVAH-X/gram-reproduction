# GRAM Reproduction

Implementation of **Generative Recursive Reasoning** (Baek, Jo, Kim, Ren, Bengio, Ahn — arXiv:2605.19376, May 2026), reproduced from the paper alone (the official code is not yet public). Trained on N-Queens 8×8.

The model implements:

- Truncated surrogate ELBO over deep-supervision steps
- Amortized variational inference with shared `f_L` / `f_H` recursive modules between prior `p_θ(ε|u)` and posterior `q_φ(ε|u, e_y)`
- Dreamer-style balanced KL: `α·KL(sg(q)||p) + (1−α)·KL(q||sg(p))` with α=0.8
- LPRM (Latent Process Reward Model) value head for best-of-N selection
- ACT halt-only Q-head (paper Appendix A.1)
- RoPE positional encoding
- EMA of model weights (decay 0.9999)

## Files

### Model + data (always loaded)
| file | what |
| --- | --- |
| `gram_model.py` | Model definition. Module is named `gram_model` (not `gram`) because the PyPI `gram` package shadows the import. |
| `data_nqueens.py` | Enumerates all 92 N-Queens 8×8 solutions, masks at p∈[0.3,0.8] |

### Paper-scale training (use either, they are equivalent)
| file | what |
| --- | --- |
| `gram_paper.ipynb` | Paper-scale Jupyter notebook. D=512, K=4, T=3, N_sup=16, batch=768, lr=1e-4, wd=1.0, EMA=0.9999, 30k steps. |
| `train_paper.py`   | Same config as above, pure Python script (`python train_paper.py`). All hparams overridable via CLI flags. |

### Laptop validation (small config — NOT paper-scale)
| file | what | typical runtime |
| --- | --- | --- |
| `smoke_test.py`     | Phase-1 forward-pass shape test. D=128, no training. | seconds |
| `train_smoke.py`    | Phase-2 synthetic ELBO check (y = reverse(x)). D=128, β=0.3, 600 steps. | ~1 min |
| `train_nqueens.py`  | Phase-3a small N-Queens validation. D=128, β=0.1, 1000 steps. **Will not reach paper numbers** — it's intentionally tiny so it finishes on a MacBook. | ~90 s on MPS |

The laptop scripts use **lr=1e-3, wd=0.01, batch=32, no EMA, no RoPE, no halt** because at D=128 paper hparams over-regularize the small model. They are scaffolding to verify the math; they are not the experiment.

## Hyperparameter alignment with paper

| symbol | paper | this repo | confidence |
| --- | --- | --- | --- |
| D (d_model) | 512 | 512 | ✓ paper |
| heads | 8 | 8 | ✓ paper |
| FFN hidden | 512 | 512 | ✓ paper |
| f_L, f_H layers | 2 each | 2 each | ✓ paper |
| K (inner loop) — N-Queens | 4 | 4 | ✓ paper |
| T (transitions/step) | 3 | 3 | ✓ paper |
| N_sup (sup. steps) | 16 | 16 | ✓ paper |
| β (KL weight) — N-Queens 8×8 | 0.07 | 0.07 | ✓ paper |
| KL balance α | 0.8 (Dreamer) | 0.8 | ✓ paper |
| AdamW lr | 1e-4 | 1e-4 | ✓ paper |
| AdamW wd | 1.0 | 1.0 | ✓ paper |
| grad clip | 1.0 | 1.0 | ✓ paper |
| EMA decay | 0.9999 | 0.9999 | ✓ paper |
| **batch size (global)** | **768** (8×4090) | **768** (B=64 × accum=12) | ✓ matched via grad accum |
| Halt threshold σ(q^halt)>0.5 | 0.5 | 0.5 | ✓ paper |
| RoPE θ | not specified | 10000 | ⚠️ standard default |
| AdamW betas | not specified in our notes | (0.9, 0.95) | ⚠️ guess (modern-LLM default) |
| Warmup schedule | not specified in our notes | 1000-step linear | ⚠️ guess |
| halt loss weight | not specified in our notes | 0.5 | ⚠️ guess |
| LPRM loss weight | not specified in our notes | 1.0 | ⚠️ guess |
| Training length | "3000 epochs" (paper) | 30,000 optimizer steps ≈ 23M effective samples | ⚠️ epoch→step conversion ambiguous; tune if not converged |

The ⚠️ rows are the parameters most likely to need tuning. If the paper code drops, prefer those values over my guesses.

## Quickstart on a remote GPU

```bash
git clone https://github.com/SVAH-X/gram-reproduction.git
cd gram-reproduction
pip install -r requirements.txt          # only torch + nbformat

# Option A — Jupyter notebook
jupyter notebook gram_paper.ipynb        # or upload to JupyterHub / Colab

# Option B — pure Python (preferred for headless servers / nohup / tmux)
python train_paper.py                                    # 30k steps, batch 64*12=768
python train_paper.py --b-per-step 128 --accum 6         # if you have 24+ GB VRAM
python train_paper.py --steps 50000 --out-prefix run2    # longer run, separate output
nohup python train_paper.py > paper_train.out 2>&1 &     # detached run
```

Both paths print loss/recon/KL/halt/LPRM every 200 steps and a full eval (best-of-1, best-of-20, halt) every 2,000 steps for both raw and EMA weights. Checkpoints land in `gram_paper_step{N}.pt` every 5,000 steps.

VRAM budget: at `B_per_step=64`, D=512 the model fits in ~10–12 GB. To match paper batch=768 we use gradient accumulation `accum_steps=12` (so `64 × 12 = 768`). If your card has more VRAM, raise `B_per_step` and lower `accum_steps` proportionally — the effective batch must stay 768.

Wall-clock estimate: paper reports ~1 h on 8×4090 for N-Queens 8×8. With grad accum on a single GPU expect roughly **8× longer per effective batch** plus arithmetic for the smaller per-step batch — budget ~12–24 h on a single A100/4090, longer on weaker cards.

## Pass criteria

Paper-reported numbers on N-Queens 8×8: **99.7% acc / 90.3% cov**. Reproduction targets:

- **EMA best-of-1 full_token > 0.99** (= paper "acc")
- **EMA best-of-20 full_board > 0.85** (≈ paper "cov" = best-of-N coverage)
- **Halt avg_steps < N_sup** when full_board is high — model has learned to stop early
- corr(score, reward) > 0.7 — LPRM is informative
- distinct preds / 20 samples > 5 — prior has not collapsed

## Critical implementation note

`RecursiveModule` (the f_L / f_H stack) **must** end in an RMSNorm. Without it, the residual stream summed across K inner iterations × N_sup·T transitions explodes (u std ~3.8 → mu/log_var heads see huge inputs → KL element-wise ~30,000 → loss NaN at step 1). The paper does not call this out explicitly. See `RecursiveModule` in `gram_model.py`.
