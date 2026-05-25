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

| file | what |
| --- | --- |
| `gram_model.py` | Model definition. Module is named `gram_model` (not `gram`) because the PyPI `gram` package shadows the import. |
| `data_nqueens.py` | Enumerates all 92 N-Queens 8×8 solutions, masks at p∈[0.3,0.8] |
| `gram_paper.ipynb` | Paper-scale training notebook for GPU. Just open and run top-to-bottom. |
| `train_nqueens.py` | Small-scale CLI training script (D=128) — for laptop sanity checks. |
| `train_smoke.py` | Phase-2 synthetic smoke test — verify ELBO machinery. |
| `smoke_test.py` | Phase-1 forward-pass shape test. |

## Paper-scale hyperparameters

These are baked into `gram_paper.ipynb` (`GRAMConfig` + the training cell):

| symbol | value | where |
| --- | --- | --- |
| D (d_model) | 512 | `GRAMConfig.d_model` |
| heads | 8 | `GRAMConfig.n_heads` |
| FFN hidden | 512 | `GRAMConfig.ffn_hidden` |
| f_L, f_H layers | 2 each | `GRAMConfig.n_layers` |
| K (inner loop) | 4 | `GRAMConfig.K` |
| T (transitions/step) | 3 | `GRAMConfig.T` |
| N_sup (sup. steps) | 16 | `GRAMConfig.N_sup` |
| total transitions | 48 | K is *per* transition |
| β (KL weight) | 0.07 | training cell |
| KL balance α | 0.8 | training cell |
| optimizer | AdamW(0.9, 0.95) | training cell |
| learning rate | 1e-4 (1k step warmup) | training cell |
| weight decay | 1.0 | training cell |
| EMA decay | 0.9999 | model+EMA cell |
| LPRM weight | 1.0 | `train_step` |
| halt weight | 0.5 | `train_step` |
| batch size | 128 | training cell (bump if VRAM allows) |
| total steps | 50,000 | training cell |

## Quickstart on a remote GPU

```bash
git clone https://github.com/<your-username>/gram-reproduction.git
cd gram-reproduction
pip install -r requirements.txt          # only torch + nbformat
jupyter notebook gram_paper.ipynb        # or upload to JupyterHub / Colab
```

Then run cells top-to-bottom. The training cell prints loss/recon/KL/halt/LPRM every 200 steps and a full eval (best-of-1, best-of-20, halt) every 2,000 steps for both raw and EMA weights. Checkpoints land in `gram_paper_step{N}.pt` every 5,000 steps.

VRAM budget: at B=128, D=512 the model fits in ~16 GB. On a 24 GB card you can push to B=256 for ~2× throughput.

Wall-clock estimate: ~4–6 hours on a single A100, ~8–12 hours on a 4090.

## Pass criteria

Per the paper Table 2 and our Phase 4 small-scale experiments:

- **EMA best-of-1 full_token > 0.99**
- **EMA best-of-20 full_board ≫ EMA best-of-1 full_board**  ← the LPRM payoff
- **Halt avg_steps < N_sup** when full_board is high — model has learned to stop early

## Critical implementation note

`RecursiveModule` (the f_L / f_H stack) **must** end in an RMSNorm. Without it, the residual stream summed across K inner iterations × N_sup·T transitions explodes (u std ~3.8 → mu/log_var heads see huge inputs → KL element-wise ~30,000 → loss NaN at step 1). The paper does not call this out explicitly. See `RecursiveModule` in `gram_model.py`.
