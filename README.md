# GRAM Reproduction

From-paper reproduction of **Generative Recursive Reasoning** (Baek et al., arXiv:2605.19376) on the two multi-solution constraint-satisfaction tasks (N-Queens and 3-Graph Coloring). The model follows paper Appendix B (K=4, T=3, N_sup=16, RoPE, SwiGLU, RMSNorm, 16 prepended puzzle/register tokens) with a slightly larger 11.55 M implementation due to explicit posterior conditioning, and the dataset structure matches Appendix C.2 exactly (all four cache files pass 84 paper-strict checks; every one of the 618 020 cached solutions is a real constraint-satisfying answer).

## Setup

```bash
# 1. CUDA-enabled PyTorch
pip install -r requirements.txt

# 2. Build the synthetic data once (writes ~250 MB into data_cache/)
python prepare_data.py --out-dir data_cache --seed 0 --p-edge 0.4

# 3. Optional sanity checks
python verify_datasets.py
python verify_solutions_exhaustive.py
python analyze_data.py --data-dir data_cache
```

## Training on the 4×A100 server

`train_paper.py` and `train_graph_coloring.py` are single-process training scripts. They use bf16 autocast on CUDA and gradient accumulation to reach the paper's global batch size of 768 on **one** GPU. To use all four A100s, launch four runs in parallel on the four task / size combinations:

```bash
# pin each task to its own GPU
CUDA_VISIBLE_DEVICES=0 nohup python -u train_paper.py            --n 8  --out-prefix gram_nqueens_n8     > nq8.out  2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u train_paper.py            --n 10 --out-prefix gram_nqueens_n10    > nq10.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u train_graph_coloring.py   --n 8  --out-prefix gram_graphcolor_n8  > gc8.out  2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u train_graph_coloring.py   --n 10 --out-prefix gram_graphcolor_n10 > gc10.out 2>&1 &
```

A100 has ~3× the FLOPs of the RTX 4090 the paper used, so each task should fit comfortably on a single GPU at `--global-batch 768 --b-per-step 64` (defaults). If you OOM, drop `--b-per-step` to 32 or 16 — the script keeps the effective batch at 768 via gradient accumulation. Checkpoints land in the working directory as `<prefix>_step*.pt` and `<prefix>_final.pt`; logs in `<prefix>.log`.

After training, run paper-style evaluation against the full test set:

```bash
python eval_paper.py --task nqueens   --n 8  --checkpoint gram_nqueens_n8_final.pt    --use-ema --batch-size 64
python eval_paper.py --task nqueens   --n 10 --checkpoint gram_nqueens_n10_final.pt   --use-ema --batch-size 64
python eval_paper.py --task graphcolor --n 8  --checkpoint gram_graphcolor_n8_final.pt  --use-ema --batch-size 64
python eval_paper.py --task graphcolor --n 10 --checkpoint gram_graphcolor_n10_final.pt --use-ema --batch-size 64
```

Outputs go to `data_cache/analysis/eval_*.json` and `eval_*.buckets.csv`.

## How epochs / steps are designed (paper-strict)

The paper reports "epochs" in Appendix B.2 Table 7. Throughout the GRAM/HRM/TRM line, **one epoch == one trajectory batch** — a global batch of 768 examples that is run through the *entire* recursive trajectory of `T_total = T × N_sup` transitions. Within that single trajectory the optimizer takes `N_sup = 16` gradient steps, one per supervision segment (Appendix A.3 / Eq. 14).

This repo follows that convention literally:

```
total optimizer steps = (paper epochs) × N_sup
```

| task          | paper epochs | N_sup | total optimizer steps |
|---------------|-------------:|------:|----------------------:|
| N-Queens 8×8  |         3000 |    16 |                48 000 |
| N-Queens 10×10|         1000 |    16 |                16 000 |
| Graph Color 8 |         5000 |    16 |                80 000 |
| Graph Color 10|         5000 |    16 |                80 000 |

The training scripts print `paper trajectory batches`, `segment updates/trajectory`, and `planned training steps` so you can read this directly off the run header. Each segment update is one `opt.step()` call on a global batch of 768 examples — exactly what the paper trains on. Set `--max-steps` to cap early during smoke runs.

Paper-stated knobs and reproduction stabilizers already wired in:

- **AdamW**: lr=1e-4, weight_decay=1.0, gradient clipping at 1.0. By default decay is applied only to matrix weights; pass `--decay-all` to reproduce indiscriminate AdamW decay.
- **Global batch = 768**, microbatch defaults to 64 (gradient accumulation x12).
- **EMA decay = 0.9999** (applied every optimizer step; eval uses EMA weights when `--use-ema` is set).
- **KL balance β_bal = 0.8** (Appendix B.2).
- **β (KL weight) per task**: 0.07 / 0.045 / 0.5 / 0.45 for NQ-8 / NQ-10 / GC-8 / GC-10 (auto-selected, override with `--beta`).
- **N-Queens CE weighting**: queen token weight defaults to `N-1` via `--queen-loss-weight auto`, preventing the copy-input/empty-majority local optimum.
- **No LR warmup** (paper does not specify one, and `lr=1e-4` is stated as the constant rate).
- **bf16 autocast** on A100; KL is computed in fp32 to keep gradient signal stable.

## Live monitoring

Both training scripts log every 5 optimizer steps by default:

```
traj  123/3000 seg  7/16 step    1979/48000 | loss 0.4187 rp 0.1284 rq 0.0210 kl 1.43 kl_true 1.43 mp 1.21 mq 1.67 halt 0.04 lprm 0.001 r 0.973 acc_p 0.984 acc_q 0.998 gn 0.61 t 412.3s
  >> raw test n=512 acc 0.9821 coverage@20 0.7864 avg_q 7.96/8 keep 0.997
  >> EMA test n=512 acc 0.9893 coverage@20 0.8194 avg_q 7.99/8 keep 0.999
```

Eval rolls every 1000 steps by default (`--eval-every`). To tail a run:

```bash
tail -f gram_nqueens_n8.log
```

To plot live or after-the-fact (publication-quality, via SciencePlots):

```bash
python plot_results.py curves   --log gram_nqueens_n8.log
python plot_results.py hist     --data-dir data_cache
python plot_results.py coverage --buckets data_cache/analysis/eval_nqueens_n8.buckets.csv
```

Figures are written to `data_cache/analysis/figs/`.

## What's been verified before launch

- Model architecture re-derived from paper Appendix B; param count **11.55 M** (paper says ~10 M; we are slightly over because we replaced the paper's underspecified additive posterior conditioning `u + e_y` with `posterior_net(concat([u, e_y]))` after the additive form was empirically broken — see CHANGELOG).
- 16 prepended puzzle/register tokens added; halt/value heads read position 0 (the first register token).
- Both training scripts smoke-pass end-to-end (`--max-steps 2`).
- `verify_datasets.py` — 84/84 paper-strict checks across all 4 cache files.
- `verify_solutions_exhaustive.py` — every one of 618 020 cached solutions across the four datasets is a true constraint-satisfying answer (rows/cols/diagonals distinct for N-Queens; no monochrome edges and canonical color labels for Graph Coloring).
- Stochastic guidance verified active (two model(x) calls on the same input give mean-abs-diff 0.44 ≠ 0).

## File index

```
gram_model.py                 GRAM module (encoder, recursive core, decoder, stochastic guidance, EMA, ACT halt head, LPRM value head)
data_nqueens.py               N-Queens generator, dataset, metric, validity checker
data_graph_coloring.py        Graph Coloring generator, dataset, metric, conflict/coverage helpers
prepare_data.py               Materialize all four cache files (.pt + metadata.json + summary.txt)
analyze_data.py               Solution-count histograms (Figure 10/12 inputs)
train_paper.py                Train GRAM on N-Queens (--n 8 or 10)
train_graph_coloring.py       Train GRAM on Graph Coloring (--n 8 or 10)
eval_paper.py                 Paper-style evaluation: acc@1, coverage@N for N in {1,5,10,20}, bucketed by # solutions
verify_datasets.py            Paper-strict dataset checks (84 assertions)
verify_solutions_exhaustive.py  Brute-force solution validity over every cached answer (~600K solutions)
plot_results.py               Training curves + figures via SciencePlots
smoke_test.py                 Tiny model+forward shape test
data_cache/                   Materialized .pt files + summary.txt + analysis/
```

## Known remaining unknowns

The paper does not state the Graph Coloring edge probability, AdamW betas, or the auxiliary loss weights for the halt/LPRM heads. Defaults are exposed as CLI flags (`--p-edge`, `--halt-weight`, `--lprm-weight`). If the authors release an official implementation, swap these in.
