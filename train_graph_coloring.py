"""GRAM Graph 3-Coloring training (paper-scale).

Same model + optimizer + EMA + LPRM + halt machinery as train_paper.py
(N-Queens). The differences are all on the data side:
    - vocab_size = 6 (no-edge / edge / 3 colors / mask)
    - seq_len    = N*N + N  (adjacency concatenated to coloring)
    - y_mask     = True only at the N color positions
    - eval metric = conflict edges (paper: 2.7 for N=8, 3.3 for N=10)

Defaults for β: paper Table — 0.5 for N=8, 0.45 for N=10.

Usage:
    python train_graph_coloring.py --n 8
    python train_graph_coloring.py --n 10
    python train_graph_coloring.py --n 8 --b-per-step 128 --accum 6
"""
import argparse
import time
from contextlib import contextmanager

import torch
from torch.optim import AdamW

from gram_model import GRAM, GRAMConfig, EMA
from data_graph_coloring import (
    GraphColoringBatcher, conflict_edges,
    VOCAB_SIZE, COLOR_BASE, NUM_COLORS, MASK,
)


@contextmanager
def _nullcontext():
    yield


@torch.no_grad()
def _color_accuracy(pred_full, y_full, n):
    """Token accuracy on color positions only. pred_full / y_full: (B, n²+n)."""
    pred_c = pred_full[:, n * n:]
    y_c    = y_full[:, n * n:]
    correct = (pred_c == y_c)
    full_tok   = correct.float().mean().item()
    full_board = correct.all(dim=1).float().mean().item()
    return full_tok, full_board


@torch.no_grad()
def evaluate(model, batcher, batch_size, device, N_best=20,
             use_ema=False, ema=None):
    model.eval()
    n = batcher.n
    x, y, _, adj = batcher.sample(batch_size)
    x, y, adj = x.to(device), y.to(device), adj.to(device)

    ctx = ema.swap_in(model) if use_ema else _nullcontext()
    with ctx:
        # best-of-1
        logits1 = model(x)
        pred1 = logits1.argmax(-1)
        ft1, fb1 = _color_accuracy(pred1, y, n)
        col1 = (pred1[:, n * n:] - COLOR_BASE).clamp(0, NUM_COLORS - 1)
        ce1 = conflict_edges(col1, adj).mean().item()

        # best-of-N
        logitsN, scores = model.forward_best_of_n(x, N=N_best)
        predN = logitsN.argmax(-1)
        ftN, fbN = _color_accuracy(predN, y, n)
        colN = (predN[:, n * n:] - COLOR_BASE).clamp(0, NUM_COLORS - 1)
        ceN = conflict_edges(colN, adj).mean().item()

        if model.cfg.use_halt:
            logitsH, n_steps_h = model.forward_with_halt(x)
            predH = logitsH.argmax(-1)
            ftH, fbH = _color_accuracy(predH, y, n)
            colH = (predH[:, n * n:] - COLOR_BASE).clamp(0, NUM_COLORS - 1)
            ceH = conflict_edges(colH, adj).mean().item()
            avg_steps = n_steps_h.float().mean().item()
        else:
            ftH = fbH = ceH = avg_steps = float("nan")

    return dict(
        n1_full_tok=ft1, n1_full_color=fb1, n1_conflicts=ce1,
        nN_full_tok=ftN, nN_full_color=fbN, nN_conflicts=ceN,
        halt_full_tok=ftH, halt_full_color=fbH, halt_conflicts=ceH,
        halt_avg_steps=avg_steps, score_mean=scores.mean().item(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",          type=int, default=8,
                    help="number of nodes (paper: 8 or 10)")
    ap.add_argument("--p-edge",     type=float, default=0.4,
                    help="Erdős-Rényi edge probability")
    ap.add_argument("--cache-size", type=int, default=4096,
                    help="number of pre-cached 3-colorable graphs")
    ap.add_argument("--b-per-step", type=int, default=64)
    ap.add_argument("--accum",      type=int, default=12)
    ap.add_argument("--steps",      type=int, default=30_000)
    ap.add_argument("--warmup",     type=int, default=1_000)
    ap.add_argument("--lr",         type=float, default=1e-4)         # ✓ paper
    ap.add_argument("--wd",         type=float, default=1.0)          # ✓ paper
    ap.add_argument("--beta",       type=float, default=None,
                    help="KL weight; defaults: 0.5 (n=8), 0.45 (n=10)")
    ap.add_argument("--kl-balance", type=float, default=0.8)          # ✓ paper
    ap.add_argument("--ema-decay",  type=float, default=0.9999)       # ✓ paper
    ap.add_argument("--halt-weight",type=float, default=0.5)          # ⚠ guess
    ap.add_argument("--lprm-weight",type=float, default=1.0)          # ⚠ guess
    ap.add_argument("--log-every",  type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=2_000)
    ap.add_argument("--ckpt-every", type=int, default=5_000)
    ap.add_argument("--out-prefix", type=str, default=None)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--device",     type=str, default=None)
    args = ap.parse_args()

    PAPER_BETA = {8: 0.5, 10: 0.45}
    if args.beta is None:
        if args.n not in PAPER_BETA:
            raise ValueError(f"--n={args.n}: paper specifies n=8 (β=0.5) and "
                             f"n=10 (β=0.45). Pass --beta explicitly otherwise.")
        args.beta = PAPER_BETA[args.n]
    if args.out_prefix is None:
        args.out_prefix = f"gram_graphcolor_n{args.n}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu  : {torch.cuda.get_device_name(0)}")
    torch.manual_seed(args.seed)

    n = args.n
    seq_len = n * n + n

    cfg = GRAMConfig(
        vocab_size=VOCAB_SIZE, seq_len=seq_len,
        d_model=512, n_heads=8, ffn_hidden=512, n_layers=2,
        K=4, T=3, N_sup=16,
        use_attn=True, use_rope=True, use_halt=True,
    )
    model = GRAM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"task             : Graph 3-Coloring N={n} (seq_len {seq_len}, β {args.beta})")
    print(f"params           : {n_params/1e6:.2f}M")
    print(f"transitions/step : N_sup * T = {cfg.N_sup * cfg.T}")

    ema = EMA(model, decay=args.ema_decay)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                betas=(0.9, 0.95))

    print(f"caching {args.cache_size} 3-colorable graphs (n={n}, p_edge={args.p_edge}) ...")
    batcher = GraphColoringBatcher(n=n, p_edge=args.p_edge,
                                   cache_size=args.cache_size, seed=args.seed)
    print(f"cached {batcher.num_graphs()} graphs")

    target_batch = args.b_per_step * args.accum
    print(f"effective batch  : {args.b_per_step} * {args.accum} = {target_batch} "
          f"(paper: 768){'  ✓' if target_batch == 768 else '  ⚠ paper used 768'}")

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        return 1.0

    # -------------------------------------------------------------- training
    log = []
    t0 = time.time()
    log_path = f"{args.out_prefix}.log"
    logf = open(log_path, "w")

    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_at(step)

        opt.zero_grad(set_to_none=True)
        info_accum = {"loss": 0.0, "recon": 0.0, "kl": 0.0,
                      "lprm": 0.0, "halt": 0.0, "r": 0.0, "acc": 0.0}
        for _ in range(args.accum):
            x, y, y_mask, _ = batcher.sample(args.b_per_step)
            x, y, y_mask = x.to(device), y.to(device), y_mask.to(device)
            model.train()
            loss, info = model.train_step(
                x, y,
                beta=args.beta, kl_balance=args.kl_balance,
                lprm_weight=args.lprm_weight, halt_weight=args.halt_weight,
                y_mask=y_mask,
            )
            (loss / args.accum).backward()
            for k in info_accum:
                info_accum[k] += info[k] / args.accum

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

        if step == 1 or step % args.log_every == 0:
            msg = (f"step {step:6d} | loss {info_accum['loss']:.4f} | "
                   f"recon {info_accum['recon']:.4f} | kl {info_accum['kl']:.4f} | "
                   f"halt {info_accum['halt']:.4f} | lprm {info_accum['lprm']:.4f} | "
                   f"r {info_accum['r']:.3f} | acc {info_accum['acc']:.3f} | "
                   f"gn {gn.item():.2f} | t {time.time()-t0:.1f}s")
            print(msg); logf.write(msg + "\n"); logf.flush()
            log.append(dict(step=step, **info_accum, gnorm=gn.item(), t=time.time()-t0))

        if step % args.eval_every == 0:
            ev_raw = evaluate(model, batcher, 128, device, N_best=20, use_ema=False)
            ev_ema = evaluate(model, batcher, 128, device, N_best=20, use_ema=True, ema=ema)
            for tag, ev in (("raw", ev_raw), ("EMA", ev_ema)):
                lines = [
                    f"  >> {tag:3s}  N=1   tok {ev['n1_full_tok']:.3f} "
                    f"full {ev['n1_full_color']:.3f} conflicts {ev['n1_conflicts']:.2f}",
                    f"  >> {tag:3s}  N=20  tok {ev['nN_full_tok']:.3f} "
                    f"full {ev['nN_full_color']:.3f} conflicts {ev['nN_conflicts']:.2f}",
                    f"  >> {tag:3s}  halt  tok {ev['halt_full_tok']:.3f} "
                    f"full {ev['halt_full_color']:.3f} conflicts {ev['halt_conflicts']:.2f} "
                    f"avg_steps {ev['halt_avg_steps']:.2f}",
                ]
                for line in lines:
                    print(line); logf.write(line + "\n")
            logf.flush()

        if step % args.ckpt_every == 0:
            ckpt_path = f"{args.out_prefix}_step{step}.pt"
            torch.save({
                "cfg": cfg.__dict__,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "args": vars(args),
            }, ckpt_path)
            print(f"  >> ckpt saved: {ckpt_path}")

    final_path = f"{args.out_prefix}_final.pt"
    torch.save({
        "cfg": cfg.__dict__,
        "model": model.state_dict(),
        "ema":   ema.state_dict(),
        "log":   log,
        "args":  vars(args),
    }, final_path)
    print(f"done — saved {final_path}")
    logf.close()


if __name__ == "__main__":
    main()
