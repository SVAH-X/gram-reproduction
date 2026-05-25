"""GRAM Phase 3b — paper-scale N-Queens 8x8 training (Python script form).

This is the .py equivalent of `gram_paper.ipynb`. Same config, same loss,
same eval, same logging. Use whichever you prefer — both produce the same
checkpoints and the same metrics.

Usage on a GPU server:
    pip install -r requirements.txt
    python train_paper.py                                 # default 30k steps
    python train_paper.py --steps 50000 --b-per-step 128  # 4090/A100 with more VRAM

All hyperparameters (architectural + optimizer) match the GRAM paper for
N-Queens 8x8. See README.md for the paper-vs-this-repo confidence table.
"""
import argparse
import os
import time
from contextlib import contextmanager

import torch
from torch.optim import AdamW

from gram_model import GRAM, GRAMConfig, EMA
from data_nqueens import NQueensBatcher


# ----------------------------------------------------------------- helpers

@contextmanager
def _nullcontext():
    yield


@torch.no_grad()
def _accuracy(pred, y, x, mask_token):
    mask_pos = (x == mask_token)
    correct  = (pred == y)
    full_tok   = correct.float().mean().item()
    mask_tok   = ((correct & mask_pos).float().sum().item()
                  / max(mask_pos.sum().item(), 1))
    full_board = correct.all(dim=1).float().mean().item()
    return full_tok, mask_tok, full_board


@torch.no_grad()
def evaluate(model, batcher, batch_size, device, N_best=20, use_ema=False, ema=None):
    model.eval()
    x, y = batcher.sample(batch_size)
    x, y = x.to(device), y.to(device)

    ctx = ema.swap_in(model) if use_ema else _nullcontext()
    with ctx:
        logits1 = model(x)
        ft1, mt1, fb1 = _accuracy(logits1.argmax(-1), y, x, batcher.MASK)
        logitsN, scores = model.forward_best_of_n(x, N=N_best)
        ftN, mtN, fbN = _accuracy(logitsN.argmax(-1), y, x, batcher.MASK)
        if model.cfg.use_halt:
            logitsH, n_steps_h = model.forward_with_halt(x)
            ftH, mtH, fbH = _accuracy(logitsH.argmax(-1), y, x, batcher.MASK)
            avg_steps = n_steps_h.float().mean().item()
        else:
            ftH = mtH = fbH = avg_steps = float("nan")
    return dict(
        n1_full_token=ft1, n1_mask_token=mt1, n1_full_board=fb1,
        nN_full_token=ftN, nN_mask_token=mtN, nN_full_board=fbN,
        halt_full_token=ftH, halt_full_board=fbH,
        halt_avg_steps=avg_steps, score_mean=scores.mean().item(),
    )


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    # Paper hparams — defaults match arXiv:2605.19376 N-Queens 8x8.
    ap.add_argument("--b-per-step", type=int, default=64,
                    help="per-forward batch (raise to fit your VRAM)")
    ap.add_argument("--accum",      type=int, default=12,
                    help="grad accumulation; effective batch = b-per-step * accum (paper: 768)")
    ap.add_argument("--steps",      type=int, default=30_000,
                    help="total optimizer steps")
    ap.add_argument("--warmup",     type=int, default=1_000)
    ap.add_argument("--lr",         type=float, default=1e-4)        # ✓ paper
    ap.add_argument("--wd",         type=float, default=1.0)         # ✓ paper
    ap.add_argument("--beta",       type=float, default=0.07)        # ✓ paper N-Queens 8x8
    ap.add_argument("--kl-balance", type=float, default=0.8)         # ✓ paper (Dreamer)
    ap.add_argument("--ema-decay",  type=float, default=0.9999)      # ✓ paper
    ap.add_argument("--halt-weight",type=float, default=0.5)         # ⚠ guess
    ap.add_argument("--lprm-weight",type=float, default=1.0)         # ⚠ guess
    ap.add_argument("--log-every",  type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=2_000)
    ap.add_argument("--ckpt-every", type=int, default=5_000)
    ap.add_argument("--out-prefix", type=str, default="gram_paper")
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--device",     type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu  : {torch.cuda.get_device_name(0)}")
        print(f"vram : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    torch.manual_seed(args.seed)

    cfg = GRAMConfig(                            # ✓ all values match paper
        vocab_size=3, seq_len=64,
        d_model=512, n_heads=8, ffn_hidden=512, n_layers=2,
        K=4, T=3, N_sup=16,
        use_attn=True, use_rope=True, use_halt=True,
    )
    model = GRAM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params           : {n_params/1e6:.2f}M")
    print(f"transitions/step : N_sup * T = {cfg.N_sup * cfg.T}")

    ema = EMA(model, decay=args.ema_decay)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                betas=(0.9, 0.95))               # ⚠ betas guessed; paper unspecified

    batcher = NQueensBatcher(n=8)
    print(f"n-queens solutions: {batcher.num_solutions()}")

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
            x, y = batcher.sample(args.b_per_step)
            x, y = x.to(device), y.to(device)
            model.train()
            loss, info = model.train_step(
                x, y,
                beta=args.beta, kl_balance=args.kl_balance,
                lprm_weight=args.lprm_weight, halt_weight=args.halt_weight,
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
            log.append(dict(step=step, **info_accum,
                            gnorm=gn.item(), t=time.time()-t0))

        if step % args.eval_every == 0:
            ev_raw = evaluate(model, batcher, 128, device, N_best=20, use_ema=False)
            ev_ema = evaluate(model, batcher, 128, device, N_best=20, use_ema=True, ema=ema)
            for tag, ev in (("raw", ev_raw), ("EMA", ev_ema)):
                lines = [
                    f"  >> {tag:3s}  N=1   full_tok {ev['n1_full_token']:.3f} "
                    f"mask {ev['n1_mask_token']:.3f} board {ev['n1_full_board']:.3f}",
                    f"  >> {tag:3s}  N=20  full_tok {ev['nN_full_token']:.3f} "
                    f"mask {ev['nN_mask_token']:.3f} board {ev['nN_full_board']:.3f}",
                    f"  >> {tag:3s}  halt  full_tok {ev['halt_full_token']:.3f} "
                    f"board {ev['halt_full_board']:.3f} avg_steps {ev['halt_avg_steps']:.2f}",
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
