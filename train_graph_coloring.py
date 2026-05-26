"""Paper-aligned GRAM training for Graph Coloring.

This script follows Appendix C.2.2 as closely as possible:
  * n in {8, 10}, 3 colors.
  * Input is the strict upper triangle of the adjacency matrix.
  * Output is the length-n node-color sequence.
  * Valid 3-colorings are enumerated and canonicalized to remove color
    permutation duplicates.
  * Paper train/test graph counts are used by default.
  * Paper epochs are treated as trajectory batches; each trajectory performs
    N_sup segment-level gradient updates.

The paper says graphs are Erdos-Renyi with a fixed edge probability but does
not state that probability. The default remains p=0.4 because this project
already used that value; pass --p-edge if you want to sweep it.
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from gram_model import EMA, GRAM, GRAMConfig
from data_graph_coloring import (
    VOCAB_SIZE,
    GraphColoringEvalSet,
    GraphColoringTrainDataset,
    build_graph_coloring,
    color_logits_to_tokens,
    conflict_edges,
    default_split_sizes,
    graph_coverage,
    load_graph_coloring_cache,
    steps_per_epoch,
)


@contextmanager
def _nullcontext():
    yield


def paper_beta(n: int) -> float:
    if n == 8:
        return 0.5
    if n == 10:
        return 0.45
    raise ValueError("The paper specifies Graph Coloring only for n=8 and n=10.")


@torch.no_grad()
def evaluate(model, eval_set, batch_size, device, samples=20, max_examples=None, use_ema=False, ema=None):
    model.eval()
    ctx = ema.swap_in(model) if use_ema else _nullcontext()
    total = 0
    conflict_sum = 0.0
    coverage_sum = 0.0
    with ctx:
        for x, adj, colorings in eval_set.batches(batch_size, max_examples=max_examples):
            x = x.to(device)
            adj = adj.to(device)
            logits = model(x)
            pred_colors = color_logits_to_tokens(logits, eval_set.n)
            bsz = x.shape[0]
            conflict_sum += conflict_edges(pred_colors, adj).sum().item()

            preds = []
            for _ in range(samples):
                logits_s = model(x)
                preds.append(color_logits_to_tokens(logits_s, eval_set.n).cpu())
            sample_tensor = torch.stack(preds, dim=1)
            coverage_sum += graph_coverage(sample_tensor, colorings, eval_set.n) * bsz
            total += bsz
    return {
        "conflicts": conflict_sum / max(total, 1),
        "coverage": coverage_sum / max(total, 1),
        "n_eval": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, choices=[8, 10])
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=None, help="Optional debug cap; overrides full epoch count.")
    ap.add_argument("--global-batch", type=int, default=768)
    ap.add_argument("--b-per-step", type=int, default=64)
    ap.add_argument("--p-edge", type=float, default=0.4)
    ap.add_argument("--train-size", type=int, default=None, help="Paper default: 7002 for n=8, 13465 for n=10.")
    ap.add_argument("--test-size", type=int, default=None, help="Paper default: 255 for n=8, 192 for n=10.")
    ap.add_argument("--data-cache", type=str, default=None, help="Explicit prepare_data.py cache file. Defaults to data_cache/graphcolor_n{n}_p{p}_seed{seed}.pt when present.")
    ap.add_argument("--cache-dir", type=str, default=None, help="Optional generator cache for custom sweeps; default is disabled.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--kl-balance", type=float, default=0.8)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--halt-weight", type=float, default=1.0)
    ap.add_argument("--lprm-weight", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=0, help="Paper does not specify warmup; default is none.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--eval-max", type=int, default=512, help="Use 0 for full test set during periodic eval.")
    ap.add_argument("--coverage-samples", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--out-prefix", type=str, default=None)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    if args.global_batch % args.b_per_step != 0:
        raise ValueError("--global-batch must be divisible by --b-per-step.")
    if args.beta is None:
        args.beta = paper_beta(args.n)
    default_train, default_test = default_split_sizes(args.n)
    train_size = args.train_size or default_train
    test_size = args.test_size or default_test

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    use_amp = (device == "cuda") and (not args.no_amp) and torch.cuda.is_bf16_supported()
    amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_amp else _nullcontext

    default_cache = Path("data_cache") / f"graphcolor_n{args.n}_p{args.p_edge:g}_seed{args.seed}.pt"
    use_default_cache = args.train_size is None and args.test_size is None and default_cache.exists()
    if args.data_cache or use_default_cache:
        data_cache = Path(args.data_cache) if args.data_cache else default_cache
        build = load_graph_coloring_cache(data_cache)
        if build.n != args.n:
            raise ValueError(f"Cache {data_cache} is for n={build.n}, but --n={args.n}.")
        args.p_edge = build.p_edge
        data_source = str(data_cache)
    else:
        build = build_graph_coloring(
            args.n,
            p_edge=args.p_edge,
            train_size=train_size,
            test_size=test_size,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )
        data_source = "generated in memory"

    train_ds = GraphColoringTrainDataset(build, seed=args.seed)
    test_eval = GraphColoringEvalSet(build, split="test")
    dataset_batches_per_pass = steps_per_epoch(len(train_ds), args.global_batch)
    seq_len = args.n * (args.n - 1) // 2
    target_seq_len = args.n

    cfg = GRAMConfig(
        vocab_size=VOCAB_SIZE,
        seq_len=seq_len,
        target_seq_len=target_seq_len,
        d_model=512,
        n_heads=8,
        ffn_hidden=512,
        n_layers=2,
        K=4,
        T=3,
        N_sup=16,
        use_attn=True,
        use_rope=True,
        use_halt=True,
    )
    model = GRAM(cfg).to(device)
    ema = EMA(model, decay=args.ema_decay)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    planned_steps = args.epochs * cfg.N_sup
    total_steps = min(planned_steps, args.max_steps) if args.max_steps else planned_steps

    out_prefix = args.out_prefix or f"gram_graphcolor_n{args.n}"
    print(f"device           : {device}")
    print(f"amp              : {'bf16' if use_amp else 'off'}")
    print(f"task             : Graph Coloring n={args.n}")
    print(f"data source      : {data_source}")
    print(f"params           : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"input seq_len    : {seq_len} strict-upper-triangle tokens")
    print(f"target seq_len   : {target_seq_len} node-color tokens")
    print(f"edge probability : {args.p_edge} (paper does not state this value)")
    print(f"train/test graphs: {len(build.train_graphs)} / {len(build.test_graphs)}")
    print(f"global batch     : {args.global_batch} ({args.b_per_step} microbatch)")
    print(f"dataset batches/pass      : {dataset_batches_per_pass}")
    print(f"paper trajectory batches  : {args.epochs}")
    print(f"segment updates/trajectory: {cfg.N_sup}")
    print(f"planned training steps    : {planned_steps}")
    print(f"running training steps    : {total_steps}")
    print(f"beta             : {args.beta}")

    loader_gen = torch.Generator().manual_seed(args.seed)
    log_path = f"{out_prefix}.log"
    logf = open(log_path, "w")
    t0 = time.time()
    global_step = 0

    def lr_scale(step: int) -> float:
        if args.warmup_steps <= 0:
            return 1.0
        return min(1.0, step / max(args.warmup_steps, 1))

    try:
        loader = DataLoader(
            train_ds,
            batch_size=args.global_batch,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
            generator=loader_gen,
        )
        loader_iter = iter(loader)
        trajectory = 0
        while trajectory < args.epochs and global_step < total_steps:
            try:
                x_global, y_global = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x_global, y_global = next(loader_iter)
            trajectory += 1
            x_global = x_global.to(device)
            y_global = y_global.to(device)
            batch_actual = x_global.shape[0]
            states = [None] * ((batch_actual + args.b_per_step - 1) // args.b_per_step)

            for segment in range(1, cfg.N_sup + 1):
                if global_step >= total_steps:
                    break
                global_step += 1
                for group in opt.param_groups:
                    group["lr"] = args.lr * lr_scale(global_step)
                opt.zero_grad(set_to_none=True)
                info_accum = {"loss": 0.0, "recon": 0.0, "recon_p": 0.0, "recon_q": 0.0,
                              "kl": 0.0, "kl_true": 0.0,
                              "mu_p_std": 0.0, "mu_q_std": 0.0,
                              "lprm": 0.0, "halt": 0.0, "r": 0.0, "acc": 0.0, "acc_q": 0.0}
                next_states = []

                for mi, start in enumerate(range(0, batch_actual, args.b_per_step)):
                    end = min(start + args.b_per_step, batch_actual)
                    weight = (end - start) / batch_actual
                    x = x_global[start:end]
                    y = y_global[start:end]
                    h, l = states[mi] if states[mi] is not None else (None, None)
                    model.train()
                    with amp_ctx():
                        loss, info, h_next, l_next = model.train_supervision_segment(
                            x,
                            y,
                            h=h,
                            l=l,
                            beta=args.beta,
                            kl_balance=args.kl_balance,
                            lprm_weight=args.lprm_weight,
                            halt_weight=args.halt_weight,
                        )
                    (loss * weight).backward()
                    next_states.append((h_next, l_next))
                    for key in info_accum:
                        info_accum[key] += info[key] * weight
                states = next_states

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ema.update(model)

                if global_step == 1 or global_step % args.log_every == 0:
                    msg = (
                        f"traj {trajectory:5d}/{args.epochs} seg {segment:2d}/{cfg.N_sup} "
                        f"step {global_step:7d}/{total_steps} | "
                        f"loss {info_accum['loss']:.4f} "
                        f"rp {info_accum['recon_p']:.4f} rq {info_accum['recon_q']:.4f} "
                        f"kl {info_accum['kl']:.4f} kl_true {info_accum['kl_true']:.3f} "
                        f"mp {info_accum['mu_p_std']:.3f} mq {info_accum['mu_q_std']:.3f} "
                        f"halt {info_accum['halt']:.4f} "
                        f"lprm {info_accum['lprm']:.4f} r {info_accum['r']:.3f} "
                        f"acc_p {info_accum['acc']:.3f} acc_q {info_accum['acc_q']:.3f} "
                        f"gn {grad_norm.item():.2f} t {time.time() - t0:.1f}s"
                    )
                    print(msg)
                    logf.write(msg + "\n")
                    logf.flush()

                if args.eval_every and global_step % args.eval_every == 0:
                    eval_max = None if args.eval_max == 0 else args.eval_max
                    for tag, use_ema in (("raw", False), ("EMA", True)):
                        ev = evaluate(
                            model,
                            test_eval,
                            args.eval_batch,
                            device,
                            samples=args.coverage_samples,
                            max_examples=eval_max,
                            use_ema=use_ema,
                            ema=ema,
                        )
                        line = (
                            f"  >> {tag:3s} test n={ev['n_eval']} "
                            f"conflicts {ev['conflicts']:.4f} coverage@{args.coverage_samples} {ev['coverage']:.4f}"
                        )
                        print(line)
                        logf.write(line + "\n")
                    logf.flush()

                if args.ckpt_every and global_step % args.ckpt_every == 0:
                    path = f"{out_prefix}_step{global_step}.pt"
                    torch.save(
                        {
                            "cfg": cfg.__dict__,
                            "model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "step": global_step,
                            "trajectory": trajectory,
                            "segment": segment,
                            "args": vars(args),
                        },
                        path,
                    )
                    print(f"  >> ckpt saved: {path}")
    finally:
        final_path = f"{out_prefix}_final.pt"
        torch.save(
            {
                "cfg": cfg.__dict__,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "step": global_step,
                "args": vars(args),
            },
            final_path,
        )
        logf.close()
        print(f"done - saved {final_path}")


if __name__ == "__main__":
    main()
