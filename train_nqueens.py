"""GRAM Phase 3a — small-scale N-Queens 8x8 training.

Small config (D=128, N_sup=4, T=2, K=2) so it fits MacBook in <1h.
Once the metrics below clear, Phase 3b just bumps the config to
paper-scale (D=512, K=4, T=3, N_sup=16) and runs on company GPU.

Pass criteria for Phase 3a:
    - prior-only full-token acc >> 33% (random over 3 classes)
    - prior-only full-token acc >  50% (better than always-empty)
    - prior-only mask-token acc clearly > 0 — model is filling in real
      information, not just copying visible cells

Run:
    DEVICE=cpu  conda run -n pytorch python train_nqueens.py
    DEVICE=mps  conda run -n pytorch python train_nqueens.py
"""
import os
import time
import torch
from torch.optim import AdamW

from gram_model import GRAM, GRAMConfig
from data_nqueens import NQueensBatcher


@torch.no_grad()
def _accuracy(pred, y, x, mask_token):
    mask_pos = (x == mask_token)
    correct  = (pred == y)
    full_tok = correct.float().mean().item()
    mask_tok = ((correct & mask_pos).float().sum().item()
                / max(mask_pos.sum().item(), 1))
    full_board = correct.all(dim=1).float().mean().item()
    return full_tok, mask_tok, full_board


@torch.no_grad()
def evaluate(model, batcher, batch_size, device, N_best=20):
    """Compare best-of-1 (vanilla prior) against best-of-N (LPRM-selected).
    The width-axis improvement is the LPRM payoff; if N=20 is no better
    than N=1 the value head isn't doing anything useful yet."""
    model.eval()
    x, y = batcher.sample(batch_size)
    x, y = x.to(device), y.to(device)

    # best-of-1
    logits1 = model(x)
    ft1, mt1, fb1 = _accuracy(logits1.argmax(-1), y, x, batcher.MASK)

    # best-of-N
    logitsN, scores = model.forward_best_of_n(x, N=N_best)
    ftN, mtN, fbN = _accuracy(logitsN.argmax(-1), y, x, batcher.MASK)

    return {
        "n1_full_token":  ft1, "n1_mask_token":  mt1, "n1_full_board":  fb1,
        "nN_full_token":  ftN, "nN_mask_token":  mtN, "nN_full_board":  fbN,
        "score_mean": scores.mean().item(),
    }


def main():
    device = os.environ.get(
        "DEVICE", "mps" if torch.backends.mps.is_available() else "cpu"
    )
    torch.manual_seed(0)

    cfg = GRAMConfig(
        vocab_size=3,
        seq_len=64,
        d_model=128,
        n_heads=4,
        ffn_hidden=128,
        n_layers=2,
        K=2, T=2, N_sup=4,
    )
    model = GRAM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device       : {device}")
    print(f"params       : {n_params/1e6:.2f}M  (small config)")
    print(f"transitions/step: N_sup * T = {cfg.N_sup * cfg.T}")

    batcher = NQueensBatcher(n=8)
    print(f"n-queens solutions: {batcher.num_solutions()}")

    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    B = 32
    n_steps = 1000
    log_every = 50
    eval_every = 200
    beta = 0.1

    log_path = "train_nqueens.log"
    t0 = time.time()
    with open(log_path, "w") as logf:
        for step in range(1, n_steps + 1):
            x, y = batcher.sample(B)
            x, y = x.to(device), y.to(device)

            model.train()
            loss, info = model.train_step(x, y, beta=beta, kl_balance=0.8)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step == 1 or step % log_every == 0:
                line = (f"step {step:4d} | loss {info['loss']:.4f} | "
                        f"recon {info['recon']:.4f} | kl {info['kl']:.4f} | "
                        f"lprm {info['lprm']:.4f} | r {info['r']:.3f} | "
                        f"train_acc {info['acc']:.3f} | gnorm {gn.item():.2f} | "
                        f"t {time.time()-t0:.1f}s")
                print(line); logf.write(line + "\n"); logf.flush()

            if step % eval_every == 0:
                ev = evaluate(model, batcher, 64, device, N_best=20)
                line = (f"  >> eval N=1  | full_tok {ev['n1_full_token']:.3f} | "
                        f"mask_tok {ev['n1_mask_token']:.3f} | "
                        f"full_board {ev['n1_full_board']:.3f}")
                print(line); logf.write(line + "\n")
                line = (f"  >> eval N=20 | full_tok {ev['nN_full_token']:.3f} | "
                        f"mask_tok {ev['nN_mask_token']:.3f} | "
                        f"full_board {ev['nN_full_board']:.3f} | "
                        f"score~{ev['score_mean']:.2f}")
                print(line); logf.write(line + "\n"); logf.flush()

    final = evaluate(model, batcher, 128, device, N_best=20)
    print("\nfinal eval (batch=128):")
    for k, v in final.items():
        print(f"  {k}: {v:.3f}")

    torch.save({"cfg": cfg.__dict__, "model": model.state_dict()},
               "gram_nqueens_small.pt")
    print("\ncheckpoint saved to gram_nqueens_small.pt")


if __name__ == "__main__":
    main()
