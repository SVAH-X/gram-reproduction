"""Phase 2 training smoke test.

Synthetic task: y = reversed(x). Vocab=4, seq_len=8.
A real GRAM solves Sudoku/N-Queens; we don't care about THIS task — we just
need to confirm:
  1. train_step() runs without error
  2. loss decreases monotonically (or at least dramatically) over a few
     hundred steps
  3. token accuracy improves toward 1.0
  4. KL stays finite and doesn't collapse to 0 immediately (posterior
     collapse is a real risk in VAE-style training)

If all four hold, the truncated surrogate ELBO machinery is sound and
we can move to Phase 3 (real N-Queens data).

Run:
    conda run -n pytorch python train_smoke.py
"""
import torch
from torch.optim import AdamW
from gram_model import GRAM, GRAMConfig


def make_batch(B, seq_len, vocab_size, device):
    """y = reversed(x). Both have shape (B, seq_len), int64 ids in [0, vocab)."""
    x = torch.randint(0, vocab_size, (B, seq_len), device=device)
    y = x.flip(dims=[1])
    return x, y


def main():
    import os
    device = os.environ.get("DEVICE",
                            "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)

    cfg = GRAMConfig(
        vocab_size=4,
        seq_len=8,
        d_model=128,
        n_heads=4,
        ffn_hidden=128,
        n_layers=2,
        K=2, T=2, N_sup=4,
    )
    model = GRAM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device       : {device}")
    print(f"smoke params : {n_params/1e6:.2f}M")
    print(f"transitions/step: N_sup * T = {cfg.N_sup * cfg.T}")

    # Paper hparams scaled to smoke: same lr, same wd, same kl_balance, smaller batch.
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    B = 32
    n_steps = 600
    log_every = 50
    beta = 0.3   # higher than paper N-Queens (0.07) because synthetic task
                 # is too easy — posterior trivially memorizes y unless KL
                 # is held tight enough to drag prior along.

    log_path = "train_smoke.log"
    with open(log_path, "w") as logf:
        for step in range(1, n_steps + 1):
            x, y = make_batch(B, cfg.seq_len, cfg.vocab_size, device)
            model.train()
            loss, info = model.train_step(x, y, beta=beta, kl_balance=0.8)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step == 1 or step % log_every == 0:
                line = (
                    f"step {step:4d} | loss {info['loss']:.4f} | "
                    f"recon {info['recon']:.4f} | kl {info['kl']:.4f} | "
                    f"acc {info['acc']:.3f} | gnorm {grad_norm.item():.2f}"
                )
                print(line)
                logf.write(line + "\n")
                logf.flush()

    # Final eval pass: pure prior sampling on a fresh batch, see if argmax
    # actually solves "reverse" without seeing y.
    model.eval()
    x, y = make_batch(64, cfg.seq_len, cfg.vocab_size, device)
    logits = model(x)
    pred = logits.argmax(-1)
    eval_acc = (pred == y).float().mean().item()
    print(f"\neval (prior-only, batch=64): token acc = {eval_acc:.3f}")
    print(f"\nlog written to {log_path}")


if __name__ == "__main__":
    main()
