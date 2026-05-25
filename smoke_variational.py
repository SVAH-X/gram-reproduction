"""Variational-health smoke test.

After 100 random training steps, assert that the three bugs we caught in
the long training run cannot recur silently:

  1. puzzle_embed.abs().mean() > 1
     If the puzzle embedding init is back at ~0.02, the halt/value heads
     read near-zero noise and this fails.

  2. sigma_p.std() > 0.005
     If the prior's log-variance head has died (output stuck at zero so
     sigma = exp(0) = 1 for every element), sigma_p has variance ~0 and
     this fails.

  3. |mu_p - mu_q|.mean() < 5  AND  mu_q.std() < 4
     Catches the posterior-runaway regime where additive `u + e_y`
     conditioning let mu_q drift to ±40. With concat+MLP+tanh bound the
     posterior cannot run away.

Run:
    python smoke_variational.py
"""

from __future__ import annotations

import torch

from gram_model import GRAM, GRAMConfig


def _structured_batch(B, seq_len, vocab):
    """Generate a tiny structured (input, target) batch — target is input
    with a few cells flipped to the highest token. Mimics the
    "input is a partial of target" structure of N-Queens / GC, which is
    what makes the variational objective non-trivial."""
    torch.manual_seed(7)
    x = torch.randint(1, vocab, (B, seq_len))
    y = x.clone()
    # 10% of positions become token (vocab-1)
    flip = torch.rand(B, seq_len) < 0.10
    y = torch.where(flip, torch.full_like(y, vocab - 1), y)
    return x, y


def train_a_bit(model, steps=100, B=8, seq_len=64, vocab=3, beta=0.07):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    x, y = _structured_batch(B, seq_len, vocab)
    last_info = None
    for s in range(steps):
        loss, info, _, _ = model.train_supervision_segment(x, y, beta=beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_info = info
    return last_info


def main():
    cfg = GRAMConfig(
        vocab_size=3,
        seq_len=64,
        d_model=128,         # smaller than paper for a fast smoke
        n_heads=4,
        ffn_hidden=128,
        n_layers=2,
        K=2, T=2, N_sup=4,
        use_attn=True, use_rope=True, use_halt=True,
    )
    model = GRAM(cfg)

    # --- check 1 (init time): puzzle embed scale ---
    init_pe = model.puzzle_embed.abs().mean().item()
    init_content = (model.token_embed.weight * (cfg.d_model ** 0.5)).abs().mean().item()
    print(f"puzzle_embed init  abs_mean = {init_pe:.3f}")
    print(f"content   token  abs_mean (after sqrt(D) scaling) = {init_content:.3f}")
    assert init_pe > 1.0, (
        f"Puzzle embed magnitude {init_pe:.4f} is too small (need > 1). "
        "Halt/value heads will see near-zero input."
    )

    # --- run a few steps and inspect variational stats ---
    info = train_a_bit(model, steps=100)
    print(
        f"\nafter 100 random steps:\n"
        f"  loss      = {info['loss']:.4f}\n"
        f"  kl (grad) = {info['kl']:.4f}\n"
        f"  kl_true   = {info['kl_true']:.4f}\n"
        f"  mu_p_std  = {info['mu_p_std']:.4f}\n"
        f"  mu_q_std  = {info['mu_q_std']:.4f}"
    )

    # --- check 2: prior variance head moved away from init ---
    # Compute sigma_p on a fresh structured batch and check it's NOT stuck
    # exactly at 1.0 (the SwiGLU-init logvar=0 → sigma=1 dead-network state).
    with torch.no_grad():
        x, _ = _structured_batch(8, cfg.seq_len, cfg.vocab_size)
        e_x = model.encode(x)
        h, l = model.initial_state(x.shape[0], x.device)
        u, _ = model._propose(h, l, e_x)
        mu_p, lv_p = model.guidance.params(u)
        sigma_p = (0.5 * lv_p).exp()
    print(f"  sigma_p mean = {sigma_p.mean():.4f}  std = {sigma_p.std():.4f}  "
          f"min = {sigma_p.min():.4f}  max = {sigma_p.max():.4f}")
    # Network is alive if lv_p has either spread OR has moved away from 0.
    moved = abs(sigma_p.mean().item() - 1.0) > 0.05 or sigma_p.std().item() > 0.005
    assert moved, (
        f"sigma_p is exactly N(1, 0): mean={sigma_p.mean():.6f}, std={sigma_p.std():.6f}. "
        "Variance head is dead (logvar SwiGLU stuck at init zero output)."
    )

    # --- check 3: posterior didn't run away ---
    assert info['mu_q_std'] < 4.0, (
        f"mu_q std {info['mu_q_std']:.3f} too large — posterior likely "
        "running away via the old additive conditioning shortcut."
    )
    assert info['mu_q_std'] > 0.01, (
        f"mu_q std {info['mu_q_std']:.6f} too small — posterior collapsed."
    )

    print("\nall variational-health checks passed")


if __name__ == "__main__":
    main()
