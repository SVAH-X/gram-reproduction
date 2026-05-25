"""Phase 1 smoke test — verify shapes flow end-to-end.

Run:
    conda run -n pytorch python smoke_test.py

We use *reduced* recursion depth (N_sup=2, T=2, K=2) so this completes in
a few seconds on CPU. The full paper config (N_sup=16, T=3, K=4 = 192
transitions) would still finish on MPS but is overkill just to check shapes.
"""
import torch
from gram_model import GRAM, GRAMConfig


def test_shapes():
    cfg = GRAMConfig(
        vocab_size=3,
        seq_len=64,
        d_model=128,    # smaller than paper (D=512) for quick smoke test
        n_heads=4,
        ffn_hidden=128,
        n_layers=2,
        K=2, T=2, N_sup=2,
    )
    torch.manual_seed(0)
    model = GRAM(cfg).eval()

    B = 2
    x = torch.randint(0, cfg.vocab_size, (B, cfg.seq_len))
    with torch.no_grad():
        logits = model(x)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"input  shape : {tuple(x.shape)}")
    print(f"output shape : {tuple(logits.shape)}")
    print(f"params       : {n_params/1e6:.2f}M (smoke config)")
    assert logits.shape == (B, cfg.seq_len, cfg.vocab_size), "shape mismatch"
    assert torch.isfinite(logits).all(), "non-finite logits"
    print("shapes OK, all finite")


def test_stochasticity():
    """Two forward passes on the same input should differ — confirms
    epsilon sampling is actually injecting noise."""
    cfg = GRAMConfig(d_model=64, n_heads=2, ffn_hidden=64,
                     K=1, T=1, N_sup=1, seq_len=16)
    torch.manual_seed(1)
    model = GRAM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len))
    with torch.no_grad():
        a = model(x)
        b = model(x)
    diff = (a - b).abs().mean().item()
    print(f"two-sample diff (mean abs): {diff:.4f}  (>0 means noise active)")
    assert diff > 1e-4, "outputs identical across samples — stochasticity broken"
    print("stochastic guidance is actually sampling")


def test_paper_scale_config():
    """Build a model at paper-scale (D=512, K=4, T=3, N_sup=16) and
    just count params + run one transition. Tells us whether the full
    config is even parameterizable on this machine."""
    cfg = GRAMConfig()  # defaults = paper non-Sudoku
    model = GRAM(cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\npaper-scale params: {n_params/1e6:.2f}M  (target ~10M)")
    x = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len))
    e_x = model.encode(x)
    h = model.h0.expand(1, -1, -1).contiguous()
    l = model.l0.expand(1, -1, -1).contiguous()
    with torch.no_grad():
        h2, l2, mu, lv = model.transition(h, l, e_x)
    print(f"single-transition shapes: h={tuple(h2.shape)} l={tuple(l2.shape)} "
          f"mu={tuple(mu.shape)} log_var={tuple(lv.shape)}")


if __name__ == "__main__":
    test_shapes()
    print()
    test_stochasticity()
    test_paper_scale_config()
