"""GRAM — full implementation including RoPE, ACT halt head, EMA, and LPRM.

Paper: Baek, Jo, Kim, Ren, Bengio, Ahn — "Generative Recursive Reasoning"
       arXiv:2605.19376, May 2026.

Per supervision step s in 0..N_sup-1:
    - K inner f_L iterations followed by f_H proposal -> u_t
    - eps ~ q_phi(eps | u, e_y)        (training)  or  p_theta(eps | u)   (eval)
    - h_t = u_t + eps
    - decode logits = lm_head(norm_out(h_t))
    - cross-entropy reconstruction + balanced KL(q || p)
    - halt_logit_t = q_head(h_t)         BCE against per-example correctness

Final hidden h_T feeds value_head (LPRM) for best-of-N selection at inference.
"""
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GRAMConfig:
    vocab_size: int = 3
    seq_len:    int = 64
    target_seq_len: Optional[int] = None
    d_model:    int = 512
    n_heads:    int = 8
    ffn_hidden: int = 512
    n_layers:   int = 2          # f_L and f_H each have 2 blocks
    K:          int = 4          # inner loop iterations per transition
    T:          int = 3          # transitions per supervision step
    N_sup:      int = 16         # deep supervision steps
    use_attn:   bool = True      # Sudoku variant flips this off
    use_rope:   bool = True
    use_halt:   bool = True
    rope_theta: float = 10000.0
    # Paper B.1: "prepended with 16 puzzle embedding tokens".
    # HRM-style register tokens read by the halt/value heads via h[:,0].
    num_puzzle_tokens: int = 16

    @property
    def state_seq_len(self) -> int:
        return self.seq_len + self.num_puzzle_tokens


# ---------------------------------------------------------------- norms / FFN

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        n = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * n).type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_in, d_hidden, d_out=None):
        super().__init__()
        d_out = d_out or d_in
        self.gate = nn.Linear(d_in, d_hidden, bias=False)
        self.up   = nn.Linear(d_in, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ----------------------------------------------------------------------- RoPE

def precompute_rope(d_head: int, max_seq_len: int, theta: float = 10000.0):
    """Return (cos, sin) of shape (max_seq_len, d_head). Uses the standard
    rotate_half formulation: pair (x_i, x_{i+d/2}) → rotate by m * theta_i."""
    assert d_head % 2 == 0
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))   # (d_head/2,)
    pos = torch.arange(max_seq_len).float()
    freqs = torch.einsum("m,d->md", pos, inv_freq)                              # (S, d_head/2)
    emb = torch.cat([freqs, freqs], dim=-1)                                     # (S, d_head)
    return emb.cos(), emb.sin()


def rotate_half(x):
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """q, k: (B, h, S, dh). cos, sin: (S, dh)."""
    cos = cos[None, None]
    sin = sin[None, None]
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


# ---------------------------------------------------------------- attention

class Attention(nn.Module):
    """Multi-head bidirectional self-attention with optional RoPE."""
    def __init__(self, cfg: GRAMConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.h  = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.use_rope = cfg.use_rope
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        if cfg.use_rope:
            cos, sin = precompute_rope(self.dh, cfg.state_seq_len, cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                # (B, h, S, dh)
        if self.use_rope:
            q, k = apply_rope(q, k, self.rope_cos[:S], self.rope_sin[:S])
        out = F.scaled_dot_product_attention(q, k, v)                   # (B, h, S, dh)
        return self.proj(out.transpose(1, 2).reshape(B, S, D))


class TransformerBlock(nn.Module):
    """Pre-norm: x + Mix(norm(x)); x + SwiGLU(norm(x))."""
    def __init__(self, cfg: GRAMConfig):
        super().__init__()
        self.use_attn = cfg.use_attn
        self.norm1 = RMSNorm(cfg.d_model)
        if cfg.use_attn:
            self.mix = Attention(cfg)
        else:
            self.mix = SwiGLU(cfg.d_model, cfg.ffn_hidden)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn   = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x):
        x = x + self.mix(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RecursiveModule(nn.Module):
    """f_L or f_H. Inputs are summed at the residual stream. A final RMSNorm
    bounds output magnitude — without it the residual stream from K inner
    iterations × N_sup·T transitions explodes and the prior/posterior KL
    blows up to ~1e4 element-wise."""
    def __init__(self, cfg: GRAMConfig):
        super().__init__()
        self.blocks   = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm_out = RMSNorm(cfg.d_model)

    def forward(self, *streams):
        x = streams[0]
        for s in streams[1:]:
            x = x + s
        for blk in self.blocks:
            x = blk(x)
        return self.norm_out(x)


# ---------------------------------------------------------- guidance / heads

class StochasticGuidance(nn.Module):
    """Diagonal Gaussian head — mu and log_var via separate SwiGLUs.

    Two construction modes:
      * d_in = D    : prior path. Input is u.
      * d_in = 2*D  : posterior path. Input is concat([u, e_y]).
                      Replaces the original additive `u + e_y` conditioning,
                      which let the posterior shortcut around u by encoding
                      the answer as a giant additive shift (μ_q drifted to ±40
                      while μ_p stayed near 0, killing the prior's gradient).
    A tanh*mu_scale bound on μ also blocks the runaway: posterior can no
    longer trivially memorize y via unbounded shifts.
    """
    def __init__(self, cfg: GRAMConfig, d_in: Optional[int] = None,
                 log_var_min=-10.0, log_var_max=2.0, mu_scale: float = 4.0):
        super().__init__()
        d_in = d_in or cfg.d_model
        self.mu_net      = SwiGLU(d_in, cfg.ffn_hidden, cfg.d_model)
        self.logvar_net  = SwiGLU(d_in, cfg.ffn_hidden, cfg.d_model)
        self.lv_min, self.lv_max = log_var_min, log_var_max
        self.mu_scale = mu_scale

    def params(self, x):
        # tanh-bounded μ prevents the |mu_q| → 40 runaway that decoupled
        # posterior from prior. mu_scale=4 is loose enough that signal is
        # plenty for prediction but tight enough that posterior can't
        # encode the answer as an unbounded shift.
        mu      = self.mu_scale * torch.tanh(self.mu_net(x) / self.mu_scale)
        log_var = self.logvar_net(x).clamp(self.lv_min, self.lv_max)
        return mu, log_var

    def forward(self, x):
        mu, log_var = self.params(x)
        eps = mu + (0.5 * log_var).exp() * torch.randn_like(mu)
        return eps, mu, log_var


def kl_gaussian(mu_q, lv_q, mu_p, lv_p):
    # Force fp32 — under bf16 autocast, var.exp() and var division can lose
    # precision and cause KL drift. Inputs come from clamped log_var so the
    # cast back to fp32 is cheap and exact.
    mu_q = mu_q.float(); lv_q = lv_q.float()
    mu_p = mu_p.float(); lv_p = lv_p.float()
    var_q = lv_q.exp()
    var_p = lv_p.exp()
    return 0.5 * (var_q / var_p
                  + (mu_p - mu_q).pow(2) / var_p
                  + lv_p - lv_q
                  - 1.0)


class ValueHead(nn.Module):
    """LPRM (V-head). Paper Table 4: Linear(D -> 1).

    It reads the first token of h and predicts r in [0, 1]. Earlier local
    versions used a SwiGLU projection here, which inflated the model by about
    0.8M parameters per head and moved it away from the paper's ~10M setting.
    """
    def __init__(self, cfg: GRAMConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1, bias=True)

    def forward(self, h):
        z = self.norm(h[:, 0])
        return torch.sigmoid(self.head(z).squeeze(-1))


class HaltHead(nn.Module):
    """ACT halt-only Q-head. Paper Table 4: Linear(D -> 2).

    We keep the paper's released-code simplification described in Appendix A.1:
    inference can use only q^halt with threshold 0.5.
    """
    def __init__(self, cfg: GRAMConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 2, bias=True)

    def forward(self, h):
        z = self.norm(h[:, 0])
        return self.head(z)                     # (B, 2): [halt_q, continue_q]


# ---------------------------------------------------------- EMA

class EMA:
    """Exponential moving average of model float parameters. Paper uses
    decay=0.9999. Use `with ema.swap_in(model):` to evaluate using the
    EMA weights, then revert automatically."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            n: p.detach().clone().float()
            for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1 - d)

    @contextmanager
    def swap_in(self, model: nn.Module):
        backup = {}
        try:
            for n, p in model.named_parameters():
                if n in self.shadow:
                    backup[n] = p.detach().clone()
                    p.data.copy_(self.shadow[n].to(p.dtype))
            yield
        finally:
            for n, p in model.named_parameters():
                if n in backup:
                    p.data.copy_(backup[n])

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay  = state["decay"]
        self.shadow = state["shadow"]


# ---------------------------------------------------------- GRAM

class GRAM(nn.Module):
    def __init__(self, cfg: GRAMConfig):
        super().__init__()
        self.cfg = cfg
        self.target_seq_len = cfg.target_seq_len or cfg.seq_len
        self.num_puzzle_tokens = cfg.num_puzzle_tokens
        state_seq_len = cfg.state_seq_len

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Paper B.1: 16 prepended puzzle/register tokens, shared across the
        # batch and learned. The halt and value heads read position 0.
        # Init at the post-scaling magnitude of content tokens. Content tokens
        # use the default nn.Embedding init N(0, 1) which we then scale by
        # sqrt(D) in `encode`, so their per-element std is sqrt(D). Initing
        # puzzle_embed at N(0, sqrt(D)²) keeps both streams on the same scale.
        # The previous N(0, 0.02²) init left puzzle tokens ~1000× too small
        # and the halt/value heads (which read h[:, 0]) saw near-zero input.
        self.puzzle_embed = nn.Parameter(
            torch.randn(1, cfg.num_puzzle_tokens, cfg.d_model) * math.sqrt(cfg.d_model)
        )
        # learned pos embed only used when RoPE is off
        if not cfg.use_rope:
            self.pos_embed = nn.Parameter(torch.zeros(1, state_seq_len, cfg.d_model))

        self.f_L           = RecursiveModule(cfg)
        self.f_H           = RecursiveModule(cfg)
        self.guidance      = StochasticGuidance(cfg, d_in=cfg.d_model)
        # Posterior receives concat([u, e_y]); see StochasticGuidance docstring
        # for why additive conditioning was broken.
        self.guidance_post = StochasticGuidance(cfg, d_in=2 * cfg.d_model)

        self.norm_out  = RMSNorm(cfg.d_model)
        # Paper Table 4: Linear(D -> vocab).
        self.lm_head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)
        self.value_head = ValueHead(cfg)
        if cfg.use_halt:
            self.halt_head = HaltHead(cfg)

        if self.target_seq_len != cfg.seq_len:
            self.target_to_state = nn.Linear(self.target_seq_len, cfg.seq_len, bias=False)
            self.state_to_target = nn.Linear(cfg.seq_len, self.target_seq_len, bias=False)
        else:
            self.target_to_state = None
            self.state_to_target = None

        # Frozen z_0 = (h_0, l_0). State length includes 16 puzzle positions.
        self.register_buffer("h0", torch.randn(1, state_seq_len, cfg.d_model))
        self.register_buffer("l0", torch.randn(1, state_seq_len, cfg.d_model))

    def encode(self, x_ids):
        # Paper B.1: scale by sqrt(D), then prepend 16 puzzle embedding tokens.
        B = x_ids.shape[0]
        e = self.token_embed(x_ids) * math.sqrt(self.cfg.d_model)
        puzzle = self.puzzle_embed.expand(B, -1, -1)
        e = torch.cat([puzzle, e], dim=1)
        if not self.cfg.use_rope:
            e = e + self.pos_embed
        return e

    def encode_target_as_state(self, y_ids):
        # Paper supports posterior conditioning on the target. The target lives
        # in the content portion of the latent state; puzzle positions get a
        # zero conditioning signal so they are still driven by the prior.
        B = y_ids.shape[0]
        e_y = self.token_embed(y_ids) * math.sqrt(self.cfg.d_model)
        if self.target_to_state is not None:
            e_y = self.target_to_state(e_y.transpose(1, 2)).transpose(1, 2)
        pad = torch.zeros(
            B, self.num_puzzle_tokens, self.cfg.d_model,
            dtype=e_y.dtype, device=e_y.device,
        )
        return torch.cat([pad, e_y], dim=1)

    def decode(self, h):
        # Paper B.1: "The decoder extracts content tokens (excluding puzzle
        # embedding positions) and maps them to logits via a SwiGLU MLP head."
        h_content = h[:, self.num_puzzle_tokens:]
        z = self.norm_out(h_content)
        if self.state_to_target is not None:
            z = self.state_to_target(z.transpose(1, 2)).transpose(1, 2)
        return self.lm_head(z)

    def _propose(self, h_prev, l_prev, e_x):
        l = l_prev
        for _ in range(self.cfg.K):
            l = self.f_L(h_prev, l, e_x)
        u = self.f_H(h_prev, l)
        return u, l

    def transition(self, h_prev, l_prev, e_x):
        u, l = self._propose(h_prev, l_prev, e_x)
        eps, mu, log_var = self.guidance(u)
        return u + eps, l, mu, log_var

    def transition_train(self, h_prev, l_prev, e_x, e_y):
        u, l = self._propose(h_prev, l_prev, e_x)
        mu_p, lv_p = self.guidance.params(u)
        mu_q, lv_q = self.guidance_post.params(torch.cat([u, e_y], dim=-1))
        eps = mu_q + (0.5 * lv_q).exp() * torch.randn_like(mu_q)
        return u + eps, l, (mu_p, lv_p), (mu_q, lv_q)

    def initial_state(self, batch_size: int, device=None):
        device = device or self.h0.device
        h = self.h0.expand(batch_size, -1, -1).to(device).contiguous()
        l = self.l0.expand(batch_size, -1, -1).to(device).contiguous()
        return h, l

    def train_supervision_segment(self, x_ids, y_ids, h=None, l=None,
                                  beta: float = 0.07,
                                  kl_balance: float = 0.8,
                                  lprm_weight: float = 1.0,
                                  halt_weight: float = 0.5,
                                  y_mask: Optional[torch.Tensor] = None):
        """Train exactly one deep-supervision segment.

        This is the paper-aligned update unit: each segment runs T transitions,
        propagates gradients only through the final transition, applies the
        surrogate objective for that segment, and returns detached terminal
        state for the next segment.
        """
        B = x_ids.shape[0]
        e_x = self.encode(x_ids)
        e_y = self.encode_target_as_state(y_ids)
        if h is None or l is None:
            h, l = self.initial_state(B, x_ids.device)

        n_valid = y_mask.float().sum(-1).clamp(min=1.0) if y_mask is not None else None

        with torch.no_grad():
            for _ in range(self.cfg.T - 1):
                h, l, _, _ = self.transition(h, l, e_x)
        h, l, (mu_p, lv_p), (mu_q, lv_q) = self.transition_train(h, l, e_x, e_y)

        logits = self.decode(h)
        if y_mask is None:
            recon = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                y_ids.reshape(-1),
            )
        else:
            y_for_loss = y_ids.masked_fill(~y_mask, -100)
            recon = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                y_for_loss.reshape(-1),
                ignore_index=-100,
            )

        kl_post_grad = kl_gaussian(mu_q, lv_q, mu_p.detach(), lv_p.detach()).mean()
        kl_prior_grad = kl_gaussian(mu_q.detach(), lv_q.detach(), mu_p, lv_p).mean()
        kl = kl_balance * kl_prior_grad + (1.0 - kl_balance) * kl_post_grad
        elbo_loss = recon + beta * kl

        # True symmetric KL — what we actually care about for diagnosis.
        # The `kl` above is gradient-shaped (one half is detached on each side
        # for KL-balancing) and can read as ~0.01 while the actual posterior
        # has drifted to |μ_q| ~ 40, which we caught only by direct inspection.
        with torch.no_grad():
            kl_true = kl_gaussian(mu_q, lv_q, mu_p, lv_p).mean()
            mu_p_std = mu_p.std()
            mu_q_std = mu_q.std()

        with torch.no_grad():
            pred = logits.argmax(-1)
            correct = pred == y_ids
            if y_mask is None:
                target = correct.float().mean(dim=-1)
                acc = target.mean()
            else:
                target = (correct & y_mask).float().sum(-1) / n_valid
                acc = ((correct & y_mask).float().sum() / y_mask.float().sum().clamp(min=1)).detach()

        halt_loss = torch.tensor(0.0, device=elbo_loss.device)
        if self.cfg.use_halt:
            # Appendix A.1: ACT loss contributes only through the halt head.
            halt_logits = self.halt_head(h.detach())
            halt_loss = F.binary_cross_entropy_with_logits(halt_logits[:, 0], target)

        # Appendix A.2: value head predicts trajectory quality from latent state.
        # Detach h so the reward model does not alter the recursive core update.
        score = self.value_head(h.detach())
        lprm_loss = F.mse_loss(score, target)

        loss = elbo_loss + lprm_weight * lprm_loss
        if self.cfg.use_halt:
            loss = loss + halt_weight * halt_loss

        return loss, {
            "loss": loss.item(),
            "recon": recon.item(),
            "kl": kl.item(),
            "kl_true": kl_true.item(),
            "mu_p_std": mu_p_std.item(),
            "mu_q_std": mu_q_std.item(),
            "lprm": lprm_loss.item(),
            "halt": halt_loss.item() if self.cfg.use_halt else 0.0,
            "r": target.mean().item(),
            "acc": acc.item(),
        }, h.detach(), l.detach()

    @torch.no_grad()
    def _run_prior_trajectory(self, e_x):
        B = e_x.shape[0]
        h = self.h0.expand(B, -1, -1).contiguous()
        l = self.l0.expand(B, -1, -1).contiguous()
        for _ in range(self.cfg.N_sup * self.cfg.T):
            h, l, _, _ = self.transition(h, l, e_x)
        return h

    @torch.no_grad()
    def forward(self, x_ids):
        e_x = self.encode(x_ids)
        h_T = self._run_prior_trajectory(e_x)
        return self.decode(h_T)

    @torch.no_grad()
    def forward_best_of_n(self, x_ids, N: int = 20):
        B = x_ids.shape[0]
        e_x = self.encode(x_ids)
        e_x = e_x.repeat_interleave(N, dim=0)
        h_T = self._run_prior_trajectory(e_x)
        scores = self.value_head(h_T)

        scores = scores.view(B, N)
        h_T    = h_T.view(B, N, *h_T.shape[1:])
        best   = scores.argmax(dim=1)
        best_h = h_T[torch.arange(B), best]
        return self.decode(best_h), scores

    @torch.no_grad()
    def forward_with_halt(self, x_ids, halt_thresh: float = 0.5):
        """Run supervision steps one at a time and halt per-example as soon
        as sigmoid(q^halt) crosses the threshold. Returns (logits, n_steps).
        Caps at N_sup steps."""
        assert self.cfg.use_halt
        B = x_ids.shape[0]
        e_x = self.encode(x_ids)
        h = self.h0.expand(B, -1, -1).contiguous()
        l = self.l0.expand(B, -1, -1).contiguous()

        device = x_ids.device
        halted   = torch.zeros(B, dtype=torch.bool, device=device)
        n_steps  = torch.zeros(B, dtype=torch.long,  device=device)
        h_final  = torch.zeros_like(h)

        for s in range(self.cfg.N_sup):
            for _ in range(self.cfg.T):
                h, l, _, _ = self.transition(h, l, e_x)
            halt_p = torch.sigmoid(self.halt_head(h)[:, 0])      # (B,)
            new_halt = (~halted) & (halt_p > halt_thresh)
            h_final[new_halt] = h[new_halt]
            n_steps[new_halt] = s + 1
            halted = halted | new_halt
            if halted.all():
                break
        # any examples that never halted -> use final h
        not_halted = ~halted
        h_final[not_halted] = h[not_halted]
        n_steps[not_halted] = self.cfg.N_sup
        return self.decode(h_final), n_steps

    def train_step(self, x_ids, y_ids, beta: float = 0.07,
                   kl_balance: float = 0.8,
                   lprm_weight: float = 1.0,
                   halt_weight: float = 0.5,
                   y_mask: Optional[torch.Tensor] = None):
        """Train one batch with optional target masking.

        x_ids has shape (B, input_seq_len). y_ids has shape
        (B, target_seq_len), where target_seq_len may differ from input_seq_len.
        y_mask is optional bool (B, target_seq_len). When provided, recon CE is computed
        only at True positions, and halt-target / acc / LPRM-target are
        per-example correctness over True positions only."""
        B = x_ids.shape[0]
        e_x = self.encode(x_ids)
        e_y = self.encode_target_as_state(y_ids)
        h = self.h0.expand(B, -1, -1).contiguous()
        l = self.l0.expand(B, -1, -1).contiguous()

        n_valid = y_mask.float().sum(-1).clamp(min=1.0) if y_mask is not None else None

        recon_sum = 0.0
        kl_sum    = 0.0
        halt_sum  = 0.0
        acc_sum   = 0.0

        for _ in range(self.cfg.N_sup):
            with torch.no_grad():
                for _ in range(self.cfg.T - 1):
                    h, l, _, _ = self.transition(h, l, e_x)
            h, l, (mu_p, lv_p), (mu_q, lv_q) = self.transition_train(h, l, e_x, e_y)

            logits = self.decode(h)
            if y_mask is None:
                recon = F.cross_entropy(
                    logits.reshape(-1, self.cfg.vocab_size),
                    y_ids.reshape(-1),
                )
            else:
                y_for_loss = y_ids.masked_fill(~y_mask, -100)
                recon = F.cross_entropy(
                    logits.reshape(-1, self.cfg.vocab_size),
                    y_for_loss.reshape(-1),
                    ignore_index=-100,
                )
            kl_post_grad  = kl_gaussian(mu_q, lv_q,
                                        mu_p.detach(), lv_p.detach()).mean()
            kl_prior_grad = kl_gaussian(mu_q.detach(), lv_q.detach(),
                                        mu_p, lv_p).mean()
            kl = kl_balance * kl_prior_grad + (1.0 - kl_balance) * kl_post_grad

            recon_sum = recon_sum + recon
            kl_sum    = kl_sum    + kl

            if self.cfg.use_halt:
                # target: per-example correctness rate over valid positions
                with torch.no_grad():
                    correct = (logits.argmax(-1) == y_ids)
                    if y_mask is None:
                        target = correct.float().mean(dim=-1)
                    else:
                        target = (correct & y_mask).float().sum(-1) / n_valid
                halt_logits = self.halt_head(h)                  # (B, 2)
                halt = F.binary_cross_entropy_with_logits(halt_logits[:, 0], target)
                halt_sum = halt_sum + halt

            with torch.no_grad():
                correct = (logits.argmax(-1) == y_ids)
                if y_mask is None:
                    acc_sum += correct.float().mean().item()
                else:
                    acc_sum += ((correct & y_mask).float().sum()
                                / y_mask.float().sum().clamp(min=1)).item()

            h = h.detach()
            l = l.detach()

        recon_avg = recon_sum / self.cfg.N_sup
        kl_avg    = kl_sum    / self.cfg.N_sup
        elbo_loss = recon_avg + beta * kl_avg

        halt_avg = (halt_sum / self.cfg.N_sup) if self.cfg.use_halt else torch.tensor(0.0, device=elbo_loss.device)

        # LPRM training: separate prior-only trajectory (no_grad through it),
        # value head MSE against per-example correctness rate.
        h_prior_T = self._run_prior_trajectory(e_x)
        with torch.no_grad():
            logits_p = self.decode(h_prior_T)
            correct_p = (logits_p.argmax(-1) == y_ids)
            if y_mask is None:
                r = correct_p.float().mean(dim=-1)
            else:
                r = (correct_p & y_mask).float().sum(-1) / n_valid
        score = self.value_head(h_prior_T)
        lprm_loss = F.mse_loss(score, r)

        loss = elbo_loss + lprm_weight * lprm_loss
        if self.cfg.use_halt:
            loss = loss + halt_weight * halt_avg

        return loss, {
            "loss":  loss.item(),
            "recon": recon_avg.item(),
            "kl":    kl_avg.item(),
            "kl_true": 0.0,    # legacy path; segment-level training reports real value
            "mu_p_std": 0.0,
            "mu_q_std": 0.0,
            "lprm":  lprm_loss.item(),
            "halt":  halt_avg.item() if self.cfg.use_halt else 0.0,
            "r":     r.mean().item(),
            "acc":   acc_sum / self.cfg.N_sup,
        }
