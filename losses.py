"""
losses.py
Loss functions used for training the surrogate models.

MVE loss  — Gaussian negative log-likelihood that jointly learns mean + variance.
Spearman  — Differentiable soft-rank Spearman correlation loss.
Combined  — MVE + λ·Spearman, the loss used for finetuning in the paper.
"""

import torch
import torch.nn as nn


# ── MVE (Mean Variance Estimation) ────────────────────────────────────────────

def mve_loss(mu: torch.Tensor, var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Gaussian NLL loss over a batch of predictions.

    L = (1/N) Σᵢ ½ [log(σ²ᵢ) + (μᵢ − yᵢ)² / σ²ᵢ]

    Parameters
    ----------
    mu  : predicted means,     shape (N,)
    var : predicted variances, shape (N,) — must be positive (use softplus)
    y   : ground-truth labels, shape (N,)
    """
    return 0.5 * (var.log() + (mu - y).pow(2) / var).mean()


# ── Soft Spearman ─────────────────────────────────────────────────────────────

def _soft_rank(y: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    Differentiable soft rank (Engilberge et al., 2019).

    softRank(yᵢ) = 1 + Σⱼ≠ᵢ σ((yⱼ − yᵢ) / τ)

    Higher y → rank ≈ 1 (fewest elements above it).
    Lower  y → rank ≈ N.

    Parameters
    ----------
    y   : 1-D tensor of values, shape (N,)
    tau : temperature — smaller τ → harder (closer to true rank)
    """
    # diff[i, j] = (y[j] - y[i]) / tau
    diff = (y.unsqueeze(0) - y.unsqueeze(1)) / tau   # [N, N]
    sig  = torch.sigmoid(diff)                        # ≈ 1 if y[j] > y[i]
    mask = 1.0 - torch.eye(len(y), device=y.device)
    return 1.0 + (sig * mask).sum(dim=1)              # [N]


def spearman_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    tau: float  = 1.0,
    eps: float  = 1e-8,
) -> torch.Tensor:
    """
    1 − ρ_soft, where ρ_soft is the soft Spearman rank correlation.

    L_Spearman = 1 − [Σᵢ (r̃ᵢᵖʳᵉᵈ · r̃ᵢᵗʳᵘᵉ)] /
                     [√Σᵢ(r̃ᵢᵖʳᵉᵈ)² · √Σᵢ(r̃ᵢᵗʳᵘᵉ)² + ε]

    where r̃ᵢ = rᵢ − r̄  (mean-centred soft ranks).

    Works whether docking scores are negated (higher = better) or not —
    just be consistent with y_pred and y_true signs.

    Parameters
    ----------
    y_pred : model predictions,    shape (N,)
    y_true : ground-truth targets, shape (N,)
    tau    : soft-rank temperature
    eps    : numerical stability constant
    """
    r_pred = _soft_rank(y_pred, tau=tau)
    r_true = _soft_rank(y_true, tau=tau)

    r_pred_c = r_pred - r_pred.mean()
    r_true_c = r_true - r_true.mean()

    rho = (r_pred_c * r_true_c).sum() / (
        r_pred_c.pow(2).sum().sqrt() * r_true_c.pow(2).sum().sqrt() + eps
    )
    return 1.0 - rho


# ── Combined loss ──────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    MVE loss + λ · Spearman loss — the loss used for surrogate finetuning.

    Parameters
    ----------
    spearman_weight : λ, relative weight of the Spearman term (default 0.1)
    tau             : soft-rank temperature (default 1.0)
    """

    def __init__(self, spearman_weight: float = 0.1, tau: float = 1.0):
        super().__init__()
        self.spearman_weight = spearman_weight
        self.tau = tau

    def forward(
        self,
        mu:  torch.Tensor,
        var: torch.Tensor,
        y:   torch.Tensor,
    ) -> torch.Tensor:
        l_mve      = mve_loss(mu, var, y)
        l_spearman = spearman_loss(mu, y, tau=self.tau)
        return l_mve + self.spearman_weight * l_spearman
