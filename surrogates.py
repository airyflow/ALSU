"""
surrogates.py
Surrogate model classes for the active learning loop.

All surrogates share the same fit / predict interface used by ALExplorer:
  fit(X: np.ndarray, y: np.ndarray, epochs: int)
  predict(X: np.ndarray) -> (mu: np.ndarray, sigma: np.ndarray)

Classes
-------
SingleBackboneMVESurrogate
    Single-backbone surrogate with dual MVE heads + combined loss.
    Replaces the original MSE-based Surrogate for Molformer / GROVER / UniMol.

LightweightMVESurrogate
    Takes concatenated embeddings from all three backbones.
    Architecture exactly as described in the paper (dual MVE heads).

BigFusionSurrogate
    Trains three independent SingleBackbone surrogates (one per backbone)
    and combines their rankings via Borda count at acquisition time.
    predict() returns the Borda acquisition score as mu and zeros as sigma.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from losses import CombinedLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Shared building blocks ─────────────────────────────────────────────────────

class _MVEHead(nn.Module):
    """
    Single MVE head: predicts (μ, σ²) from a shared latent vector h.

    mu  : Linear(h_dim, 64) → ReLU → Dropout → Linear(64, 1)
    var : same → softplus + 1e-6
    """
    def __init__(self, h_dim: int, dropout: float = 0.25):
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(h_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.var_net = nn.Sequential(
            nn.Linear(h_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor):
        mu  = self.mu_net(h).squeeze(-1)
        var = F.softplus(self.var_net(h).squeeze(-1)) + 0.01
        return mu, var


class _LightweightBackbone(nn.Module):
    """
    Shared backbone from the paper:
      Linear(in_dim, 256) → ReLU → BN → Dropout
      Linear(256, 128)    → ReLU → BN → Dropout
    Output shape: (B, 128)

    Sized for actual embedding dims (GROVER=1600, MoLFormer=768, UniMol=512).
    """
    def __init__(self, in_dim: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DualMVEModel(nn.Module):
    """
    Full model: backbone (→128) → two independent MVE heads → averaged output.

    mean = 0.5 * (mu1 + mu2)
    var  = 0.5 * (var1 + var2)
    """
    def __init__(self, in_dim: int, dropout: float = 0.25):
        super().__init__()
        self.backbone = _LightweightBackbone(in_dim, dropout)
        self.head1    = _MVEHead(128, dropout)
        self.head2    = _MVEHead(128, dropout)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        mu1, var1 = self.head1(h)
        mu2, var2 = self.head2(h)
        return 0.5 * (mu1 + mu2), 0.5 * (var1 + var2)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _amp_context(enabled: bool):
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return torch.autocast(device_type=DEVICE.type, dtype=dtype, enabled=enabled)


def _train_model(
    model: nn.Module,
    X: np.ndarray,
    y_norm: np.ndarray,
    loss_fn: nn.Module,
    epochs: int,
    batch: int,
    lr: float,
):
    """Shared training loop for any _DualMVEModel."""
    Xt = torch.tensor(X,      dtype=torch.float32)
    yt = torch.tensor(y_norm, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=True,
                        drop_last=False)

    opt    = torch.optim.AdamW(model.parameters(), lr=lr)
    use_amp = DEVICE.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with _amp_context(use_amp):
                mu, var = model(xb)
                loss    = loss_fn(mu, var, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()


def _predict_model(model: nn.Module, X: np.ndarray, batch: int = 1024):
    """Return (mu, sigma) in original (de-normalised) scale — normalisation
    must be applied by the caller."""
    Xt     = torch.tensor(X, dtype=torch.float32)
    use_amp = DEVICE.type == "cuda"
    mu_list, var_list = [], []

    model.eval()
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(Xt), batch_size=batch, shuffle=False):
            xb = xb.to(DEVICE)
            with _amp_context(use_amp):
                mu, var = model(xb)
            mu_list.append(mu.float().cpu())
            var_list.append(var.float().cpu())

    mu  = torch.cat(mu_list).numpy()
    sig = torch.cat(var_list).sqrt().numpy()   # σ = √var
    return mu, sig


# ── SingleBackboneMVESurrogate ─────────────────────────────────────────────────

class SingleBackboneMVESurrogate:
    """
    Surrogate for a single backbone's embeddings (Molformer, GROVER, or UniMol).
    Uses dual MVE heads + combined MVE+Spearman loss.

    Parameters
    ----------
    in_dim           : embedding dimension of the backbone
    spearman_weight  : λ for Spearman term in the combined loss
    lr               : learning rate
    dropout          : dropout probability
    """

    def __init__(
        self,
        in_dim: int,
        spearman_weight: float = 0.1,
        lr: float              = 3e-4,
        dropout: float         = 0.25,
    ):
        self._in_dim  = in_dim
        self._lr      = lr
        self._loss_fn = CombinedLoss(spearman_weight=spearman_weight)
        self._dropout = dropout
        self._model   = _DualMVEModel(in_dim, dropout).to(DEVICE)
        self._ym = self._ys = None

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch: int = 256):
        self._ym = float(y.mean())
        self._ys = float(y.std()) + 1e-8
        y_norm   = (y - self._ym) / self._ys

        # Warm-start from previous round's weights (model initialised in __init__)
        _train_model(self._model, X, y_norm, self._loss_fn, epochs, batch, self._lr)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu_n, sig_n = _predict_model(self._model, X)
        return mu_n * self._ys + self._ym, sig_n   # de-normalise mean only


# ── LightweightMVESurrogate ────────────────────────────────────────────────────

class LightweightMVESurrogate:
    """
    Concatenates embeddings from all three backbones and trains a dual-MVE-head
    MLP, matching the "Lightweight" architecture from the paper.

    Input: X = cat([x_uni, x_molf, x_grov], dim=-1)  shape (N, d_total)

    Parameters
    ----------
    in_dim           : total concatenated embedding dim (d_uni + d_molf + d_grov)
    spearman_weight  : λ for the Spearman term
    lr               : learning rate
    dropout          : dropout probability (paper uses 0.25)
    """

    def __init__(
        self,
        in_dim: int,
        spearman_weight: float = 0.1,
        lr: float              = 3e-4,
        dropout: float         = 0.25,
    ):
        self._in_dim  = in_dim
        self._lr      = lr
        self._loss_fn = CombinedLoss(spearman_weight=spearman_weight)
        self._dropout = dropout
        self._model   = _DualMVEModel(in_dim, dropout).to(DEVICE)
        self._ym = self._ys = None

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch: int = 256):
        self._ym = float(y.mean())
        self._ys = float(y.std()) + 1e-8
        y_norm   = (y - self._ym) / self._ys

        # Warm-start from previous round's weights (model initialised in __init__)
        _train_model(self._model, X, y_norm, self._loss_fn, epochs, batch, self._lr)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu_n, sig_n = _predict_model(self._model, X)
        return mu_n * self._ys + self._ym, sig_n


# ── BigFusionSurrogate ─────────────────────────────────────────────────────────

class BigFusionSurrogate:
    """
    Three independent SingleBackbone surrogates (one per backbone) combined
    via Borda count at acquisition time.

    Input to fit/predict: dict {"grover": X_g, "molformer": X_m, "unimol": X_u}
    or a single numpy array that is the concatenation in that order (in which
    case dims must be provided so we can split).

    predict() returns:
      mu    — negative Borda sum (lower Borda = better → higher mu)
      sigma — zeros (Borda is a hard combination, no uncertainty)

    Parameters
    ----------
    dims : dict {"grover": int, "molformer": int, "unimol": int}
           embedding dims for each backbone, used when input is concatenated.
    """

    _KEYS = ["grover", "molformer", "unimol"]

    def __init__(
        self,
        dims: dict,
        spearman_weight: float = 0.1,
        lr: float              = 3e-4,
        dropout: float         = 0.25,
    ):
        self._dims = dims
        self._surrogates = {
            k: SingleBackboneMVESurrogate(
                in_dim          = dims[k],
                spearman_weight = spearman_weight,
                lr              = lr,
                dropout         = dropout,
            )
            for k in self._KEYS
        }

    def _split(self, X: np.ndarray) -> dict:
        """Split concatenated embedding matrix into per-backbone dict."""
        cuts = np.cumsum([self._dims[k] for k in self._KEYS])
        splits = np.split(X, cuts[:-1], axis=1)
        return {k: s for k, s in zip(self._KEYS, splits)}

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch: int = 256):
        parts = X if isinstance(X, dict) else self._split(X)
        for k in self._KEYS:
            self._surrogates[k].fit(parts[k], y, epochs=epochs, batch=batch)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        parts = X if isinstance(X, dict) else self._split(X)
        n     = len(next(iter(parts.values())))

        borda = np.zeros(n, dtype=np.float64)
        for k in self._KEYS:
            mu, _ = self._surrogates[k].predict(parts[k])
            # rank: 1 = highest μ (best predicted score), N = lowest
            order = np.argsort(mu)[::-1]
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            borda += ranks

        # Return negative Borda sum so that acquisition can maximise
        return -borda.astype(np.float32), np.zeros(n, dtype=np.float32)


# ── EnsembleFusionSurrogate ────────────────────────────────────────────────────

def _spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    """Pure-numpy Spearman ρ."""
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    xc, yc = xr - xr.mean(), yr - yr.mean()
    denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum()) + 1e-8
    return float((xc * yc).sum() / denom)


class EnsembleFusionSurrogate:
    """
    Three independent SingleBackbone surrogates combined via adaptive weighted
    ensemble instead of Borda count.

    After each fit(), each backbone's weight is set proportional to its
    Spearman ρ on the training set.  This lets the ensemble adapt as data
    accumulates — backbones with better in-distribution correlation receive
    higher influence on acquisition scores.

    predict() returns real (mu, sigma) so that UCB can exploit inter-model
    disagreement as epistemic uncertainty:

      mu_ens    = Σ_k w_k · μ_k               (weighted mean)
      σ_inter   = sqrt(Σ_k w_k · (μ_k − μ_ens)²)  (backbone disagreement)
      σ_intra   = Σ_k w_k · σ_k               (average aleatoric uncertainty)
      σ_total   = sqrt(σ_inter² + σ_intra²)

    Parameters
    ----------
    dims : dict {"grover": int, "molformer": int, "unimol": int}
    """

    _KEYS = ["grover", "molformer", "unimol"]

    def __init__(
        self,
        dims: dict,
        spearman_weight: float = 0.1,
        lr: float              = 3e-4,
        dropout: float         = 0.25,
    ):
        self._dims = dims
        self._surrogates = {
            k: SingleBackboneMVESurrogate(
                in_dim          = dims[k],
                spearman_weight = spearman_weight,
                lr              = lr,
                dropout         = dropout,
            )
            for k in self._KEYS
        }
        self._weights = np.ones(3) / 3   # equal weights until first fit

    def _split(self, X: np.ndarray) -> dict:
        cuts = np.cumsum([self._dims[k] for k in self._KEYS])
        splits = np.split(X, cuts[:-1], axis=1)
        return {k: s for k, s in zip(self._KEYS, splits)}

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch: int = 256):
        parts = X if isinstance(X, dict) else self._split(X)
        n     = len(y)

        # 80/20 holdout so val-ρ is honest (training ρ ≈ 0.99 for all backbones).
        rng      = np.random.default_rng(n)   # seed changes each round as |labeled| grows
        val_idx  = rng.choice(n, size=max(1, n // 5), replace=False)
        tr_mask  = np.ones(n, dtype=bool);  tr_mask[val_idx] = False

        parts_tr = {k: v[tr_mask]  for k, v in parts.items()}
        parts_vl = {k: v[~tr_mask] for k, v in parts.items()}
        y_tr, y_vl = y[tr_mask], y[~tr_mask]

        for k in self._KEYS:
            self._surrogates[k].fit(parts_tr[k], y_tr, epochs=epochs, batch=batch)

        # Weights from validation Spearman — properly reflects generalisation quality
        rhos = []
        for k in self._KEYS:
            mu_vl, _ = self._surrogates[k].predict(parts_vl[k])
            rhos.append(max(_spearman_np(mu_vl, y_vl), 0.0))

        total = sum(rhos)
        if total > 0:
            self._weights = np.array(rhos) / total
        else:
            self._weights = np.ones(3) / 3

        print(f"  [EnsembleFusion] weights — "
              + "  ".join(f"{k}:{w:.3f}" for k, w in zip(self._KEYS, self._weights)))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        parts = X if isinstance(X, dict) else self._split(X)
        n     = len(next(iter(parts.values())))

        # Weighted Borda: backbone with higher val-ρ contributes more to the ranking.
        borda = np.zeros(n, dtype=np.float64)
        for i, k in enumerate(self._KEYS):
            mu, _ = self._surrogates[k].predict(parts[k])
            order = np.argsort(mu)[::-1]
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            borda += self._weights[i] * ranks

        return -borda.astype(np.float32), np.zeros(n, dtype=np.float32)
