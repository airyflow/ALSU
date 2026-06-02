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
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

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
        spearman_weight: float  = 0.1,
        lr: float               = 3e-4,
        dropout: float          = 0.25,
        weights: dict | None    = None,
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
        # Normalise weights so they sum to 1; default = uniform
        if weights is None:
            self._w = {k: 1 / len(self._KEYS) for k in self._KEYS}
        else:
            total = sum(weights[k] for k in self._KEYS)
            self._w = {k: weights[k] / total for k in self._KEYS}

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
            order = np.argsort(mu)[::-1]
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            borda += self._w[k] * ranks   # weighted Borda contribution

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
    Three independent SingleBackbone surrogates combined via UCB on normalised
    Borda ranks.  Unlike BigFusion's greedy Borda, acquisition uses inter-model
    rank disagreement as epistemic uncertainty, allowing exploration of molecules
    where backbones disagree (potential high-scorers missed by any single backbone).

    UCB_score = -mean_rank_norm + beta * std_rank_norm
    predict() bakes UCB into mu and returns zeros for sigma — use with acq_greedy.

    sigma is rank std across backbones (∈ [0, ~0.4]).  beta=0.2 is calibrated so
    exploration contributes ~20% of the signal for the top candidates.

    Parameters
    ----------
    dims : dict {"grover": int, "molformer": int, "unimol": int}
    beta : exploration weight (default 0.2)
    """

    _KEYS = ["grover", "molformer", "unimol"]

    def __init__(
        self,
        dims: dict,
        spearman_weight: float = 0.1,
        lr: float              = 3e-4,
        dropout: float         = 0.25,
        beta: float            = 0.2,
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
        self._beta = beta

    def _split(self, X: np.ndarray) -> dict:
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
            order = np.argsort(mu)[::-1]
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(1, n + 1)
            borda += ranks

        return -borda.astype(np.float32), np.zeros(n, dtype=np.float32)


# ── LearnedFusionSurrogate ─────────────────────────────────────────────────────

class LearnedFusionSurrogate:
    """
    Three independent SingleBackbone surrogates whose predictions are combined
    by a RidgeCV linear meta-learner trained on held-out labeled data.

    Unlike Borda count (rank-based, scale-invariant), the meta-learner combines
    raw predicted scores: mu_meta = w_g*mu_g + w_m*mu_m + w_u*mu_u + bias.
    This corrects for backbone-specific scale biases and learns the relative
    contribution of each backbone from data rather than from heuristics.

    fit() workflow:
      1. 80/20 split of labeled data.
      2. Train all three backbone surrogates on the 80% training split.
      3. Predict on the 20% holdout → honest (non-overfitted) backbone µ values.
      4. Fit RidgeCV meta-learner on [µ_g, µ_m, µ_u] → y_holdout.
      Backbone models stay fitted on the 80% split (not retrained on full data).

    predict() returns (mu_meta, zeros) — use with acq_greedy.

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
        self._meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        self._meta_fitted = False

    def _split(self, X: np.ndarray) -> dict:
        cuts = np.cumsum([self._dims[k] for k in self._KEYS])
        splits = np.split(X, cuts[:-1], axis=1)
        return {k: s for k, s in zip(self._KEYS, splits)}

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch: int = 256):
        parts = X if isinstance(X, dict) else self._split(X)
        n     = len(y)

        # 80/20 holdout — meta-learner must see out-of-sample backbone predictions
        rng     = np.random.default_rng(n)
        val_idx = rng.choice(n, size=max(1, n // 5), replace=False)
        tr_mask = np.ones(n, dtype=bool);  tr_mask[val_idx] = False

        parts_tr = {k: v[tr_mask]  for k, v in parts.items()}
        parts_vl = {k: v[~tr_mask] for k, v in parts.items()}
        y_tr, y_vl = y[tr_mask], y[~tr_mask]

        for k in self._KEYS:
            self._surrogates[k].fit(parts_tr[k], y_tr, epochs=epochs, batch=batch)

        # Collect holdout predictions → feature matrix for meta-learner
        val_mus = np.stack(
            [self._surrogates[k].predict(parts_vl[k])[0] for k in self._KEYS],
            axis=1,
        )  # (n_val, 3)
        self._meta.fit(val_mus, y_vl)
        self._meta_fitted = True

        coef_str = "  ".join(f"{k}:{c:.3f}" for k, c in
                              zip(self._KEYS, self._meta.coef_))
        print(f"  [LearnedFusion] meta coef — {coef_str}  "
              f"bias:{self._meta.intercept_:.3f}  α={self._meta.alpha_:.3g}")

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        parts = X if isinstance(X, dict) else self._split(X)
        n     = len(next(iter(parts.values())))

        mus = np.stack(
            [self._surrogates[k].predict(parts[k])[0] for k in self._KEYS],
            axis=1,
        )  # (N, 3)

        if self._meta_fitted:
            mu_out = self._meta.predict(mus).astype(np.float32)
        else:
            mu_out = mus.mean(axis=1).astype(np.float32)

        return mu_out, np.zeros(n, dtype=np.float32)


# ── NonlinearFusionSurrogate ───────────────────────────────────────────────────

class _FusionMLP(nn.Module):
    """
    Tiny nonlinear fusion head: 6 → 32 → 16 → 1 with GELU.
    Input: [µ_g, σ_g, µ_m, σ_m, µ_u, σ_u]
    """
    def __init__(self, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NonlinearFusionSurrogate:
    """
    Three backbone surrogates + tiny MLP meta-head trained on held-out data.

    The MLP receives [µ_g, σ_g, µ_m, σ_m, µ_u, σ_u] — six features — so it
    can learn conditional patterns like "trust UniMol more when MoLFormer's
    σ is high" or "down-weight GROVER when its µ contradicts MoLFormer."
    A linear meta-learner (LearnedFusion) cannot express these interactions.

    fit() workflow (same 80/20 split as LearnedFusion):
      1. Train backbone surrogates on 80% of labeled data.
      2. Collect (µ, σ) from all three backbones on 20% holdout.
      3. Train _FusionMLP on those 6-feature vectors → y_holdout.

    predict() returns (µ_meta, zeros) — use with acq_greedy.

    Parameters
    ----------
    dims       : dict {"grover": int, "molformer": int, "unimol": int}
    meta_epochs: training epochs for the fusion MLP (default 500)
    meta_lr    : learning rate for fusion MLP (default 3e-3)
    """

    _KEYS = ["grover", "molformer", "unimol"]

    def __init__(
        self,
        dims: dict,
        spearman_weight: float = 0.1,
        lr: float              = 3e-4,
        dropout: float         = 0.25,
        meta_epochs: int       = 500,
        meta_lr: float         = 3e-3,
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
        self._meta_epochs = meta_epochs
        self._meta_lr     = meta_lr
        self._mlp: _FusionMLP | None = None

    def _split(self, X: np.ndarray) -> dict:
        cuts = np.cumsum([self._dims[k] for k in self._KEYS])
        splits = np.split(X, cuts[:-1], axis=1)
        return {k: s for k, s in zip(self._KEYS, splits)}

    def _backbone_features(self, parts: dict) -> np.ndarray:
        """Stack [µ_g, σ_g, µ_m, σ_m, µ_u, σ_u] → (N, 6) float32."""
        cols = []
        for k in self._KEYS:
            mu, sig = self._surrogates[k].predict(parts[k])
            cols.extend([mu, sig])
        return np.stack(cols, axis=1).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch: int = 256):
        parts = X if isinstance(X, dict) else self._split(X)
        n     = len(y)

        # 80/20 holdout — MLP must see out-of-sample backbone predictions
        rng     = np.random.default_rng(n)
        val_idx = rng.choice(n, size=max(1, n // 5), replace=False)
        tr_mask = np.ones(n, dtype=bool);  tr_mask[val_idx] = False

        parts_tr = {k: v[tr_mask]  for k, v in parts.items()}
        parts_vl = {k: v[~tr_mask] for k, v in parts.items()}
        y_tr, y_vl = y[tr_mask], y[~tr_mask]

        for k in self._KEYS:
            self._surrogates[k].fit(parts_tr[k], y_tr, epochs=epochs, batch=batch)

        # 6-feature matrix from holdout backbone predictions
        X_vl = self._backbone_features(parts_vl)            # (n_val, 6)
        Xt   = torch.tensor(X_vl).to(DEVICE)
        yt   = torch.tensor(y_vl, dtype=torch.float32).to(DEVICE)

        # Fresh MLP each round (small dataset, fast to train from scratch)
        self._mlp = _FusionMLP().to(DEVICE)
        opt = torch.optim.Adam(
            self._mlp.parameters(), lr=self._meta_lr, weight_decay=1e-3
        )

        self._mlp.train()
        for _ in range(self._meta_epochs):
            opt.zero_grad()
            F.mse_loss(self._mlp(Xt), yt).backward()
            opt.step()

        self._mlp.eval()
        with torch.no_grad():
            val_rho = _spearman_np(self._mlp(Xt).cpu().numpy(), y_vl)
        print(f"  [NonlinearFusion] val Spearman ρ = {val_rho:.3f}  (n_val={len(y_vl)})")

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        parts = X if isinstance(X, dict) else self._split(X)
        n     = len(next(iter(parts.values())))

        if self._mlp is None:
            return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

        feats = self._backbone_features(parts)               # (N, 6)
        self._mlp.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, n, 4096):
                xb = torch.tensor(feats[i : i + 4096]).to(DEVICE)
                preds.append(self._mlp(xb).float().cpu().numpy())

        return np.concatenate(preds), np.zeros(n, dtype=np.float32)


# ── OOFFusionSurrogate ─────────────────────────────────────────────────────────

class OOFFusionSurrogate:
    """
    K-fold out-of-fold (OOF) stacking: every labeled molecule gets an honest
    out-of-sample backbone prediction, the RidgeCV meta-learner is fitted on
    all N OOF predictions, then the main surrogates are retrained on 100% of
    the labeled data.

    Unlike LearnedFusion / NonlinearFusion (80/20 holdout), no labels are
    sacrificed — the backbone surrogates always see the full labeled set.

    fit() workflow:
      1. K-fold split of labeled data.
      2. For each fold: train fresh backbone surrogates on K-1 folds,
         predict µ on the held-out fold → collect OOF predictions.
      3. Fit RidgeCV on all N OOF [µ_g, µ_m, µ_u] → y.
      4. Retrain main surrogates (warm-start) on 100% of labeled data.

    Fold surrogates are trained for fold_epoch_frac × epochs (default 1/3)
    to keep runtime similar to the 80/20-holdout variants.
    Fresh initialization per fold avoids warm-start data leakage into OOF.

    predict() returns (µ_meta, zeros) — use with acq_greedy.

    Parameters
    ----------
    dims            : dict {"grover": int, "molformer": int, "unimol": int}
    n_folds         : number of CV folds (default 3)
    fold_epoch_frac : fraction of epochs used for fold models (default 0.33)
    """

    _KEYS = ["grover", "molformer", "unimol"]

    def __init__(
        self,
        dims: dict,
        spearman_weight: float  = 0.1,
        lr: float               = 3e-4,
        dropout: float          = 0.25,
        n_folds: int            = 3,
        fold_epoch_frac: float  = 0.33,
    ):
        self._dims            = dims
        self._spearman_weight = spearman_weight
        self._lr              = lr
        self._dropout         = dropout
        self._n_folds         = n_folds
        self._fold_epoch_frac = fold_epoch_frac
        self._surrogates = {
            k: SingleBackboneMVESurrogate(
                in_dim          = dims[k],
                spearman_weight = spearman_weight,
                lr              = lr,
                dropout         = dropout,
            )
            for k in self._KEYS
        }
        self._meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        self._meta_fitted = False

    def _split(self, X: np.ndarray) -> dict:
        cuts = np.cumsum([self._dims[k] for k in self._KEYS])
        splits = np.split(X, cuts[:-1], axis=1)
        return {k: s for k, s in zip(self._KEYS, splits)}

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch: int = 256):
        parts = X if isinstance(X, dict) else self._split(X)
        n     = len(y)
        fold_epochs = max(20, int(epochs * self._fold_epoch_frac))

        # ── Step 1: K-fold OOF backbone predictions ───────────────────────────
        oof_mus = np.zeros((n, 3), dtype=np.float32)
        kf = KFold(n_splits=self._n_folds, shuffle=True, random_state=n)

        for fold_tr, fold_vl in kf.split(np.arange(n)):
            parts_tr = {k: v[fold_tr] for k, v in parts.items()}
            parts_vl = {k: v[fold_vl] for k, v in parts.items()}
            y_tr     = y[fold_tr]

            # Fresh surrogates per fold — no warm-start to avoid data leakage
            fold_surrs = {
                k: SingleBackboneMVESurrogate(
                    in_dim          = self._dims[k],
                    spearman_weight = self._spearman_weight,
                    lr              = self._lr,
                    dropout         = self._dropout,
                )
                for k in self._KEYS
            }
            for k in self._KEYS:
                fold_surrs[k].fit(parts_tr[k], y_tr, epochs=fold_epochs, batch=batch)
            for i, k in enumerate(self._KEYS):
                oof_mus[fold_vl, i] = fold_surrs[k].predict(parts_vl[k])[0]

            # Explicitly free GPU memory before next fold
            for s in fold_surrs.values():
                del s._model
            del fold_surrs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Step 2: Meta-learner on all OOF predictions ───────────────────────
        self._meta.fit(oof_mus, y)
        self._meta_fitted = True

        oof_rho  = _spearman_np(self._meta.predict(oof_mus), y)
        coef_str = "  ".join(f"{k}:{c:.3f}" for k, c in
                              zip(self._KEYS, self._meta.coef_))
        print(f"  [OOFFusion] coef — {coef_str}  "
              f"bias:{self._meta.intercept_:.3f}  OOF ρ={oof_rho:.3f}")

        # ── Step 3: Retrain main surrogates on 100% data (warm-start) ─────────
        for k in self._KEYS:
            self._surrogates[k].fit(parts[k], y, epochs=epochs, batch=batch)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        parts = X if isinstance(X, dict) else self._split(X)
        n     = len(next(iter(parts.values())))

        mus = np.stack(
            [self._surrogates[k].predict(parts[k])[0] for k in self._KEYS], axis=1
        )  # (N, 3)

        if self._meta_fitted:
            mu_out = self._meta.predict(mus).astype(np.float32)
        else:
            mu_out = mus.mean(axis=1).astype(np.float32)

        return mu_out, np.zeros(n, dtype=np.float32)


# ── LightweightMoLFormerScheduleSurrogate ─────────────────────────────────────

class LightweightMoLFormerScheduleSurrogate:
    """
    Two-phase scheduled surrogate:

    Phase 1 — first n_lt_rounds fit() calls:
        Train the Lightweight MLP head on frozen concatenated embeddings
        [GROVER_frozen | MoLFormer_frozen | UniMol_frozen].

    Phase 2 — remaining fit() calls:
        1. Finetune MoLFormer backbone on accumulated labeled data
           (BackboneFinetuner with small lr to avoid catastrophic forgetting).
        2. Re-extract MoLFormer embeddings for the full pool.
        3. Signal Experiment to rebuild its fused matrix with the new embeddings.
        4. The Lightweight MLP head stays FROZEN — no weight updates.
           Acquisition uses the frozen head's output on the improved embeddings.

    Rationale: early rounds have too few labels to finetune a large language
    model without overfitting; the frozen-embedding MLP provides a stable,
    fast surrogate.  Once enough labels accumulate, finetuning MoLFormer
    produces more task-specific embeddings that the already-calibrated MLP
    head can exploit without retraining.

    Parameters
    ----------
    dims         : {"grover": D_g, "molformer": D_m, "unimol": D_u}
    pool_smiles  : ordered list of all pool SMILES (MoLFormer re-extraction order)
    emb_dict     : mutable reference to Experiment.emb_dict; "molformer" key is
                   updated in-place after each finetune round
    dataset_name : e.g. "Enamine50k"
    n_lt_rounds  : lightweight rounds before switching to finetune (default 3)
    ft_epochs    : MoLFormer finetune epochs per round (default 10)
    ft_lr_bb     : backbone learning rate (small; default 1e-5)
    ft_lr_head   : regression head lr for finetuner (default 1e-4)
    """

    needs_smiles = True   # Experiment will pass labeled_smiles to fit()

    def __init__(
        self,
        dims: dict,
        pool_smiles: list,
        emb_dict: dict,
        dataset_name: str   = "Enamine50k",
        n_lt_rounds: int    = 3,
        ft_epochs: int      = 10,
        ft_lr_bb: float     = 1e-5,
        ft_lr_head: float   = 1e-4,
        spearman_weight: float = 0.1,
        lr: float              = 3e-4,
        dropout: float         = 0.25,
    ):
        self._dims         = dims
        self._pool_smiles  = pool_smiles
        self._emb_dict     = emb_dict
        self._dataset_name = dataset_name
        self._n_lt         = n_lt_rounds
        self._ft_epochs    = ft_epochs
        self._ft_lr_bb     = ft_lr_bb
        self._ft_lr_head   = ft_lr_head
        self._round_count  = 0

        total_dim = sum(dims[k] for k in ["grover", "molformer", "unimol"])
        self._lightweight = LightweightMVESurrogate(
            in_dim          = total_dim,
            spearman_weight = spearman_weight,
            lr              = lr,
            dropout         = dropout,
        )

        self._finetuner  = None           # lazy-init (loading MoLFormer is expensive)
        self._smi2idx    = {s: i for i, s in enumerate(pool_smiles)}
        self.embeddings_refreshed = False # Experiment checks this after fit()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch: int  = 256,
        labeled_smiles: list | None = None,
    ):
        self._round_count += 1

        if self._round_count <= self._n_lt:
            # ── Phase 1: train Lightweight MLP ────────────────────────────────
            self._lightweight.fit(X, y, epochs=epochs, batch=batch)

        else:
            # ── Phase 2: finetune MoLFormer ───────────────────────────────────
            assert labeled_smiles is not None, (
                "LightweightMoLFormerScheduleSurrogate needs labeled_smiles in phase 2"
            )

            # Load MoLFormer into the finetuner once (model stays in memory)
            if self._finetuner is None:
                from backbone_finetuner import BackboneFinetuner
                self._finetuner = BackboneFinetuner(
                    backbone     = "molformer",
                    dataset_name = self._dataset_name,
                    pool_smiles  = self._pool_smiles,
                )

            # y is _SIGN * oracle_score (positive, higher = better).
            # BackboneFinetuner normalises labels internally, so sign doesn't matter.
            self._finetuner.finetune(
                labeled_smiles = labeled_smiles,
                labeled_scores = y,
                n_epochs       = self._ft_epochs,
                batch_size     = 32,
                lr_backbone    = self._ft_lr_bb,
                lr_head        = self._ft_lr_head,
            )

            # Re-extract full-pool MoLFormer embeddings with finetuned weights
            new_molf = self._finetuner.extract_pool_embeddings(batch_size=256)
            self._emb_dict["molformer"] = new_molf   # update shared dict in-place

            print(f"  [3lt2mf] MoLFormer finetuned & re-extracted. "
                  f"shape={new_molf.shape}")

            # Rebuild X_tr with updated MoLFormer columns and retrain the MLP head.
            # The head starts from its round-3 weights (warm-start) and adapts quickly
            # to the new embedding distribution rather than discarding prior learning.
            pool_idx  = np.array([self._smi2idx[s] for s in labeled_smiles])
            new_X_tr  = np.concatenate([
                self._emb_dict["grover"][pool_idx],
                new_molf[pool_idx],
                self._emb_dict["unimol"][pool_idx],
            ], axis=1)
            self._lightweight.fit(new_X_tr, y, epochs=epochs, batch=batch)

            # Signal Experiment to rebuild X_all for diagnostics + acquisition
            self.embeddings_refreshed = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # X is a slice of the (possibly refreshed) fused matrix passed by Experiment.
        # After embeddings_refreshed is handled, X contains updated MoLFormer columns.
        return self._lightweight.predict(X)


# ── SingleBackboneFinetuneScheduleSurrogate ──────────────────────────────────────

class SingleBackboneFinetuneScheduleSurrogate:
    """
    Single-backbone surrogate with scheduled fine-tuning.

    Phase 1 — first n_ft_delay fit() calls:
        Freeze backbone, train dual MVE head on frozen embeddings.

    Phase 2 — remaining fit() calls:
        1. Finetune backbone on accumulated labeled data (BackboneFinetuner).
        2. Re-extract embeddings for the full pool.
        3. Signal Experiment to rebuild fused matrix.
        4. Train dual MVE head on new embeddings (warm-start from phase 1).

    This approach finetunes a single backbone for task-specific representations.

    Parameters
    ----------
    backbone        : "grover" | "unimol" | "molformer"
    in_dim          : embedding dimension of the backbone
    pool_smiles     : ordered list of all pool SMILES
    emb_dict        : mutable reference to Experiment.emb_dict (key: backbone name)
    dataset_name    : e.g. "Enamine50k"
    n_ft_delay      : rounds before switching to finetune (default 2)
    ft_epochs       : backbone finetune epochs per round (default 10)
    ft_lr_bb        : backbone learning rate (default 1e-5)
    ft_lr_head      : regression head lr for finetuner (default 1e-4)
    spearman_weight : λ for Spearman term in MVE head loss
    lr              : learning rate for MVE head
    dropout         : dropout probability
    """

    needs_smiles = True

    def __init__(
        self,
        backbone: str,
        in_dim: int,
        pool_smiles: list,
        emb_dict: dict,
        dataset_name: str = "Enamine50k",
        n_ft_delay: int = 2,
        ft_epochs: int = 10,
        ft_lr_bb: float = 1e-5,
        ft_lr_head: float = 1e-4,
        spearman_weight: float = 0.1,
        lr: float = 3e-4,
        dropout: float = 0.25,
    ):
        self._backbone = backbone
        self._in_dim = in_dim
        self._pool_smiles = pool_smiles
        self._emb_dict = emb_dict
        self._dataset_name = dataset_name
        self._n_ft_delay = n_ft_delay
        self._ft_epochs = ft_epochs
        self._ft_lr_bb = ft_lr_bb
        self._ft_lr_head = ft_lr_head
        self._round_count = 0

        self._mve = _DualMVEModel(in_dim, dropout)
        self._mve = self._mve.to(DEVICE)
        self._loss_fn = CombinedLoss(spearman_weight=spearman_weight)
        self._lr = lr
        self._dropout = dropout
        self._ym = self._ys = None

        self._finetuner = None
        self._smi2idx = {s: i for i, s in enumerate(pool_smiles)}
        self.embeddings_refreshed = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch: int = 256,
        labeled_smiles: list | None = None,
    ):
        self._round_count += 1
        self._ym = float(y.mean())
        self._ys = float(y.std()) + 1e-8
        y_norm = (y - self._ym) / self._ys

        if self._round_count <= self._n_ft_delay:
            # ── Phase 1: train MVE head on frozen embeddings ────────────────────
            _train_model(self._mve, X, y_norm, self._loss_fn, epochs, batch, self._lr)

        else:
            # ── Phase 2: finetune backbone ─────────────────────────────────────
            assert labeled_smiles is not None, (
                f"SingleBackboneFinetuneScheduleSurrogate({self._backbone}) "
                "needs labeled_smiles in phase 2"
            )

            # Load finetuner once
            if self._finetuner is None:
                from backbone_finetuner import BackboneFinetuner
                self._finetuner = BackboneFinetuner(
                    backbone=self._backbone,
                    dataset_name=self._dataset_name,
                    pool_smiles=self._pool_smiles,
                )

            # Finetune backbone + regression head
            self._finetuner.finetune(
                labeled_smiles=labeled_smiles,
                labeled_scores=y,
                n_epochs=self._ft_epochs,
                batch_size=32,
                lr_backbone=self._ft_lr_bb,
                lr_head=self._ft_lr_head,
            )

            # Re-extract embeddings
            new_emb = self._finetuner.extract_pool_embeddings(batch_size=256)
            self._emb_dict[self._backbone] = new_emb

            print(
                f"  [{self._backbone}_finetune] backbone finetuned & "
                f"re-extracted. shape={new_emb.shape}"
            )

            # Rebuild X_tr and retrain MVE head (warm-start)
            pool_idx = np.array([self._smi2idx[s] for s in labeled_smiles])
            new_X_tr = new_emb[pool_idx]
            _train_model(self._mve, new_X_tr, y_norm, self._loss_fn, epochs, batch, self._lr)

            # Signal Experiment to rebuild fused matrix
            self.embeddings_refreshed = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu_n, sig_n = _predict_model(self._mve, X)
        return mu_n * self._ys + self._ym, sig_n


# ── FTFusionSurrogate ─────────────────────────────────────────────────────────

class FTFusionSurrogate:
    """
    Multi-phase all-backbone fine-tuning fusion.

    Phase 1 — first n_ft_delay fit() calls:
        Freeze all backbones, train Lightweight MVE head on concatenated frozen embeddings.

    Phase 2 — remaining fit() calls:
        1. Finetune all three backbones sequentially on accumulated labeled data.
        2. Re-extract embeddings for the full pool.
        3. Signal Experiment to rebuild fused matrix.
        4. Retrain Lightweight MVE head on new concatenated embeddings (warm-start).

    This approach finetunes all backbones together for a unified task-specific
    representation, unlike BigFusion which keeps them independent.

    Parameters
    ----------
    dims            : {"grover": D_g, "molformer": D_m, "unimol": D_u}
    pool_smiles     : ordered list of all pool SMILES
    emb_dict        : mutable reference to Experiment.emb_dict
    dataset_name    : e.g. "Enamine50k"
    n_ft_delay      : rounds before switching to finetune (default 2)
    ft_epochs       : backbone finetune epochs per round (default 8)
    ft_lr_bb        : backbone learning rate (default 1e-5)
    ft_lr_head      : regression head lr for finetuner (default 1e-4)
    spearman_weight : λ for Spearman term in MVE head loss
    lr              : learning rate for MVE head
    dropout         : dropout probability
    """

    needs_smiles = True

    def __init__(
        self,
        dims: dict,
        pool_smiles: list,
        emb_dict: dict,
        dataset_name: str = "Enamine50k",
        n_ft_delay: int = 2,
        ft_epochs: int = 8,
        ft_lr_bb: float = 1e-5,
        ft_lr_head: float = 1e-4,
        spearman_weight: float = 0.1,
        lr: float = 3e-4,
        dropout: float = 0.25,
    ):
        self._dims = dims
        self._pool_smiles = pool_smiles
        self._emb_dict = emb_dict
        self._dataset_name = dataset_name
        self._n_ft_delay = n_ft_delay
        self._ft_epochs = ft_epochs
        self._ft_lr_bb = ft_lr_bb
        self._ft_lr_head = ft_lr_head
        self._round_count = 0

        total_dim = sum(dims[k] for k in ["grover", "molformer", "unimol"])
        self._lightweight = _DualMVEModel(total_dim, dropout)
        self._lightweight = self._lightweight.to(DEVICE)
        self._loss_fn = CombinedLoss(spearman_weight=spearman_weight)
        self._lr = lr
        self._ym = self._ys = None

        self._finetuners = {}  # {"grover": ft, "molformer": ft, "unimol": ft}
        self._smi2idx = {s: i for i, s in enumerate(pool_smiles)}
        self.embeddings_refreshed = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch: int = 256,
        labeled_smiles: list | None = None,
    ):
        self._round_count += 1
        self._ym = float(y.mean())
        self._ys = float(y.std()) + 1e-8
        y_norm = (y - self._ym) / self._ys

        if self._round_count <= self._n_ft_delay:
            # ── Phase 1: train MVE head on frozen embeddings ────────────────────
            _train_model(self._lightweight, X, y_norm, self._loss_fn, epochs, batch, self._lr)

        else:
            # ── Phase 2: finetune all three backbones ──────────────────────────
            assert labeled_smiles is not None, (
                "FTFusionSurrogate needs labeled_smiles in phase 2"
            )

            from backbone_finetuner import BackboneFinetuner

            # Lazy-init finetuners
            for backbone in ["grover", "molformer", "unimol"]:
                if backbone not in self._finetuners:
                    self._finetuners[backbone] = BackboneFinetuner(
                        backbone=backbone,
                        dataset_name=self._dataset_name,
                        pool_smiles=self._pool_smiles,
                    )

            # Finetune all three backbones on the same labeled set
            print(f"  [FTFusion] finetuning all 3 backbones…")
            for backbone in ["grover", "molformer", "unimol"]:
                self._finetuners[backbone].finetune(
                    labeled_smiles=labeled_smiles,
                    labeled_scores=y,
                    n_epochs=self._ft_epochs,
                    batch_size=32,
                    lr_backbone=self._ft_lr_bb,
                    lr_head=self._ft_lr_head,
                )

                # Re-extract embeddings
                new_emb = self._finetuners[backbone].extract_pool_embeddings(batch_size=256)
                self._emb_dict[backbone] = new_emb

            print(f"  [FTFusion] all backbones re-extracted")

            # Rebuild concatenated matrix and retrain MVE head
            pool_idx = np.array([self._smi2idx[s] for s in labeled_smiles])
            new_X_tr = np.concatenate([
                self._emb_dict["grover"][pool_idx],
                self._emb_dict["molformer"][pool_idx],
                self._emb_dict["unimol"][pool_idx],
            ], axis=1)
            _train_model(self._lightweight, new_X_tr, y_norm, self._loss_fn, epochs, batch, self._lr)

            # Signal Experiment to rebuild fused matrix
            self.embeddings_refreshed = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu_n, sig_n = _predict_model(self._lightweight, X)
        return mu_n * self._ys + self._ym, sig_n
