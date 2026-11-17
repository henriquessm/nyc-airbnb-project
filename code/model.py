"""
MLP and training loop (PyTorch) for the regression on log1p(price).

    from model import MLP, train, predict, n_params

The network returns one linear output (a log-price). The loop does mini-batch
AdamW, evaluates the validation loss after every epoch, keeps the best weights
and stops when the validation loss has not improved for `patience` epochs.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    """Linear -> ReLU -> [Dropout] per hidden layer, then one linear unit.

    Hidden weights use He (Kaiming) normal init, made for ReLU: half of the units
    are zero on average, so the variance is scaled by 2 / fan_in to keep the
    activations from shrinking layer after layer. Biases start at zero.
    """

    def __init__(self, n_in: int, hidden=(256, 128), dropout: float = 0.0):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            lin = nn.Linear(d, h)
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        return self.out(self.body(x)).squeeze(-1)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _tensor(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=DEVICE)


@torch.no_grad()
def predict(model: nn.Module, X) -> np.ndarray:
    model.eval()
    return model(_tensor(X)).cpu().numpy()


def train(model: nn.Module, X_tr, y_tr, X_va, y_va, *, lr=1e-3, batch_size=256,
          weight_decay=0.0, max_epochs=300, patience=20, seed=42):
    """Mini-batch AdamW with MSE loss and early stopping on the validation loss.

    Returns a history dict with the per-epoch train and validation loss (both
    computed in eval mode on the full set, so dropout does not inflate the
    training curve) and the best epoch. The model comes back with the best
    weights loaded.
    """
    set_seed(seed)
    model.to(DEVICE)
    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt, yt, Xv, yv = _tensor(X_tr), _tensor(y_tr), _tensor(X_va), _tensor(y_va)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = len(Xt)

    hist = {"train_loss": [], "val_loss": []}
    best, wait, best_state, best_epoch = float("inf"), 0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_loss = loss_fn(model(Xt), yt).item()
            va_loss = loss_fn(model(Xv), yv).item()
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        if va_loss < best:
            best, wait, best_epoch = va_loss, 0, epoch
            best_state = copy.deepcopy(model.state_dict())  # keep the best weights
        else:
            wait += 1
            if wait == patience:
                break

    model.load_state_dict(best_state)
    hist["best_epoch"] = best_epoch
    hist["epochs_run"] = len(hist["val_loss"])
    hist["best_val_loss"] = best
    return hist


if __name__ == "__main__":
    # smoke test: the network has to fit y = sum(x) on random data
    set_seed(0)
    X = np.random.randn(2000, 5)
    y = X.sum(axis=1)
    m = MLP(5, hidden=(32,))
    h = train(m, X[:1500], y[:1500], X[1500:], y[1500:], lr=3e-3, max_epochs=200)
    mae = np.abs(predict(m, X[1500:]) - y[1500:]).mean()
    print(f"params {n_params(m)}, stopped at {h['epochs_run']}, val MAE {mae:.3f}")
    assert mae < 0.2, mae
