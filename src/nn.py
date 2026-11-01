"""Small PyTorch network on the same features, validated and retrained the same way as LightGBM."""

import logging

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import feature as F

log = logging.getLogger(__name__)

SEED = 42
HIDDEN = (128, 64)
DROPOUT = 0.1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH = 1024
MAX_EPOCHS = 60
PATIENCE = 6
VALID_YEARS = (2021, 2022, 2023)


class Prep:
    def __init__(self, X, y):
        self.center = X.median()
        self.spread = (X.quantile(0.9) - X.quantile(0.1)).replace(0, 1).fillna(1)
        self.flags = [c for c in X.columns if X[c].isna().any()]
        self.y_mean, self.y_std = y.mean(), y.std()

    def x(self, X):
        z = ((X - self.center) / self.spread).fillna(0)
        flags = X[self.flags].isna().astype(float).add_suffix("_missing")
        return torch.tensor(np.hstack([z.to_numpy(), flags.to_numpy()]), dtype=torch.float32)

    def y(self, y):
        return torch.tensor(((y - self.y_mean) / self.y_std).to_numpy(), dtype=torch.float32)

    def unscale(self, out):
        return out * self.y_std + self.y_mean


def _net(n_in):
    layers, size = [], n_in
    for h in HIDDEN:
        layers += [nn.Linear(size, h), nn.ReLU(), nn.Dropout(DROPOUT)]
        size = h
    return nn.Sequential(*layers, nn.Linear(size, 1))


def _predict(net, x):
    net.eval()
    with torch.no_grad():
        return net(x).squeeze(1).numpy()


def fit(X, y, epochs=MAX_EPOCHS, X_valid=None, y_valid=None):
    torch.manual_seed(SEED)
    prep = Prep(X, y)
    x, t = prep.x(X), prep.y(y)
    net = _net(x.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    xv, tv = (prep.x(X_valid), prep.y(y_valid)) if X_valid is not None else (None, None)
    best, best_epoch, best_state = np.inf, epochs, None
    gen = torch.Generator().manual_seed(SEED)
    for epoch in range(1, epochs + 1):
        net.train()
        for idx in torch.randperm(len(x), generator=gen).split(BATCH):
            opt.zero_grad()
            nn.functional.mse_loss(net(x[idx]).squeeze(1), t[idx]).backward()
            opt.step()
        if xv is None:
            continue
        loss = float(np.mean((_predict(net, xv) - tv.numpy()) ** 2))
        if loss < best:
            best, best_epoch = loss, epoch
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, prep, best_epoch


def _excess(df, target, g):
    y = df[target] - df[f"{g}_winter_base"]
    return y, y.notna()


def _forecast(net, prep, rows, g):
    return prep.unscale(_predict(net, prep.x(rows[F.FORECAST_FEATURES]))) + rows[f"{g}_winter_base"].to_numpy()


def validate(df, target, g, years=VALID_YEARS):
    y, usable = _excess(df, target, g)
    pred, epochs = pd.Series(np.nan, index=df.index), {}
    for year in years:
        in_year = df["YEAR_MONTH"] // 100 == year
        train, valid = usable & (df["YEAR_MONTH"] // 100 < year), usable & in_year
        net, prep, epochs[year] = fit(df.loc[train, F.FORECAST_FEATURES], y[train],
                                      X_valid=df.loc[valid, F.FORECAST_FEATURES], y_valid=y[valid])
        pred[in_year] = _forecast(net, prep, df[in_year], g)
        log.info("pytorch %s: %d epochs", year, epochs[year])
    return pred, epochs


def rolling_forecast(df, target, g, epochs, months):
    y, usable = _excess(df, target, g)
    pred = pd.Series(np.nan, index=df.index)
    for ym in months:
        m = F._month_index(ym) - F.HORIZON
        origin = (m // 12) * 100 + m % 12 + 1
        train = usable & (df["YEAR_MONTH"] <= origin)
        net, prep, _ = fit(df.loc[train, F.FORECAST_FEATURES], y[train], epochs)
        rows = df["YEAR_MONTH"] == ym
        pred[rows] = _forecast(net, prep, df[rows], g)
    return pred
