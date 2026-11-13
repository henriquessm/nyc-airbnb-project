"""
Generates every figure and every number quoted in docs/projects/regression/index.md.

    python code/regression.py      # ~10 min on a GPU, longer on CPU

Figures go to docs/projects/regression/figures/, numbers to
docs/projects/regression/results.json.

Order of the script = order of the report. The test set is touched only in
section 5 (and reused, after every choice is frozen, for the error analysis
of section 6).
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DEVICE, MLP, n_params, predict, set_seed, train  # noqa: E402
from preprocess import (  # noqa: E402
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, build_preprocessor,
    load_data, split_data, transform_target,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "projects" / "regression"
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = {}
T0 = time.time()

sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
CAT = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]
DPI = 140

VAL_SIZE = 0.15
MAX_EPOCHS, PATIENCE = 300, 20
SEEDS = [42, 43, 44, 45, 46]


def save(fig, n, slug):
    path = FIG_DIR / f"fig{n:02d}_{slug}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.name}")


def r(x, nd=3):
    return None if x is None else float(np.round(x, nd))


def to_dollars(log_pred):
    return np.expm1(log_pred)


def metrics(y_dollars, pred_dollars):
    """Metrics on the original scale (USD). r2_log is R^2 on log1p(price),
    the scale the EDA used (group medians gave 0.537 there)."""
    y, p = np.asarray(y_dollars, float), np.asarray(pred_dollars, float)
    return {
        "mae": r(mean_absolute_error(y, p), 2),
        "rmse": r(root_mean_squared_error(y, p), 2),
        "r2": r(r2_score(y, p), 4),
        "medae": r(median_absolute_error(y, p), 2),
        "mape": r(100 * np.mean(np.abs((y - p) / y)), 2),
        "r2_log": r(r2_score(np.log1p(y), np.log1p(p)), 4),
    }


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ============================================================================
# 0. Starting point: same data, same split, validation carved from train
# ============================================================================
df = load_data()
X_train_full, X_test, y_train_full, y_test = split_data(df)  # EDA split, seed 42

# validation comes out of the training set, stratified on log-price deciles
# like the EDA split, so the three sets see the same price range
strata = pd.qcut(np.log1p(y_train_full), q=10, labels=False, duplicates="drop")
X_tr, X_va, y_tr, y_va = train_test_split(
    X_train_full, y_train_full, test_size=VAL_SIZE,
    random_state=RANDOM_STATE, stratify=strata,
)

# the EDA pipeline, fitted on the training part only (validation and test
# are only transformed)
pre = build_preprocessor()
Xtr = pre.fit_transform(X_tr)
Xva = pre.transform(X_va)
Xte = pre.transform(X_test)
ytr, yva, yte = transform_target(y_tr), transform_target(y_va), transform_target(y_test)
feat_names = list(pre.get_feature_names_out())

RESULTS["split"] = {
    "train": len(X_tr), "val": len(X_va), "test": len(X_test),
    "train_full": len(X_train_full), "n_features": Xtr.shape[1],
    "median_price": {"train": r(y_tr.median(), 1), "val": r(y_va.median(), 1),
                     "test": r(y_test.median(), 1)},
}
RESULTS["device"] = str(DEVICE)
log(f"split {RESULTS['split']}")

# ============================================================================
# 1. Baselines (fitted on train, evaluated on validation)
# ============================================================================
baselines = {}

dummy_mean = DummyRegressor(strategy="mean").fit(Xtr, ytr)          # mean of log-price
dummy_median = DummyRegressor(strategy="median").fit(Xtr, y_tr)      # median in USD
baselines["dummy_mean"] = metrics(y_va, to_dollars(dummy_mean.predict(Xva)))
baselines["dummy_median"] = metrics(y_va, dummy_median.predict(Xva))

ridge_alphas = [0.01, 0.1, 1, 10, 100]
ridge_val = {}
for a in ridge_alphas:
    m = Ridge(alpha=a).fit(Xtr, ytr)
    ridge_val[a] = metrics(y_va, to_dollars(m.predict(Xva)))
best_alpha = min(ridge_val, key=lambda a: ridge_val[a]["mae"])
ridge = Ridge(alpha=best_alpha).fit(Xtr, ytr)
baselines["ridge"] = ridge_val[best_alpha]

hgb = HistGradientBoostingRegressor(random_state=RANDOM_STATE).fit(Xtr, ytr)
baselines["hgb"] = metrics(y_va, to_dollars(hgb.predict(Xva)))

RESULTS["baselines_val"] = baselines
RESULTS["ridge_alpha_search"] = {str(a): v for a, v in ridge_val.items()}
RESULTS["ridge_alpha"] = best_alpha
RESULTS["dummy_mean_value"] = r(float(np.expm1(dummy_mean.constant_.ravel()[0])), 2)
RESULTS["dummy_median_value"] = r(float(dummy_median.constant_.ravel()[0]), 2)
log(f"baselines on val: { {k: v['mae'] for k, v in baselines.items()} }")

# ============================================================================
# 2-3. The network and its training (no regularisation yet)
# ============================================================================
Y0 = float(ytr.mean())  # output bias starts at the trivial baseline
BASE = dict(hidden=(256, 128), dropout=0.0, norm=None, lr=1e-3, weight_decay=0.0,
            batch_size=256)


def run(cfg, seed=RANDOM_STATE):
    set_seed(seed)  # before building the model, so the seed also fixes the init
    model = MLP(Xtr.shape[1], hidden=cfg["hidden"], dropout=cfg["dropout"],
                norm=cfg["norm"], out_bias=Y0)
    hist = train(model, Xtr, ytr, Xva, yva, lr=cfg["lr"], batch_size=cfg["batch_size"],
                 weight_decay=cfg["weight_decay"], max_epochs=MAX_EPOCHS,
                 patience=PATIENCE, seed=seed)
    val = metrics(y_va, to_dollars(predict(model, Xva)))
    return model, hist, val


def cfg_name(c):
    h = "x".join(map(str, c["hidden"]))
    return (f"[{h}] lr={c['lr']:g} wd={c['weight_decay']:g} "
            f"drop={c['dropout']:g} norm={c['norm'] or '-'}")


base_model, base_hist, base_val = run(BASE)
P = n_params(base_model)
RESULTS["network_base"] = {
    "config": {**BASE, "hidden": list(BASE["hidden"])},
    "n_params": P, "params_per_train_row": r(P / len(X_tr), 2),
    "layer_params": {
        "hidden1": Xtr.shape[1] * 256 + 256, "hidden2": 256 * 128 + 128, "out": 128 + 1,
    },
    "best_epoch": base_hist["best_epoch"], "epochs_run": base_hist["epochs_run"],
    "val": base_val,
    "train_loss_at_best": r(base_hist["train_loss"][base_hist["best_epoch"] - 1], 4),
    "val_loss_at_best": r(base_hist["best_val_loss"], 4),
    "train_loss_last": r(base_hist["train_loss"][-1], 4),
    "val_loss_last": r(base_hist["val_loss"][-1], 4),
    "val_loss_epoch1": r(base_hist["val_loss"][0], 4),
    "steps_per_epoch": int(np.ceil(len(X_tr) / BASE["batch_size"])),
}
log(f"base network: {P} params, best epoch {base_hist['best_epoch']}, val {base_val}")

# a longer run without early stopping, only to see where the curves go
# if nobody stops them (patience = max_epochs)
set_seed(RANDOM_STATE)
long_model = MLP(Xtr.shape[1], hidden=BASE["hidden"], out_bias=Y0)
long_hist = train(long_model, Xtr, ytr, Xva, yva, lr=BASE["lr"],
                  batch_size=BASE["batch_size"], max_epochs=120, patience=120,
                  seed=RANDOM_STATE)
RESULTS["network_long"] = {
    "epochs": 120, "best_epoch": long_hist["best_epoch"],
    "val_loss_best": r(long_hist["best_val_loss"], 4),
    "val_loss_last": r(long_hist["val_loss"][-1], 4),
    "train_loss_last": r(long_hist["train_loss"][-1], 4),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
for ax, h, title in [(axes[0], base_hist, "(a) with early stopping (patience 20)"),
                     (axes[1], long_hist, "(b) 120 epochs, nobody stops it")]:
    ep = np.arange(1, len(h["train_loss"]) + 1)
    ax.plot(ep, h["train_loss"], color=CAT[0], label="train")
    ax.plot(ep, h["val_loss"], color=CAT[1], label="validation")
    ax.axvline(h["best_epoch"], color="0.4", ls="--", lw=1,
               label=f"best epoch ({h['best_epoch']})")
    ax.set(xlabel="epoch", ylabel="Huber loss on log1p(price)", title=title)
    ax.legend()
fig.suptitle("Figure 1 - Training and validation loss per epoch, base network [256x128], no regularisation")
save(fig, 1, "curves_base")

# ============================================================================
# 4. Regularisation and hyperparameters (chosen on validation)
# ============================================================================
REG = {
    "none": dict(BASE),
    "weight decay 0.05": {**BASE, "weight_decay": 0.05},
    "dropout 0.2": {**BASE, "dropout": 0.2},
    "BatchNorm": {**BASE, "norm": "batch"},
    "dropout 0.2 + weight decay 0.05": {**BASE, "dropout": 0.2, "weight_decay": 0.05},
}
reg_res, reg_hist, reg_entries = {}, {}, []
for name, cfg in REG.items():
    if name == "none":
        model, h, v = base_model, base_hist, base_val
    else:
        model, h, v = run(cfg)
    reg_hist[name] = h
    if name != "none":  # the base config enters the search table below
        reg_entries.append({"name": cfg_name(cfg), "cfg": cfg, "val": v,
                            "best_epoch": h["best_epoch"], "n_params": n_params(model)})
    gap = h["val_loss"][h["best_epoch"] - 1] - h["train_loss"][h["best_epoch"] - 1]
    reg_res[name] = {**v, "best_epoch": h["best_epoch"], "epochs_run": h["epochs_run"],
                     "train_loss": r(h["train_loss"][h["best_epoch"] - 1], 4),
                     "val_loss": r(h["best_val_loss"], 4), "gap": r(gap, 4)}
    log(f"reg {name}: {reg_res[name]}")
RESULTS["regularisation"] = reg_res

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
for i, (name, h) in enumerate(reg_hist.items()):
    ep = np.arange(1, len(h["val_loss"]) + 1)
    axes[0].plot(ep, h["val_loss"], color=CAT[i], label=name)
    axes[1].plot(ep, np.array(h["val_loss"]) - np.array(h["train_loss"]), color=CAT[i],
                 label=name)
axes[0].set(xlabel="epoch", ylabel="validation Huber loss", title="(a) validation loss",
            ylim=(None, max(reg_res[n]["val_loss"] for n in reg_res) * 1.25))
axes[1].axhline(0, color="0.5", lw=0.8)
axes[1].set(xlabel="epoch", ylabel="validation - train loss",
            title="(b) generalisation gap")
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8)
fig.suptitle("Figure 2 - Regularisation techniques compared with the network without any")
save(fig, 2, "regularisation")

# random search: 12 configurations sampled from a grid, plus the base one
GRID = {
    "hidden": [(128, 64), (256, 128), (512, 256), (256, 256, 128)],
    "lr": [3e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 0.01, 0.1],
    "dropout": [0.0, 0.1, 0.3],
    "norm": [None, "batch"],
}
combos = list(itertools.product(*GRID.values()))
rng = np.random.default_rng(RANDOM_STATE)
picked = rng.choice(len(combos), size=12, replace=False)
search_cfgs = [dict(zip(GRID.keys(), combos[i]), batch_size=256) for i in picked]
search_cfgs.append(dict(BASE))

search = []
for cfg in search_cfgs:
    if cfg == BASE:
        model, h, v = base_model, base_hist, base_val
    else:
        model, h, v = run(cfg)
    search.append({"name": cfg_name(cfg), "cfg": cfg, "val": v,
                   "best_epoch": h["best_epoch"], "n_params": n_params(model)})
    log(f"search {cfg_name(cfg)}: MAE {v['mae']} epoch {h['best_epoch']}")
# every configuration trained on validation is a candidate, including the
# regularisation variants above: the rule is "lowest validation MAE"
search += reg_entries
search.sort(key=lambda s: s["val"]["mae"])
BEST = search[0]["cfg"]
RESULTS["search"] = [
    {"name": s["name"], "n_params": s["n_params"], "best_epoch": s["best_epoch"], **s["val"]}
    for s in search
]
RESULTS["grid_size"] = len(combos)
RESULTS["chosen"] = {**BEST, "hidden": list(BEST["hidden"]), "name": search[0]["name"],
                     "n_params": search[0]["n_params"]}
log(f"chosen: {search[0]['name']}")

# ============================================================================
# 5. Test evaluation, run once, after every choice above is frozen
# ============================================================================
test_runs, test_models = [], {}
for seed in SEEDS:
    model, h, v = run(BEST, seed=seed)
    m = metrics(y_test, to_dollars(predict(model, Xte)))
    test_runs.append({"seed": seed, "best_epoch": h["best_epoch"],
                      "epochs_run": h["epochs_run"], "val_mae": v["mae"], **m})
    test_models[seed] = (model, h)
    log(f"test seed {seed}: {m}")

net_df = pd.DataFrame(test_runs)
keys = ["mae", "rmse", "r2", "medae", "mape", "r2_log"]
RESULTS["test_network_runs"] = test_runs
RESULTS["test_network"] = {k: {"mean": r(net_df[k].mean(), 4), "std": r(net_df[k].std(ddof=1), 4)}
                           for k in keys}

# baselines on the same test set; HGB also over the same 5 seeds
test_base = {
    "dummy_mean": metrics(y_test, to_dollars(dummy_mean.predict(Xte))),
    "dummy_median": metrics(y_test, dummy_median.predict(Xte)),
    "ridge": metrics(y_test, to_dollars(ridge.predict(Xte))),
}
hgb_runs = [metrics(y_test, to_dollars(
    HistGradientBoostingRegressor(random_state=s).fit(Xtr, ytr).predict(Xte)))
    for s in SEEDS]
hgb_df = pd.DataFrame(hgb_runs)
test_base["hgb"] = {k: r(hgb_df[k].mean(), 4) for k in keys}
RESULTS["test_baselines"] = test_base
RESULTS["test_hgb"] = {k: {"mean": r(hgb_df[k].mean(), 4), "std": r(hgb_df[k].std(ddof=1), 4)}
                       for k in keys}

net_mae = net_df["mae"].mean()
ridge_mae = test_base["ridge"]["mae"]
RESULTS["criterion"] = {
    "rule": "network test MAE at least 10% lower than the best of the trivial and linear baselines",
    "reference": "ridge", "reference_mae": ridge_mae,
    "threshold_mae": r(0.9 * ridge_mae, 2),
    "network_mae": r(net_mae, 2),
    "improvement_pct": r(100 * (1 - net_mae / ridge_mae), 2),
    "met": bool(net_mae <= 0.9 * ridge_mae),
    "gap_vs_ridge_in_std": r((ridge_mae - net_mae) / net_df["mae"].std(ddof=1), 1),
    "gap_vs_hgb": r(test_base["hgb"]["mae"] - net_mae, 2),
}
log(f"criterion: {RESULTS['criterion']}")

# the seed-42 model is the one used for the figures and the error analysis
final_model, final_hist = test_models[RANDOM_STATE]
pred_te = to_dollars(predict(final_model, Xte))
yt = y_test.to_numpy()
RESULTS["final_model"] = {"seed": RANDOM_STATE, "best_epoch": final_hist["best_epoch"],
                          "epochs_run": final_hist["epochs_run"],
                          "test": metrics(yt, pred_te)}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
ax = axes[0]
ax.scatter(yt, pred_te, s=4, alpha=0.2, color=CAT[0], label="test listing")
lim = [8, 12000]
ax.plot(lim, lim, color=CAT[5], lw=1.2, label="perfect prediction")
ax.set(xscale="log", yscale="log", xlim=lim, ylim=(20, 1500), xlabel="actual price (USD)",
       ylabel="predicted price (USD)", title="(a) predicted vs actual (log axes)")
ax.legend(loc="upper left")
res = yt - pred_te
ax = axes[1]
inside = np.abs(res) <= 600
ax.scatter(pred_te[inside], res[inside], s=4, alpha=0.2, color=CAT[0], label="test listing")
ax.axhline(0, color=CAT[5], lw=1.2, label="zero residual")
ax.set(xscale="log", ylim=(-600, 600), xlabel="predicted price (USD, log axis)",
       ylabel="residual = actual - predicted (USD)",
       title=f"(b) residuals in USD ({int((~inside).sum())} points beyond ±$600 not shown)")
ax.legend(loc="upper left")
ax = axes[2]
lres = np.log1p(yt) - np.log1p(pred_te)
ax.scatter(pred_te, lres, s=4, alpha=0.2, color=CAT[0], label="test listing")
ax.axhline(0, color=CAT[5], lw=1.2, label="zero residual")
ax.set(xscale="log", xlabel="predicted price (USD, log axis)",
       ylabel="log1p(actual) - log1p(predicted)", title="(c) residuals on the log scale")
ax.legend(loc="upper left")
fig.suptitle("Figure 3 - Chosen network on the test set (seed 42)")
save(fig, 3, "pred_vs_actual")

fig, ax = plt.subplots(figsize=(8, 3.8))
rows = [("Dummy (mean of log)", [test_base["dummy_mean"]["mae"]]),
        ("Dummy (median)", [test_base["dummy_median"]["mae"]]),
        ("Ridge", [ridge_mae]),
        ("Gradient boosting", list(hgb_df["mae"])),
        ("MLP (chosen)", list(net_df["mae"]))]
for i, (name, vals) in enumerate(rows):
    ax.scatter(vals, [i] * len(vals), s=40, color=CAT[i], zorder=3,
               label=f"{name} ({len(vals)} run{'s' if len(vals) > 1 else ''})")
ax.set_yticks(range(len(rows)), [n for n, _ in rows])
ax.set(xlabel="test MAE (USD)", title="Figure 4 - Test MAE per model and per seed")
ax.legend(fontsize=8, loc="upper right")
save(fig, 4, "test_seeds")

# ============================================================================
# 6. Error analysis (final model, test set; nothing is chosen from here)
# ============================================================================
ea = X_test.copy()
ea["price"] = yt
ea["pred"] = pred_te
ea["abs_err"] = np.abs(yt - pred_te)
ea["signed"] = pred_te - yt  # > 0 = overestimate
bands = [0, 50, 100, 200, 500, np.inf]
band_lbl = ["< $50", "$50-100", "$100-200", "$200-500", ">= $500"]
ea["price_band"] = pd.cut(ea["price"], bins=bands, labels=band_lbl, right=False)
ea["availability"] = np.where(ea["availability_365"] == 0, "closed (0 days)", "open (> 0 days)")
ea["min_nights_30"] = np.where(ea["minimum_nights"] >= 30, ">= 30 nights", "< 30 nights")


def group_err(col):
    g = ea.groupby(col, observed=True).agg(
        n=("abs_err", "size"), mae=("abs_err", "mean"), medae=("abs_err", "median"),
        bias=("signed", "mean"), median_price=("price", "median"))
    g["mae_rel_pct"] = 100 * g["mae"] / g["median_price"]
    return {str(k): {c: r(v, 2) for c, v in row.items()} for k, row in g.iterrows()}


groups = {c: group_err(c) for c in
          ["room_type", "neighbourhood_group", "price_band", "availability", "min_nights_30"]}
hood = ea.groupby("neighbourhood").agg(n=("abs_err", "size"), mae=("abs_err", "mean"),
                                       median_price=("price", "median"))
hood = hood[hood["n"] >= 30].sort_values("mae", ascending=False)
groups["neighbourhood_top5"] = {k: {c: r(v, 2) for c, v in row.items()}
                                for k, row in hood.head(5).iterrows()}
groups["neighbourhood_bottom3"] = {k: {c: r(v, 2) for c, v in row.items()}
                                   for k, row in hood.tail(3).iterrows()}
RESULTS["error_groups"] = groups

worst = ea.nlargest(10, "abs_err")
worst_cols = ["price", "pred", "abs_err", "neighbourhood", "room_type", "minimum_nights",
              "number_of_reviews", "availability_365", "calculated_host_listings_count"]
RESULTS["worst"] = [
    {**{c: (r(v, 1) if isinstance(v, (float, np.floating)) else
            (int(v) if isinstance(v, (int, np.integer)) else v))
        for c, v in row[worst_cols].items()},
     "name": df.loc[idx, "name"]}
    for idx, row in worst.iterrows()
]
top1pct = ea["abs_err"].nlargest(int(0.01 * len(ea))).sum() / ea["abs_err"].sum()
RESULTS["error_share_top1pct"] = r(100 * top1pct, 1)
RESULTS["n_test_ge_1000"] = int((ea["price"] >= 1000).sum())
below = ea["price"] < 1000
RESULTS["mae_below_1000"] = {
    "network": r(ea.loc[below, "abs_err"].mean(), 2),
    "ridge": r(np.abs(yt - to_dollars(ridge.predict(Xte)))[below.to_numpy()].mean(), 2),
    "rmse_network": r(root_mean_squared_error(yt[below.to_numpy()], pred_te[below.to_numpy()]), 2),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), sharey=False)
for ax, col, title in [(axes[0], "room_type", "(a) room type"),
                       (axes[1], "neighbourhood_group", "(b) borough"),
                       (axes[2], "price_band", "(c) actual price band")]:
    g = pd.DataFrame(groups[col]).T
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["mae"], width=0.4, color=CAT[0], label="MAE (USD)")
    ax.bar(x + 0.2, g["bias"], width=0.4, color=CAT[1], label="mean(pred - actual)")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x, [f"{i}\n(n={int(n)})" for i, n in zip(g.index, g["n"])], fontsize=8)
    ax.set(title=title, ylabel="USD")
    ax.legend(fontsize=8)
fig.suptitle("Figure 5 - Test error of the chosen network per subgroup")
save(fig, 5, "error_groups")

# permutation importance on the raw columns (before the pipeline), so a
# categorical with 109 dummies counts as one feature
RAW = NUMERIC_FEATURES + CATEGORICAL_FEATURES
GROUPS = {c: [c] for c in RAW}
GROUPS["all location (lat, long, borough, neighbourhood)"] = [
    "latitude", "longitude", "neighbourhood_group", "neighbourhood"]
N_REPEATS = 5


def perm_importance(predict_fn):
    base = mean_absolute_error(yt, predict_fn(pre.transform(X_test)))
    prng = np.random.default_rng(RANDOM_STATE)
    out = {}
    for name, cols in GROUPS.items():
        deltas = []
        for _ in range(N_REPEATS):
            Xp = X_test.copy()
            idx = prng.permutation(len(Xp))
            for c in cols:  # same permutation for the whole group, rows stay coherent
                Xp[c] = Xp[c].to_numpy()[idx]
            deltas.append(mean_absolute_error(yt, predict_fn(pre.transform(Xp))) - base)
        out[name] = {"mean": r(np.mean(deltas), 2), "std": r(np.std(deltas, ddof=1), 2)}
    return out


pi_net = perm_importance(lambda X: to_dollars(predict(final_model, X)))
pi_ridge = perm_importance(lambda X: to_dollars(ridge.predict(X)))
RESULTS["perm_importance"] = {"network": pi_net, "ridge": pi_ridge, "repeats": N_REPEATS}

order = sorted(RAW, key=lambda c: pi_net[c]["mean"])
fig, ax = plt.subplots(figsize=(9, 4.6))
y = np.arange(len(order))
ax.barh(y + 0.2, [pi_net[c]["mean"] for c in order], height=0.4, color=CAT[0],
        xerr=[pi_net[c]["std"] for c in order], label="MLP")
ax.barh(y - 0.2, [pi_ridge[c]["mean"] for c in order], height=0.4, color=CAT[1],
        xerr=[pi_ridge[c]["std"] for c in order], label="Ridge")
ax.set_yticks(y, order)
ax.set(xlabel="increase in test MAE when the column is shuffled (USD)",
       title=f"Figure 6 - Permutation importance on the test set ({N_REPEATS} repeats)")
ax.legend()
save(fig, 6, "permutation_importance")

RESULTS["runtime_s"] = r(time.time() - T0, 0)
(OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
log("wrote results.json")
