"""Forecasting components of the twin, all evaluated by rolling-origin backtests (only data before the origin).

Aggregate series (weekly): registrations, feedback events, new raters, active agents.
  Models: naive (last week), mean of the last 4 weeks, Holt's linear exponential smoothing (parameters chosen on the
  history before the origin).
Agent activity: P(agent receives >= 1 feedback in the next 7 days), logistic regression on simple state features,
  trained only on labelled windows that ended before the origin; baselines: recent activity, total popularity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

from sybagent.twin.state import DAY, TwinState

WEEK = 7 * DAY


# ------------------------------------------------------------------ aggregate series
def weekly_series(state: TwinState, t_start: float, n_weeks: int) -> pd.DataFrame:
    fb = state.feedback[~state.feedback["revoked"].astype(bool)]
    first_rater = fb.groupby("rater")["ts"].min()
    rows = []
    for w in range(n_weeks):
        a, b = t_start + w * WEEK, t_start + (w + 1) * WEEK
        f = fb[(fb["ts"] >= a) & (fb["ts"] < b)]
        rows.append({"week": w, "t": a,
                     "registrations": int(((state.identities["ts"] >= a) & (state.identities["ts"] < b)).sum()),
                     "feedback": len(f), "new_raters": int(((first_rater >= a) & (first_rater < b)).sum()),
                     "active_agents": int(f["agent"].nunique())})
    return pd.DataFrame(rows)


def _holt(y: np.ndarray, alpha: float, beta: float) -> tuple[float, float, float]:
    level, trend, sse = y[0], (y[1] - y[0]) if len(y) > 1 else 0.0, 0.0
    for v in y[1:]:
        pred = level + trend
        sse += (v - pred) ** 2
        new_level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
    return level, trend, sse


def holt_forecast(y: np.ndarray, h: int) -> float:
    best = None
    for a in np.linspace(0.05, 1.0, 20):
        for b in np.linspace(0.0, 0.5, 11):
            lv, tr, sse = _holt(y, a, b)
            if best is None or sse < best[0]:
                best = (sse, lv, tr)
    return max(0.0, best[1] + h * best[2])


def backtest_series(series: pd.DataFrame, col: str, min_hist: int = 6, horizons=(1, 4)) -> pd.DataFrame:
    y = series[col].to_numpy(dtype=float)
    rows = []
    for o in range(min_hist, len(y)):
        hist = y[:o]
        for h in horizons:
            if o + h - 1 >= len(y):
                continue
            truth = y[o + h - 1]
            preds = {"naive": hist[-1], "mean4": hist[-4:].mean(), "holt": holt_forecast(hist, h)}
            for m, p in preds.items():
                rows.append({"series": col, "origin": o, "h": h, "model": m, "pred": p, "truth": truth,
                             "abs_err": abs(p - truth)})
    return pd.DataFrame(rows)


def mase_table(bt: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error scaled by the in-sample one-step naive error of the whole series (MASE)."""
    out = []
    for (col, h, m), g in bt.groupby(["series", "h", "model"]):
        scale = np.mean(np.abs(np.diff(series[col].to_numpy(dtype=float))))
        out.append({"series": col, "h": h, "model": m, "mae": g["abs_err"].mean(),
                    "mase": g["abs_err"].mean() / scale if scale > 0 else float("nan"), "origins": len(g)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ agent activity
FEATURES = ["log_total", "log_raters", "log_fb7", "log_fb28", "recency", "age", "never"]


def agent_features(state: TwinState, t: float) -> pd.DataFrame:
    snap = state.at(t)
    reg = state.identities[state.identities["ts"] < t].drop_duplicates("agent").set_index("agent")["ts"]
    fb = snap.feedback
    g = fb.groupby("agent")
    df = pd.DataFrame(index=reg.index)
    df["total"] = g.size().reindex(df.index).fillna(0)
    df["raters"] = g["rater"].nunique().reindex(df.index).fillna(0)
    df["fb7"] = fb[fb["ts"] >= t - 7 * DAY].groupby("agent").size().reindex(df.index).fillna(0)
    df["fb28"] = fb[fb["ts"] >= t - 28 * DAY].groupby("agent").size().reindex(df.index).fillna(0)
    last = g["ts"].max().reindex(df.index)
    df["log_total"] = np.log1p(df["total"])
    df["log_raters"] = np.log1p(df["raters"])
    df["log_fb7"] = np.log1p(df["fb7"])
    df["log_fb28"] = np.log1p(df["fb28"])
    df["recency"] = np.where(last.isna(), 0.0, np.exp(-(t - last.fillna(t)) / (14 * DAY)))
    df["age"] = np.log1p((t - reg) / DAY)
    df["never"] = (df["total"] == 0).astype(float)
    return df


def labels(state: TwinState, t: float, index: pd.Index, horizon: float = 7 * DAY) -> np.ndarray:
    got = set(state.window(t, t + horizon)["agent"])
    return np.array([a in got for a in index], dtype=float)


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    xb = np.hstack([x, np.ones((len(x), 1))])

    def loss(w):
        z = xb @ w
        ll = np.sum(np.logaddexp(0, z) - y * z) / len(y)
        return ll + l2 * np.sum(w[:-1] ** 2)

    def grad(w):
        p = 1 / (1 + np.exp(-(xb @ w)))
        g = xb.T @ (p - y) / len(y)
        g[:-1] += 2 * l2 * w[:-1]
        return g

    return minimize(loss, np.zeros(xb.shape[1]), jac=grad, method="L-BFGS-B").x


def predict_logistic(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(np.hstack([x, np.ones((len(x), 1))]) @ w)))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    ys = y[order]
    if ys.sum() == 0:
        return float("nan")
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def backtest_activity(state: TwinState, origins: list[float], train_weeks: int = 4,
                      max_rows: int = 60_000, seed: int = 0) -> pd.DataFrame:
    """For each origin t: train on the windows starting at t-8d .. t-(7+7*train_weeks)d, whose labels end before t."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in origins:
        xs, ys = [], []
        for k in range(1, train_weeks + 1):
            t_tr = t - 7 * DAY - k * 7 * DAY
            f = agent_features(state, t_tr)
            if len(f) == 0:
                continue
            y = labels(state, t_tr, f.index)
            x = f[FEATURES].to_numpy()
            if len(x) > max_rows:
                keep = rng.choice(len(x), max_rows, replace=False)
                x, y = x[keep], y[keep]
            xs.append(x)
            ys.append(y)
        if not xs:
            continue
        x_tr, y_tr = np.vstack(xs), np.concatenate(ys)
        if y_tr.sum() == 0:
            continue
        mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-9
        w = fit_logistic((x_tr - mu) / sd, y_tr)
        f = agent_features(state, t)
        y = labels(state, t, f.index)
        x = f[FEATURES].to_numpy()
        scores = {"twin_logistic": predict_logistic(w, (x - mu) / sd),
                  "recent_7d": f["fb7"].to_numpy() + 1e-6 * f["total"].to_numpy(),
                  "popularity": f["total"].to_numpy()}
        for name, s in scores.items():
            rows.append({"t": t, "model": name, "agents": len(y), "positives": int(y.sum()),
                         "auc": auc(s, y), "avg_precision": average_precision(s, y)})
    return pd.DataFrame(rows)
