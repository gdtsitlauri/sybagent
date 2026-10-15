"""Study 14: digital twin of the ERC-8004 registries.

    python scripts/twin.py sync --chain base                  # pull new blocks from the chain into the mirror
    python scripts/twin.py backtest --chain base --out results/14_digital_twin
    python scripts/twin.py whatif --chain base --tag trust --sybils 1000

The backtest uses the frozen crawl in data/raw/erc8004 so that it is reproducible; `sync` updates data/twin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sybagent.defenses.proposed import anchor_eclipse_bound  # noqa: E402
from sybagent.twin import forecast, risk  # noqa: E402
from sybagent.twin.state import DAY, TwinState  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "erc8004"
TWIN = ROOT / "data" / "twin"
K = 5
A = 3


def cmd_sync(args) -> None:
    st = TwinState.load(RAW, TWIN, args.chain)
    stats = st.sync(max_blocks=args.max_blocks)
    st.save(TWIN)
    print(json.dumps(stats, indent=2))
    log = ROOT / "results" / "14_digital_twin" / "sync_log.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**stats, "utc": pd.Timestamp.now("UTC").isoformat()}]).to_csv(
        log, mode="a", header=not log.exists(), index=False)


def origins(st: TwinState, warmup_weeks: int, horizon_days: float) -> list[float]:
    lo, hi = st.span
    first = lo + warmup_weeks * 7 * DAY
    return list(np.arange(first, hi - horizon_days * DAY, 7 * DAY))


def cmd_backtest(args) -> None:
    out = Path(args.out) / args.chain
    out.mkdir(parents=True, exist_ok=True)
    st = TwinState.load(RAW, TWIN, args.chain, use_mirror=False)
    lo, hi = st.span
    n_weeks = int((hi - lo) // (7 * DAY))
    meta = {"chain": args.chain, "start": lo, "end": hi, "weeks": n_weeks, "feedback": len(st.feedback),
            "registrations": len(st.identities)}

    # 1. aggregate forecasts
    series = forecast.weekly_series(st, lo, n_weeks)
    series.to_csv(out / "weekly_series.csv", index=False)
    bts = pd.concat([forecast.backtest_series(series, c) for c in
                     ("registrations", "feedback", "new_raters", "active_agents")], ignore_index=True)
    bts.to_csv(out / "series_backtest.csv", index=False)
    forecast.mase_table(bts, series).to_csv(out / "series_mase.csv", index=False)
    print(forecast.mase_table(bts, series).round(3).to_string(), flush=True)

    # 2. agent activity
    act = forecast.backtest_activity(st, origins(st, 6, 7))
    act.to_csv(out / "activity_backtest.csv", index=False)
    print(act.groupby("model")[["auc", "avg_precision"]].mean().round(3).to_string(), flush=True)

    # 3. eclipse cost over time and the risk backtest
    risk_rows, events = [], []
    for t in origins(st, 4, 28):
        rt = risk.risk_table(st, t, k=K, a=A, min_agents=K)
        ev = risk.takeover_events(st, t, 28 * DAY, K, K)
        risk_rows.append(rt)
        events.append(rt.merge(ev, on=["t", "tag"]))
    for t in origins(st, 4, 0)[-1:]:
        risk_rows.append(risk.risk_table(st, t, k=K, a=A, min_agents=K))
    rk = pd.concat(risk_rows, ignore_index=True)
    rk.to_csv(out / "risk_over_time.csv", index=False)
    ev = pd.concat(events, ignore_index=True)
    ev.to_csv(out / "takeover_backtest.csv", index=False)
    y = ev["takeover"].to_numpy(dtype=float)
    res = {"pairs": len(ev), "takeovers": int(y.sum()),
           "auc_low_cost": forecast.auc(-ev["full_eclipse_raters"].to_numpy(dtype=float), y),
           "auc_activity": forecast.auc(ev["activity_28d"].to_numpy(dtype=float), y),
           "auc_agents": forecast.auc(ev["agents"].to_numpy(dtype=float), y)}
    # does cost add information beyond activity? logistic on both, in-sample deviance comparison
    if 0 < y.sum() < len(y):
        x1 = np.log1p(ev[["activity_28d"]].to_numpy(dtype=float))
        x2 = np.hstack([x1, np.log1p(ev[["full_eclipse_raters"]].to_numpy(dtype=float))])
        from scipy.stats import chi2
        ll = []
        for x in (x1, x2):
            mu, sd = x.mean(0), x.std(0) + 1e-9
            w = forecast.fit_logistic((x - mu) / sd, y, l2=0.0)
            p = np.clip(forecast.predict_logistic(w, (x - mu) / sd), 1e-12, 1 - 1e-12)
            ll.append(float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))))
        res["lr_stat_cost_given_activity"] = 2 * (ll[1] - ll[0])
        res["lr_p_value"] = float(chi2.sf(res["lr_stat_cost_given_activity"], 1))
    print(json.dumps(res, indent=1), flush=True)

    # 4. what-if checks at the last origin: apply the computed attack and verify it is exactly sufficient
    t = origins(st, 4, 0)[-1]
    snap = st.at(t)
    rt = risk.risk_table(st, t, k=K, a=A, min_agents=K).sort_values("agents", ascending=False).head(10)
    checks = []
    rng = np.random.default_rng(0)
    for _, r in rt.iterrows():
        ok = risk.topk_sybil_share(risk.apply_attack(snap, r["tag"], K, int(r["full_eclipse_raters"]), t), r["tag"], K)
        short = risk.topk_sybil_share(risk.apply_attack(snap, r["tag"], K, int(r["full_eclipse_raters"]) - 1, t),
                                      r["tag"], K)
        checks.append({"tag": r["tag"], "agents": r["agents"], "raters_needed": r["full_eclipse_raters"],
                       "topk_sybil_share_with_cost": ok, "topk_sybil_share_one_less": short})
    n_reg = snap.n_registered
    mc = []
    for b_frac in (0.25, 1.0, 4.0):
        b = int(b_frac * n_reg)
        attacked = risk.apply_attack(snap, "_none_", b, 0, t)
        mc.append({"registered": n_reg, "sybil_agents": b, "anchors": A,
                   "monte_carlo": risk.anchors_monte_carlo(attacked, A, 20_000, rng),
                   "exact": risk.anchors_no_honest_exact(n_reg, b, A),
                   "theorem1_bound": anchor_eclipse_bound(n_reg, b, A)})
    pd.DataFrame(checks).to_csv(out / "whatif_checks.csv", index=False)
    pd.DataFrame(mc).to_csv(out / "whatif_anchors.csv", index=False)
    print(pd.DataFrame(checks).to_string(), pd.DataFrame(mc).to_string(), sep="\n", flush=True)
    meta.update(res)
    (out / "summary.json").write_text(json.dumps(meta, indent=2, default=float))


def cmd_whatif(args) -> None:
    st = TwinState.load(RAW, TWIN, args.chain)
    t = st.span[1] + 1
    snap = st.at(t)
    rk = risk.tag_rankings(snap).get(args.tag)
    if rk is None:
        sys.exit(f"unknown tag {args.tag!r}")
    cost = risk.reputation_eclipse_cost(rk, K)
    attacked = risk.apply_attack(snap, args.tag, K, args.raters if args.raters >= 0 else cost["full_eclipse_raters"], t)
    print(json.dumps({"tag": args.tag, "registered": snap.n_registered, **cost,
                      "topk_sybil_share_after_attack": risk.topk_sybil_share(attacked, args.tag, K),
                      "p_no_honest_anchor_bound": anchor_eclipse_bound(snap.n_registered, args.sybils, A),
                      "sybil_agents_for_p0.5": risk.anchors_identities_needed(snap.n_registered, A, 0.5)},
                     indent=2, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--chain", required=True)
    s.add_argument("--max-blocks", type=int)
    b = sub.add_parser("backtest")
    b.add_argument("--chain", required=True)
    b.add_argument("--out", default=str(ROOT / "results" / "14_digital_twin"))
    w = sub.add_parser("whatif")
    w.add_argument("--chain", required=True)
    w.add_argument("--tag", required=True)
    w.add_argument("--sybils", type=int, default=1000)
    w.add_argument("--raters", type=int, default=-1)
    args = ap.parse_args()
    {"sync": cmd_sync, "backtest": cmd_backtest, "whatif": cmd_whatif}[args.cmd](args)


if __name__ == "__main__":
    main()
