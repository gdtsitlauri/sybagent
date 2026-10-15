"""Write paper/numbers.tex and generated tables from results/ (every number in the paper comes from here)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = ROOT / "paper"
N_HONEST = 200
NA = "n/a"
DEF_LABEL = {"none": "no defense", "degree_bound": "degree bound", "random_mix": "random mix",
             "sybilrank": "SybilRank", "sumup": "SumUp", "proposed": "anchors (ours)"}


def pct(x: float) -> str:
    return f"{100 * x:.0f}\\%" if x >= 0.1 or x == 0 else f"{100 * x:.1f}\\%"


def num(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def sci(x: float) -> str:
    if x == 0:
        return "0"
    if x >= 1e-3:
        return f"{x:.3f}"
    mant, e = f"{x:.1e}".split("e")
    return f"${mant}\\times10^{{{int(e)}}}$"


def bound(n: int, b: int, a: int, beta: float = 0.0) -> float:
    p = beta + (1 - beta) * b / (n + b - a)
    return min(1.0, p) ** a


def _csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def attack(m: dict) -> int:
    s = _csv(RES / "1_attack" / "summary.csv")
    B = 200
    m["AtkB"] = str(B)
    if s is None:
        for k in ["AtkStaticShare", "AtkStaticEcl", "AtkAdaptShare", "AtkAdaptEcl", "AtkNoAttrShare", "HonestQuality"]:
            m[k] = NA
        return 0

    def get(a: float, adaptive: bool, metric: str, q: float = 0.85) -> float:
        sel = (s["adversary.n_sybils"] == B) & (s["adversary.attractiveness"] == a) & \
              (s["adversary.service_quality"] == q) & (s["adversary.adaptive_cards"] == adaptive)
        return float(s.loc[sel, metric].mean())

    m["AtkStaticShare"] = pct(get(0.5, False, "tail_target_sybil_share"))
    m["AtkStaticEcl"] = pct(get(0.5, False, "tail_targets_eclipsed_frac"))
    m["AtkAdaptShare"] = pct(get(0.5, True, "tail_target_sybil_share"))
    m["AtkAdaptEcl"] = pct(get(0.5, True, "tail_targets_eclipsed_frac"))
    m["AtkNoAttrShare"] = pct(get(0.0, False, "tail_target_sybil_share"))
    m["HonestQuality"] = num(float(s.loc[s["adversary.n_sybils"] == 0, "tail_mean_quality"].mean()))
    return len(s)


def prop2(m: dict) -> int:
    v = _csv(RES / "7_theory_check" / "prop2.csv")
    m["AtkPropTwoB"] = "8"
    if v is None:
        m["AtkPropTwoMaxN"] = NA
        return 0
    none = v[v["defense"] == "none"]
    m["AtkPropTwoMaxN"] = str(int(none["n_honest"].max()))
    return len(v)


def defenses(m: dict) -> int:
    d = _csv(RES / "2_defenses" / "summary.csv")
    DB, SB = 200, 50
    m["DefB"], m["DefSmallB"] = str(DB), str(SB)
    m["DefOursBound"] = pct(bound(N_HONEST, DB, 3))
    keys = ["DefNoneShare", "DefNoneEcl", "DefOursShare", "DefOursEcl", "DefOursMinHonest", "DefOursQuality",
            "DefNoneQuality", "DefBestName", "DefBestShare", "DefBestEcl", "DefSmallNoneEcl", "DefSmallOursEcl",
            "DefBaselineEclRange", "DefOursEclCount", "DefOursEclP", "DefOursNoAnchorAll"]
    if d is None:
        for k in keys:
            m[k] = NA
        return 0
    adaptive = d[d["adversary.adaptive_cards"] == True]  # noqa: E712
    g = adaptive[adaptive["adversary.n_sybils"] == DB].groupby("defense.kind")[
        ["tail_target_sybil_share", "tail_targets_eclipsed_frac", "tail_target_min_honest_neighbors",
         "tail_mean_quality"]].mean()
    m["DefNoneShare"] = pct(g.loc["none", "tail_target_sybil_share"])
    m["DefNoneEcl"] = pct(g.loc["none", "tail_targets_eclipsed_frac"])
    m["DefOursShare"] = pct(g.loc["proposed", "tail_target_sybil_share"])
    m["DefOursEcl"] = pct(g.loc["proposed", "tail_targets_eclipsed_frac"])
    m["DefOursMinHonest"] = num(g.loc["proposed", "tail_target_min_honest_neighbors"], 1)
    m["DefOursQuality"] = num(g.loc["proposed", "tail_mean_quality"], 3)
    m["DefNoneQuality"] = num(g.loc["none", "tail_mean_quality"], 3)
    base = g.drop(index=["none", "proposed"])
    best = base["tail_target_sybil_share"].idxmin()
    m["DefBestName"] = DEF_LABEL[best]
    m["DefBestShare"] = pct(base.loc[best, "tail_target_sybil_share"])
    m["DefBestEcl"] = pct(base.loc[best, "tail_targets_eclipsed_frac"])
    lo, hi = base["tail_targets_eclipsed_frac"].min(), base["tail_targets_eclipsed_frac"].max()
    m["DefBaselineEclRange"] = pct(lo) if pct(lo) == pct(hi) else f"{pct(lo)}--{pct(hi)}"
    ours = adaptive[(adaptive["adversary.n_sybils"] == DB) & (adaptive["defense.kind"] == "proposed")]
    n_targets_total = len(ours) * 5
    k = int(round(float(ours["tail_targets_eclipsed_frac"].sum()) * 5))
    from scipy.stats import binomtest
    m["DefOursEclCount"] = f"{k}/{n_targets_total}"
    m["DefOursEclP"] = f"{binomtest(k, n_targets_total, bound(N_HONEST, DB, 3), alternative='greater').pvalue:.2f}"
    m["DefOursNoAnchorAll"] = pct(float(ours["tail_honest_no_honest_anchor_frac"].mean()))
    gs = adaptive[adaptive["adversary.n_sybils"] <= SB].groupby("defense.kind")["tail_targets_eclipsed_frac"].mean()
    m["DefSmallNoneEcl"] = pct(gs["none"])
    m["DefSmallOursEcl"] = pct(gs["proposed"])

    rows = []
    for k in ["none", "degree_bound", "random_mix", "sybilrank", "sumup", "proposed"]:
        if k in g.index:
            r = g.loc[k]
            rows.append(f"{DEF_LABEL[k]} & {pct(r.tail_target_sybil_share)} & {pct(r.tail_targets_eclipsed_frac)} & "
                        f"{num(r.tail_target_min_honest_neighbors, 1)} & {num(r.tail_mean_quality, 3)} \\\\")
    (PAPER / "tab_defenses.tex").write_text(
        "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Defense & Sybil & Eclipsed & Min & Quality \\\\\n & share & & honest & \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    return len(d)


def anchors(m: dict) -> int:
    a = _csv(RES / "3_anchor_bound" / "summary.csv")
    m["AnchorsBoundThreeEight"] = sci(bound(N_HONEST, 800, 3))
    if a is None:
        for k in ["AnchorsWithinSE", "AnchorsMaxExcess", "AnchorsMeasThreeEight"]:
            m[k] = NA
        return 0
    ga = a.groupby(["adversary.n_sybils", "defense.params.n_anchors"])["tail_honest_no_honest_anchor_frac"].mean()
    n_eff = a["seed"].nunique() * N_HONEST
    cells = [(val, bound(N_HONEST, b, k)) for (b, k), val in ga.items()]
    within = sum(val <= bd + 2 * (bd * (1 - bd) / n_eff) ** 0.5 for val, bd in cells)
    m["AnchorsWithinSE"] = f"{within}/{len(cells)}"
    m["AnchorsMaxExcess"] = f"{100 * max(0.0, max(val - bd for val, bd in cells)):.1f}"
    m["AnchorsMeasThreeEight"] = sci(float(ga.get((800, 3), float("nan"))))
    return len(a)


def adaptive_attacks(m: dict) -> int:
    ad = _csv(RES / "4_adaptive_attacks" / "summary.csv")
    m["FloodBound"] = sci(bound(N_HONEST, 1600, 3))
    if ad is None:
        for k in ["FloodNoAnchor", "BiasNoAnchor", "BiasBound", "BiasZeroNoAnchor", "SleeperBefore", "SleeperAfter",
                  "SleeperNoAnchorBefore", "SleeperNoAnchorAfter"]:
            m[k] = NA
        return 0

    def sel(b: int, beta: float, k: str) -> pd.DataFrame:
        return ad[(ad["adversary.n_sybils"] == b) & (ad["adversary.registry_bias"] == beta) & (ad["defense.kind"] == k)]

    # flooding and biased sampling are measured BEFORE the sleeper switch (rounds 55-74)
    def pre_switch(b: int, beta: float) -> float:
        vals = []
        for f in sorted((RES / "4_adaptive_attacks" / "runs").glob("run_*.csv")):
            cfg = json.loads(f.with_suffix(".json").read_text())
            a_ = cfg["adversary"]
            if a_["n_sybils"] == b and a_["registry_bias"] == beta and cfg["defense"]["kind"] == "proposed":
                df = pd.read_csv(f)
                vals.append(df[(df["round"] >= 55) & (df["round"] < 75)]["honest_no_honest_anchor_frac"].mean())
        return sum(vals) / len(vals) if vals else float("nan")

    m["FloodNoAnchor"] = sci(pre_switch(1600, 0.0))
    m["BiasNoAnchor"] = sci(pre_switch(200, 0.5))
    m["BiasBound"] = sci(bound(N_HONEST, 200, 3, beta=0.5))
    m["BiasZeroNoAnchor"] = sci(pre_switch(200, 0.0))
    share_b, share_a, na_b, na_a = [], [], [], []
    for f in sorted((RES / "4_adaptive_attacks" / "runs").glob("run_*.csv")):
        cfg = json.loads(f.with_suffix(".json").read_text())
        if (cfg["adversary"]["n_sybils"] == 200 and cfg["adversary"]["registry_bias"] == 0.0
                and cfg["defense"]["kind"] == "proposed"):
            df = pd.read_csv(f)
            pre, post = df[(df["round"] >= 55) & (df["round"] < 75)], df[df["round"] >= 130]
            share_b.append(pre["target_sybil_share"].mean())
            share_a.append(post["target_sybil_share"].mean())
            na_b.append(pre["honest_no_honest_anchor_frac"].mean())
            na_a.append(post["honest_no_honest_anchor_frac"].mean())
    avg = lambda xs: sum(xs) / len(xs)  # noqa: E731
    m["SleeperBefore"] = pct(avg(share_b)) if share_b else NA
    m["SleeperAfter"] = pct(avg(share_a)) if share_a else NA
    m["SleeperNoAnchorBefore"] = sci(avg(na_b)) if na_b else NA
    m["SleeperNoAnchorAfter"] = sci(avg(na_a)) if na_a else NA
    return len(ad)


def audit(m: dict) -> int:
    s = _csv(RES / "5_second_opinions" / "summary.csv")
    keys = ["AuditNoneWrong", "AuditAnchorsWrong", "AuditFullWrong", "AuditAlwaysWrong", "AuditSybilBlocks",
            "AuditHonestBlocks", "AuditCost", "AuditCostAlways", "AuditServedWrongFull"]
    if s is None:
        for k in keys:
            m[k] = NA
        return 0
    n_all = len(s)
    s = s[s["adversary.n_sybils"] == 200]
    rho, u_col, d_col = "adversary.answer_wrong_prob", "defense.params.audit_prob", "defense.kind"

    def wrong(kind: str, r: float, u: float | None = None, metric: str = "tail_wrong_used_rate") -> float:
        sel = (s[d_col] == kind) & (s[rho] == r)
        if u is not None:
            sel &= s[u_col] == u
        return float(s.loc[sel, metric].mean())

    m["AuditNoneWrong"] = pct(wrong("none", 0.3))
    m["AuditAnchorsWrong"] = pct(wrong("proposed", 0.3, 0.0))
    m["AuditFullWrong"] = pct(wrong("proposed", 0.3, 0.3))
    m["AuditAlwaysWrong"] = pct(wrong("proposed", 0.3, 1.0))
    m["AuditServedWrongFull"] = pct(wrong("proposed", 0.3, 0.3, "tail_wrong_answer_rate"))
    tasks = N_HONEST * 100
    sel = lambda u: s[(s[d_col] == "proposed") & (s[rho] == 0.3) & (s[u_col] == u)]  # noqa: E731
    full = sel(0.3)
    m["AuditSybilBlocks"] = f"{full['final_blocked_sybil_pairs'].mean():,.0f}".replace(",", "{,}")
    m["AuditHonestBlocks"] = f"{full['final_blocked_honest_pairs'].mean():,.0f}".replace(",", "{,}")
    m["AuditCost"] = num(float(full["final_audit_calls"].mean()) / tasks, 2)
    m["AuditCostAlways"] = num(float(sel(1.0)["final_audit_calls"].mean()) / tasks, 2)
    rows = []
    us = [0.0, 0.1, 0.3, 1.0]
    for r in sorted(s[rho].unique()):
        cells = [wrong("none", r)] + [wrong("proposed", r, u) for u in us]
        rows.append(f"{r:g} & " + " & ".join(pct(c) for c in cells) + " \\\\")
    (PAPER / "tab_audit.tex").write_text(
        "\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "$\\rho$ & none & anchors & $u{=}0.1$ & $u{=}0.3$ & $u{=}1$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    return n_all


def robust_audit(m: dict) -> int:
    base = _csv(RES / "5b_robust_second_opinions" / "baselines" / "summary.csv")
    main_ = _csv(RES / "5b_robust_second_opinions" / "main" / "summary.csv")
    keys = ["RbBadMaj", "RbBadUna", "RbGoodMaj", "RbGoodUna", "RbWrongMaj", "RbWrongUna", "RbBadMajCat", "RbBadUnaCat",
            "RbHonestBlocksMax", "RbNoAnchorZero", "RbNoAnchorTwoThirds", "RbNoAnchorAll", "RbNoneWrong",
            "RbAnchorsWrong", "RbAlwaysLieGlob", "RbAlwaysLieCat"]
    if base is None or main_ is None:
        for k in keys:
            m[k] = NA
        return 0
    R, C, RHO, CAT = "defense.params.audit_rule", "adversary.coordinated_lies", "adversary.answer_wrong_prob", \
        "defense.params.anchor_category_frac"

    def g(rule: str, cat: float, metric: str, coord: bool = True, rho: float = 0.3) -> float:
        sel = (main_[R] == rule) & (main_[CAT] == cat) & (main_[C] == coord) & (main_[RHO] == rho)
        return float(main_.loc[sel, metric].mean())

    m["RbBadMaj"] = num(g("majority", 0.0, "tail_bad_corrections"), 1)
    m["RbBadUna"] = num(g("unanimous", 0.0, "tail_bad_corrections"), 1)
    m["RbGoodMaj"] = num(g("majority", 0.0, "tail_good_corrections"), 1)
    m["RbGoodUna"] = num(g("unanimous", 0.0, "tail_good_corrections"), 1)
    m["RbWrongMaj"] = pct(g("majority", 0.0, "tail_wrong_used_rate"))
    m["RbWrongUna"] = pct(g("unanimous", 0.0, "tail_wrong_used_rate"))
    m["RbBadMajCat"] = num(g("majority", 1.0, "tail_bad_corrections"), 1)
    m["RbBadUnaCat"] = num(g("unanimous", 1.0, "tail_bad_corrections"), 1)
    m["RbHonestBlocksMax"] = f"{int(main_['final_blocked_honest_pairs'].max())}"
    m["RbNoAnchorZero"] = pct(g("majority", 0.0, "tail_honest_no_honest_anchor_frac"))
    m["RbNoAnchorTwoThirds"] = pct(g("majority", 0.67, "tail_honest_no_honest_anchor_frac"))
    m["RbNoAnchorAll"] = pct(g("majority", 1.0, "tail_honest_no_honest_anchor_frac"))
    m["RbAlwaysLieGlob"] = pct(g("majority", 0.0, "tail_wrong_used_rate", rho=1.0))
    m["RbAlwaysLieCat"] = pct(g("majority", 1.0, "tail_wrong_used_rate", rho=1.0))
    sb = base[(base[RHO] == 0.3) & (base[C] == True)]  # noqa: E712
    m["RbNoneWrong"] = pct(float(sb.loc[sb["defense.kind"] == "none", "tail_wrong_used_rate"].mean()))
    m["RbAnchorsWrong"] = pct(float(sb.loc[sb["defense.kind"] == "proposed", "tail_wrong_used_rate"].mean()))

    rows = []
    for rule in ["majority", "unanimous"]:
        for cat, label in [(0.0, "0/3"), (0.67, "2/3"), (1.0, "3/3")]:
            rows.append(f"{rule} & {label} & {pct(g(rule, cat, 'tail_wrong_used_rate'))} & "
                        f"{num(g(rule, cat, 'tail_bad_corrections'), 1)} & {num(g(rule, cat, 'tail_good_corrections'), 1)} & "
                        f"{pct(g(rule, cat, 'tail_honest_no_honest_anchor_frac'))} \\\\")
    (PAPER / "tab_robust.tex").write_text(
        "\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{llrrrr}\n\\toprule\n"
        "Rule & Cat. & Wrong & Bad & Good & No honest \\\\\n & anchors & used & corr. & corr. & anchor \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    return len(base) + len(main_)


def llm(m: dict) -> int:
    L = _csv(RES / "6_llm_selector" / "summary.csv")
    keys = [f"Llm{a}{d}{x}" for a in ["Adapt", "Static"] for d in ["None", "Ours"] for x in ["Share", "Min"]]
    for k in keys:
        m[k] = NA
    if L is None:
        return 0
    gl = L.groupby(["adversary.adaptive_cards", "defense.kind"])[
        ["tail_target_sybil_share", "tail_target_min_honest_neighbors"]].mean()
    for flag, an in [(True, "Adapt"), (False, "Static")]:
        for k, dn in [("none", "None"), ("proposed", "Ours")]:
            if (flag, k) in gl.index:
                m[f"Llm{an}{dn}Share"] = pct(gl.loc[(flag, k), "tail_target_sybil_share"])
                m[f"Llm{an}{dn}Min"] = num(gl.loc[(flag, k), "tail_target_min_honest_neighbors"], 1)
    return len(L)


def real_data(m: dict) -> None:
    ep = ROOT / "tables" / "erc8004_structure.csv"
    e = pd.read_csv(ep, index_col=0) if ep.exists() else pd.DataFrame()
    for chain, pre in [("ethereum", "Eth"), ("base", "Base")]:
        if chain not in e.columns:
            for k in ["SingleRater", "TopTen", "MixRatio"]:
                m[pre + k] = NA
            continue
        c = e[chain].astype(float)
        m[pre + "SingleRater"] = pct(c["agents_single_rater_frac"])
        m[pre + "TopTen"] = pct(c["top10_rater_share"])
        m[pre + "MixRatio"] = f"{c['giant_mixing_time_est'] / c['fast_mixing_reference_logn']:,.0f}".replace(",", "{,}")
    cols = [c for c in ["ethereum", "base"] if c in e.columns]
    if cols:
        spec = [("Feedback events", "feedback", "int"), ("Distinct raters", "raters", "int"),
                ("Rated agents", "agents_rated", "int"), ("Top-10 rater share", "top10_rater_share", "pct"),
                ("Agents with 1 rater", "agents_single_rater_frac", "pct"),
                ("Raters rating 1 agent", "raters_single_agent_frac", "pct"),
                ("Components", "components", "int"), ("Giant component", "giant_component_frac", "pct"),
                ("Mixing time (steps)", "giant_mixing_time_est", "int"),
                (r"$\log n$", "fast_mixing_reference_logn", "f1")]
        lines = []
        for label, key, fmt in spec:
            vals = []
            for c in cols:
                x = float(e.loc[key, c])
                if fmt == "int":
                    vals.append(f"{int(round(x)):,}".replace(",", "{,}"))
                elif fmt == "pct":
                    vals.append(pct(x))
                else:
                    vals.append(f"{x:.1f}")
            lines.append(label + " & " + " & ".join(vals) + " \\\\")
        (PAPER / "tab_erc8004.tex").write_text(
            "\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n & "
            + " & ".join(c.capitalize() for c in cols) + " \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    mp = ROOT / "tables" / "moltbook_structure.csv"
    if mp.exists():
        mb = pd.read_csv(mp, index_col=0)["moltbook"].astype(float)
        m["MoltAgents"] = f"{int(mb['agents_in_graph']):,}".replace(",", "{,}")
        m["MoltEdges"] = f"{int(mb['edges']):,}".replace(",", "{,}")
        m["MoltTopOnePct"] = pct(mb["top1pct_agents_interaction_share"])
        m["MoltMixRatio"] = f"{mb['giant_mixing_time_est'] / mb['fast_mixing_reference_logn']:.0f}"
    else:
        for k in ["MoltAgents", "MoltEdges", "MoltTopOnePct", "MoltMixRatio"]:
            m[k] = NA
    m["ErcPaperSybil"] = "90.6\\%"
    m["CrawlDate"] = "October 2026"


def _int(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", "{,}")


def onchain(m: dict) -> None:
    g, r = _csv(RES / "10_onchain_sampler" / "gas.csv"), _csv(RES / "10_onchain_sampler" / "grinding.csv")
    cp = RES / "10_onchain_sampler" / "checks.json"
    keys = ["OcReqGas", "OcCbGasThree", "OcTotalGas", "OcGasSpread", "OcGrindMaxDev", "OcPerAnchorGas", "OcBlockhashGasThree", "OcChiP", "OcRequests",
            "OcMismatch", "OcLate", "OcReroll", "OcGrindOne", "OcGrindHundred", "OcGrindThousandAFour", "OcQThree"]
    if g is None or r is None or not cp.exists():
        for k in keys:
            m[k] = NA
        return
    c = json.loads(cp.read_text())
    big = g[g["registry_size"] == g["registry_size"].max()].set_index("anchors")
    m["OcReqGas"] = _int(big.loc[3, "request_gas"])
    m["OcCbGasThree"] = _int(big.loc[3, "callback_gas"])
    m["OcTotalGas"] = f"{(big.loc[3, 'request_gas'] + big.loc[3, 'callback_gas']) / 1000:.0f}{{,}}000"
    m["OcPerAnchorGas"] = _int((big.loc[8, "callback_gas"] - big.loc[4, "callback_gas"]) / 4)
    m["OcBlockhashGasThree"] = _int(big.loc[3, "blockhash_gas"])
    spread = g.groupby("anchors")["callback_gas"].agg(lambda s: s.max() - s.min()).max()
    m["OcGasSpread"] = _int(spread)
    u, f = c["uniformity"], c["frontrun_reroll"]
    m["OcChiP"] = f"{u['p_value']:.2f}"
    m["OcRequests"] = _int(u["requests"])
    m["OcMismatch"] = str(u["reference_mismatches"])
    m["OcLate"] = f"{f['anchors_from_late_identities']}/{f['trials']}"
    m["OcReroll"] = "rejected" if f["second_request_while_pending_rejected"] else "accepted"
    s = r[(r["sybils"] == 50) & (r["anchors"] == 3)].set_index("rerolls")
    m["OcQThree"] = pct(s.loc[0, "q_exact"])
    m["OcGrindOne"] = pct(s.loc[1, "p_no_honest_anchor"])
    m["OcGrindHundred"] = pct(s.loc[100, "p_no_honest_anchor"])
    s4 = r[(r["sybils"] == 50) & (r["anchors"] == 4)].set_index("rerolls")
    m["OcGrindThousandAFour"] = pct(s4.loc[1000, "p_no_honest_anchor"])
    m["OcGrindMaxDev"] = f"{(r['p_no_honest_anchor'] - r['prop6_prediction']).abs().max():.3f}"


def strategic(m: dict) -> int:
    s = _csv(RES / "11_strategic_attacker" / "summary.csv")
    br = _csv(RES / "11_strategic_attacker" / "games" / "best_responses.csv")
    lin = _csv(RES / "11_strategic_attacker" / "games" / "damage_linearity.csv")
    eq = _csv(RES / "11_strategic_attacker" / "games" / "audit_equilibria.csv")
    ig = _csv(RES / "11_strategic_attacker" / "games" / "identity_game.csv")
    keys = ["StDefenders", "StRlMaxGain", "StRlGammaZero", "StLinRtwo", "StBigWrongUone", "StBigWrongUzero",
            "StSmallWrongUone", "StAaWrong", "StAaNoAnchor", "StNoAaNoAnchor", "StAaHonestBlocks",
            "StPriceSmall", "StPriceBig", "IgAnchorsDeter", "IgBNOneRho"]
    if s is None or br is None:
        for k in keys:
            m[k] = NA
        return 0
    m["StDefenders"] = str(len(br))
    m["StRlMaxGain"] = f"{br['rl_minus_best_fixed'].max():.3f}"
    pols = list((RES / "11_strategic_attacker" / "q").glob("*_policy.json"))
    gz = sum(json.loads(p.read_text())["gamma"] == 0.0 for p in pols)
    m["StRlGammaZero"] = f"{gz}/{len(pols)}"
    m["StLinRtwo"] = f"{lin['r2'].min():.3f}"
    b = br.set_index(["n_sybils", "audit_prob", "audit_anchors", "audit_rule"])
    m["StBigWrongUzero"] = pct(b.loc[(200, 0.0, False, "majority"), "wrong_used"])
    m["StBigWrongUone"] = pct(b.loc[(200, 1.0, False, "majority"), "wrong_used"])
    m["StSmallWrongUone"] = pct(b.loc[(50, 1.0, False, "majority"), "wrong_used"])
    m["StAaWrong"] = pct(b.loc[(200, 1.0, True, "majority"), "wrong_used"])
    m["StAaHonestBlocks"] = _int(b.loc[(200, 1.0, True, "majority"), "blocked_honest_pairs"])
    # Theorem-1 quantity under anchor audits, from the always-lie runs
    rows = []
    for f in (RES / "11_strategic_attacker" / "runs").glob("B200_u1.0_aa*_majority_lie1.0_s*.csv"):
        rows.append({"aa": "_aa1_" in f.name,
                     "v": pd.read_csv(f).tail(20)["honest_no_honest_anchor_frac"].mean()})
    r = pd.DataFrame(rows).groupby("aa")["v"].mean()
    m["StAaNoAnchor"] = pct(r[True])
    m["StNoAaNoAnchor"] = pct(r[False])
    m["StBoundBig"] = pct(bound(N_HONEST, 200, 3))
    be = _csv(RES / "11_strategic_attacker" / "games" / "break_even.csv")
    if be is not None:
        e = be[be["audit_rule"] == "majority"].set_index("n_sybils")["break_even_price"]
        m["StPriceSmall"] = f"{e[50]:.2f}"
        m["StPriceBig"] = f"{e[200]:.3f}"
    if ig is not None:
        sim = ig[(ig["network"] == "simulator") & (ig["rho"] == 0.1)]
        deter = sim[sim["attacker_utility"] <= 0]
        m["IgAnchorsDeter"] = str(int(deter["anchors"].min())) if len(deter) else ">8"
        m["IgBNOneRho"] = f"{sim[sim['anchors'] == 3]['best_B_over_N'].iloc[0]:.1f}"
    n_train = 0
    t = _csv(RES / "11_strategic_attacker" / "rl_training.csv")
    if t is not None:
        n_train = len(t)
    return len(s) + n_train


def pareto(m: dict) -> int:
    s = _csv(RES / "12_pareto" / "summary.csv")
    knee = _csv(RES / "12_pareto" / "knee.csv")
    hv = _csv(RES / "12_pareto" / "hypervolume.csv")
    eps = _csv(RES / "12_pareto" / "epsilon_constraint.csv")
    keys = ["PaConfigs", "PaFrontSmall", "PaFrontBig", "PaKneeBig", "PaFrontAaShare", "PaKneeBigDamage",
            "PaKneeBigCost", "PaKneeBigNoAnchor", "PaKneeSafe", "PaKneeSafeDamage", "PaKneeSafeCost",
            "PaKneeSafeNoAnchor", "PaMinDamageBig"]
    if s is None or knee is None:
        for k in keys:
            m[k] = NA
        return 0
    h = hv.set_index("n_sybils")
    m["PaConfigs"] = str(int(h["configs"].iloc[0]))
    m["PaFrontSmall"] = str(int(h.loc[50, "front_size"]))
    m["PaFrontBig"] = str(int(h.loc[200, "front_size"]))
    kk = knee[knee["adversary.n_sybils"] == 200].set_index("subset")
    k = kk.loc["all"]
    ks = kk.loc["keeps_theorem1"]
    m["PaKneeSafe"] = (f"$a{{=}}{int(ks['defense.params.n_anchors'])}$, $u{{=}}{ks['defense.params.audit_prob']:g}$, "
                       f"{ks['defense.params.audit_rule']}")
    m["PaKneeSafeDamage"] = pct(ks["damage"])
    m["PaKneeSafeCost"] = num(ks["cost"], 1)
    m["PaKneeSafeNoAnchor"] = pct(ks["no_honest_anchor"])
    m["PaMinDamageBig"] = pct(f_all["damage"].min()) if (f_all := _csv(RES / "12_pareto" / "config_means.csv")[lambda d: d["adversary.n_sybils"] == 200]) is not None else NA
    m["PaKneeBig"] = (f"$a{{=}}{int(k['defense.params.n_anchors'])}$, $u{{=}}{k['defense.params.audit_prob']:g}$, "
                      f"{k['defense.params.audit_rule']}"
                      + (", anchors audited" if bool(k["defense.params.audit_anchors"]) else ""))
    m["PaKneeBigDamage"] = pct(k["damage"])
    m["PaKneeBigCost"] = num(k["cost"], 1)
    m["PaKneeBigNoAnchor"] = pct(k["no_honest_anchor"])
    f = _csv(RES / "12_pareto" / "pareto_front.csv")
    m["PaFrontAaShare"] = pct(f["defense.params.audit_anchors"].astype(bool).mean())
    return len(s)


def dense(m: dict) -> None:
    p = RES / "13_dense_subgraph" / "summary.json"
    b = _csv(RES / "13_dense_subgraph" / "blocks.csv")
    inj = _csv(RES / "13_dense_subgraph" / "injection.csv")
    keys = ["DsExact", "DsGreedy", "DsBaseTopRaters", "DsBaseTopAgents", "DsBaseTopDensity", "DsBaseTopZ",
            "DsBaseSigDens", "DsBaseSigW", "DsEthSigDens", "DsEthSigW", "DsMinerShare", "DsMinerDensity",
            "DsRecallTwo", "DsRecallFive", "DsRecallLow"]
    if not p.exists() or b is None or inj is None:
        for k in keys:
            m[k] = NA
        return
    s = json.loads(p.read_text())
    m["DsExact"] = f"{s['ethereum']['exact_max_density']:.3f}"
    m["DsGreedy"] = f"{s['ethereum']['greedy_max_density']:.3f}"
    top = b[(b["chain"] == "base") & (b["method"] == "density") & (b["block"] == 0)].iloc[0]
    m["DsBaseTopRaters"] = str(int(top["raters"]))
    m["DsBaseTopAgents"] = str(int(top["agents"]))
    m["DsBaseTopDensity"] = pct(top["pair_density"])
    m["DsBaseTopZ"] = f"{top['z']:.0f}"
    m["DsBaseSigDens"] = pct(s["base"]["density_agents_in_significant_blocks_frac"])
    m["DsBaseSigW"] = pct(s["base"]["weighted_agents_in_significant_blocks_frac"])
    m["DsEthSigDens"] = pct(s["ethereum"]["density_agents_in_significant_blocks_frac"])
    m["DsEthSigW"] = pct(s["ethereum"]["weighted_agents_in_significant_blocks_frac"])
    mv = b[(b["chain"] == "base") & (b["top_tag"] == "miner-vouch")].iloc[0]
    m["DsMinerShare"] = pct(mv["feedback_share"])
    m["DsMinerDensity"] = f"{mv['pair_density']:.3f}"
    r = inj.groupby("density")["recall"].mean()
    m["DsRecallLow"] = pct(r.loc[0.05])
    m["DsRecallTwo"] = pct(r.loc[0.2])
    m["DsRecallFive"] = pct(r.loc[0.5])


def twin(m: dict) -> None:
    for chain, pre in [("ethereum", "TwEth"), ("base", "TwBase")]:
        d = RES / "14_digital_twin" / chain
        sp = d / "summary.json"
        keys = ["Registered", "AucTwin", "AucRecent", "AucPop", "RepMedian", "AncNeeded", "AucCost", "AucActivity",
                "Takeovers", "Pairs", "HoltWins"]
        if not sp.exists():
            for k in keys:
                m[pre + k] = NA
            continue
        s = json.loads(sp.read_text())
        m[pre + "Registered"] = _int(s["registrations"])
        a = pd.read_csv(d / "activity_backtest.csv").groupby("model")["auc"].mean()
        m[pre + "AucTwin"] = f"{a['twin_logistic']:.2f}"
        m[pre + "AucRecent"] = f"{a['recent_7d']:.2f}"
        m[pre + "AucPop"] = f"{a['popularity']:.2f}"
        rk = pd.read_csv(d / "risk_over_time.csv")
        last = rk[rk["t"] == rk["t"].max()]
        m[pre + "RepMedian"] = _int(last["reputation_identities"].median())
        m[pre + "AncNeeded"] = _int(last["anchors_identities"].median())
        m[pre + "AucCost"] = f"{s['auc_low_cost']:.2f}"
        m[pre + "AucActivity"] = f"{s['auc_activity']:.2f}"
        m[pre + "Takeovers"] = _int(s["takeovers"])
        m[pre + "Pairs"] = _int(s["pairs"])
        ms = pd.read_csv(d / "series_mase.csv")
        best = ms[ms["h"] == 1].loc[lambda x: x.groupby("series")["mase"].idxmin()]
        m[pre + "HoltWins"] = str(int((best["model"] == "holt").sum()))
    lp = RES / "14_digital_twin" / "sync_log.csv"
    if lp.exists():
        lg = pd.read_csv(lp)
        steady = lg[lg["to_block"] - lg["from_block"] < 100]
        m["TwSyncLag"] = f"{steady['state_lag_seconds'].median():.0f}" if len(steady) else NA
        m["TwSyncSeconds"] = f"{steady['seconds'].median():.1f}" if len(steady) else NA
    else:
        m["TwSyncLag"] = m["TwSyncSeconds"] = NA


def robustness(m: dict) -> int:
    """Study 15 (20 seeds, sensitivity), Proposition 1b (tightness) and the DP best response."""
    from math import comb
    from scipy.stats import binomtest
    n = 0
    s = _csv(RES / "15_robustness" / "seeds20" / "summary.csv")
    keys = ["RbTwentyEcl", "RbTwentyEclK", "RbTwentyExact", "RbTwentyP", "RbTwentyNoAnchor", "RbTwentySmallEcl",
            "SensConfigs", "SensNoneEcl", "SensNoAnchorMin", "SensNoAnchorMax", "TightMaxDev", "DpSettings",
            "DpMaxGap", "DpMcMaxDev"]
    for k in keys:
        m[k] = NA
    if s is not None:
        n += len(s)
        p = s[(s["adversary.n_sybils"] == 200) & (s["defense.kind"] == "proposed")]
        k = int(round((p["final_targets_eclipsed_frac"] * 5).sum()))
        tot = 5 * len(p)
        m["RbTwentyEclK"] = f"{k}/{tot}"
        m["RbTwentyEcl"] = f"{100 * k / tot:.1f}\\%"
        m["RbTwentyExact"] = f"{100 * comb(200, 3) / comb(399, 3):.1f}\\%"
        m["RbTwentyBound"] = f"{100 * bound(N_HONEST, 200, 3):.1f}\\%"
        m["RbTwentyNoAnchorPrecise"] = f"{100 * p['tail_honest_no_honest_anchor_frac'].mean():.1f}\\%"
        m["RbTwentyP"] = f"{binomtest(k, tot, bound(N_HONEST, 200, 3), alternative='greater').pvalue:.2f}"
        m["RbTwentyNoAnchor"] = pct(p["tail_honest_no_honest_anchor_frac"].mean())
        q = s[(s["adversary.n_sybils"] == 50) & (s["defense.kind"] == "proposed")]
        m["RbTwentySmallEcl"] = pct(q["tail_targets_eclipsed_frac"].mean())
    sens = [_csv(RES / "15_robustness" / d / "summary.csv") for d in ("sensitivity", "selection")]
    if all(x is not None for x in sens):
        n += sum(len(x) for x in sens)
        allc = pd.concat(sens, ignore_index=True)
        knobs = [c for c in allc.columns if c.startswith(("network.", "selection.")) and c != "defense.kind"]
        m["SensConfigs"] = str(allc.groupby(knobs, dropna=False).ngroups)
        m["SensNoneEcl"] = pct(allc[allc["defense.kind"] == "none"]["tail_targets_eclipsed_frac"].min())
        na = allc[allc["defense.kind"] == "proposed"].groupby(knobs, dropna=False)[
            "tail_honest_no_honest_anchor_frac"].mean()
        m["SensNoAnchorMin"], m["SensNoAnchorMax"] = pct(na.min()), pct(na.max())
    s3 = _csv(RES / "3_anchor_bound" / "summary.csv")
    if s3 is not None:
        g = s3.groupby(["adversary.n_sybils", "defense.params.n_anchors"])["tail_honest_no_honest_anchor_frac"].mean()
        dev = max(abs(v - comb(b, a) / comb(N_HONEST + b - 1, a)) for (b, a), v in g.items())
        m["TightMaxDev"] = f"{dev:.3f}"
    dp = _csv(RES / "11_strategic_attacker" / "dp" / "dp_best_response.csv")
    if dp is not None:
        m["DpSettings"] = str(len(dp))
        gap = dp["optimal_minus_always"].abs().max()
        m["DpMaxGap"] = "$<10^{-9}$" if gap < 1e-9 else f"{gap:.3f}"
        m["DpMcMaxDev"] = f"{(dp['monte_carlo'] - dp['optimal']).abs().max():.2f}"
    return n


def real_cards(m: dict) -> int:
    """Study 16: real ERC-8004 agent cards, LLM-written tasks and Sybil cards."""
    st = RES / "16_real_cards" / "bundle_stats.json"
    s = _csv(RES / "16_real_cards" / "embedding" / "summary.csv")
    keys = ["RcCards", "RcAdaptCos", "RcHonestCos", "RcAdaptShareSmall", "RcAdaptEclSmall", "RcOursEclSmall",
            "RcAdaptShareBig", "RcAdaptEclBig", "RcOursEclBig", "RcOursNoAnchorBig", "RcStaticEclBig",
            "RcQualNone", "RcQualOurs"]
    for k in keys:
        m[k] = NA
    n = 0
    if st.exists():
        b = json.loads(st.read_text())
        m["RcCards"] = _int(b["unique_real_cards"])
        m["RcAdaptCos"] = f"{b['mean_cos_adaptive_vs_task']:.2f}"
        m["RcHonestCos"] = f"{b['mean_best_honest_cos_per_task']:.2f}"
    if s is not None:
        n += len(s)
        g = s.groupby(["adversary.n_sybils", "adversary.adaptive_cards", "defense.kind"]).mean(numeric_only=True)
        m["RcAdaptShareSmall"] = pct(g.loc[(50, True, "none"), "tail_target_sybil_share"])
        m["RcAdaptEclSmall"] = pct(g.loc[(50, True, "none"), "tail_targets_eclipsed_frac"])
        m["RcOursEclSmall"] = pct(g.loc[(50, True, "proposed"), "tail_targets_eclipsed_frac"])
        m["RcAdaptShareBig"] = pct(g.loc[(200, True, "none"), "tail_target_sybil_share"])
        m["RcAdaptEclBig"] = pct(g.loc[(200, True, "none"), "tail_targets_eclipsed_frac"])
        m["RcOursEclBig"] = pct(g.loc[(200, True, "proposed"), "tail_targets_eclipsed_frac"])
        m["RcOursNoAnchorBig"] = f"{100 * g.loc[(200, True, 'proposed'), 'tail_honest_no_honest_anchor_frac']:.1f}\\%"
        m["RcStaticEclBig"] = pct(g.loc[(200, False, "none"), "tail_targets_eclipsed_frac"])
        m["RcQualNone"] = num(g.loc[(200, True, "none"), "tail_mean_quality"], 3)
        m["RcQualOurs"] = num(g.loc[(200, True, "proposed"), "tail_mean_quality"], 3)
    llm = _csv(RES / "16_real_cards" / "llm" / "summary.csv")
    for k in ["RcLlmAdaptNone", "RcLlmAdaptOurs", "RcLlmStaticNone", "RcLlmStaticOurs", "RcLlmMinNone", "RcLlmMinOurs"]:
        m[k] = NA
    if llm is not None:
        n += len(llm)
        g = llm.groupby(["adversary.adaptive_cards", "defense.kind"]).mean(numeric_only=True)
        m["RcLlmAdaptNone"] = pct(g.loc[(True, "none"), "tail_target_sybil_share"])
        m["RcLlmAdaptOurs"] = pct(g.loc[(True, "proposed"), "tail_target_sybil_share"])
        m["RcLlmStaticNone"] = pct(g.loc[(False, "none"), "tail_target_sybil_share"])
        m["RcLlmStaticOurs"] = pct(g.loc[(False, "proposed"), "tail_target_sybil_share"])
        m["RcLlmMinNone"] = num(g.loc[(True, "none"), "tail_target_min_honest_neighbors"], 1)
        m["RcLlmMinOurs"] = num(g.loc[(True, "proposed"), "tail_target_min_honest_neighbors"], 1)
    return n


def shared(m: dict) -> int:
    """Study 20 (shared defenses), 20b (pooling), 21 (on-chain blocklist), twin committee planning."""
    n = 0
    main_ = _csv(RES / "20_shared_defense" / "main.csv")
    keys = ["ShNaiveFramed", "ShWorstLocalSmall", "ShWorstBestSmall", "ShWorstLocalMid", "ShWorstBestMid",
            "ShFramedHundred", "ShFramedTwoHundred", "ShAlertsFramed", "ShRlCommittee", "ShRlAlways",
            "ShVOneRl", "ShVOneMuted", "ShVOneAlways", "ShRlCommitteeRange", "ShRlLieFrac", "ShQuorumSafeBig", "ShBudgetLocal", "ShBudgetPool",
            "ShPoolWorst", "ShNoPoolWorst", "ShPoolRl", "ShPoolRlSmall", "ShPoolRlMuted", "ShPoolRlSmallMuted",
            "ShPoolWorstSmall", "ShNoPoolWorstSmall", "ShPoolHonestBlocked", "ShPoolCapFiveAlways", "BlGasFifteen", "BlGasFortyFive", "BlDrawsBad",
            "TcEthLimit", "TcBaseLimit", "TcShareLimit", "TcSeatsTen", "TcSeatsTwenty", "TcGasTen"]
    for k in keys:
        m[k] = NA
    if main_ is not None:
        n += len(main_)
        x = main_[main_["false_reports"] == 1]
        w = x.groupby(["n_sybils", "share", "attacker"])["wrong_used"].mean().unstack("attacker").max(axis=1)
        m["ShWorstLocalSmall"] = pct(w[(25, "local")])
        m["ShWorstBestSmall"] = pct(w[(25, "alerts+committee")])
        m["ShWorstLocalMid"] = pct(w[(50, "local")])
        m["ShWorstBestMid"] = pct(w[(50, "alerts+committee")])
        fr = x.groupby(["n_sybils", "share"])["global_blocks_honest"].mean()
        m["ShNaiveFramed"] = _int(fr[(25, "naive")])
        m["ShFramedHundred"] = _int(fr[(100, "committee")])
        m["ShFramedTwoHundred"] = _int(fr[(200, "committee")])
        m["ShAlertsFramed"] = _int(x[x["share"] == "alerts"]["global_blocks_honest"].max())
        b = x.groupby(["n_sybils", "share", "attacker"])["wrong_used"].mean()
        m["ShRlAlways"] = pct(b[(50, "committee", "lie1.0")])
        m["ShBudgetLocal"] = pct(b[(50, "local", "budget")])
    rl = _csv(RES / "20_shared_defense" / "rl.csv")
    if rl is not None:
        n += len(rl)
        rc = rl[(rl["n_sybils"] == 50) & (rl["share"] == "committee")]
        m["ShRlCommittee"] = pct(rc["wrong_used"].mean())
        m["ShRlCommitteeRange"] = f"{pct(rc['wrong_used'].min())}--{pct(rc['wrong_used'].max())}"
        m["ShRlLieFrac"] = pct(rc["lie_frac"].mean())
    v1 = _csv(RES / "20_shared_defense_v1_superseded" / "rl.csv")
    v1m = _csv(RES / "20_shared_defense_v1_superseded" / "main.csv")
    if v1 is not None and v1m is not None:
        r = v1[(v1["n_sybils"] == 50) & (v1["share"] == "committee")]
        m["ShVOneRl"] = pct(r["wrong_used"].mean())
        m["ShVOneMuted"] = _int(r["honest_reporters_muted"].mean())
        q = v1m[(v1m["n_sybils"] == 50) & (v1m["share"] == "committee") & (v1m["false_reports"] == 1)]
        m["ShVOneAlways"] = pct(q[q["attacker"] == "lie1.0"]["wrong_used"].mean())
    # intermediate designs (probe evidence): two-sided verdicts, then pooling with probes
    for k in ["ShProbeRl", "ShProbeAlways", "ShProbePoolRl", "ShProbePoolMuted", "ShProbeWorstSmall",
              "ShProbeWorstMid"]:
        m[k] = NA
    pr = _csv(RES / "20_shared_defense" / "probes" / "rl.csv")
    pm = _csv(RES / "20_shared_defense" / "probes" / "main.csv")
    if pr is not None and pm is not None:
        m["ShProbeRl"] = pct(pr[(pr["n_sybils"] == 50) & (pr["share"] == "committee")]["wrong_used"].mean())
        q = pm[(pm["n_sybils"] == 50) & (pm["share"] == "committee") & (pm["false_reports"] == 1)]
        m["ShProbeAlways"] = pct(q[q["attacker"] == "lie1.0"]["wrong_used"].mean())
        x = pm[pm["false_reports"] == 1]
        w = x.groupby(["n_sybils", "share", "attacker"])["wrong_used"].mean().unstack("attacker").max(axis=1)
        m["ShProbeWorstSmall"] = pct(w[(25, "alerts+committee")])
        m["ShProbeWorstMid"] = pct(w[(50, "alerts+committee")])
    pp = _csv(RES / "20_shared_defense" / "pooling_probes" / "summary.csv")
    if pp is not None:
        r = pp[(pp["attacker"] == "rl") & (pp["n_sybils"] == 25) & (pp["pool_evidence"])]
        m["ShProbePoolRl"] = pct(r["wrong_used"].mean())
        m["ShProbePoolMuted"] = _int(r["honest_reporters_muted"].mean())
    th = _csv(RES / "20_shared_defense" / "threshold.csv")
    if th is not None:
        n += len(th)
        big = th[(th["n_sybils"] == 200) & (th["quorum"] >= 0.9) & (th["committee_size"] == 45)]
        m["ShQuorumSafeBig"] = _int(big["global_blocks_honest"].mean())
    pool = _csv(RES / "20_shared_defense" / "pooling" / "summary.csv")
    if pool is not None:
        n += len(pool)
        # final design: pooling with 10 committees per round (5 congests the queue under pooling)
        p5 = pool[(pool["committee_capacity"] == 10) & (pool["attacker"] != "rl")]
        c5 = pool[(pool["committee_capacity"] == 5) & (pool["attacker"] != "rl")]
        wc = c5.groupby(["n_sybils", "pool_evidence", "attacker"])["wrong_used"].mean()
        m["ShPoolCapFiveAlways"] = pct(wc[(50, True, "lie1.0")])
        w = p5.groupby(["n_sybils", "pool_evidence", "attacker"])["wrong_used"].mean().unstack("attacker").max(axis=1)
        m["ShNoPoolWorst"] = pct(w[(50, False)])
        m["ShPoolWorst"] = pct(w[(50, True)])
        bb = p5.groupby(["n_sybils", "pool_evidence", "attacker"])["wrong_used"].mean()
        m["ShBudgetPool"] = pct(bb[(50, True, "budget")])
        m["ShPoolWorstSmall"] = pct(w[(25, True)])
        m["ShNoPoolWorstSmall"] = pct(w[(25, False)])
        m["ShPoolHonestBlocked"] = _int(p5[p5["pool_evidence"] & (p5["n_sybils"] <= 50)]["global_blocks_honest"].max())
        for b, key in ((50, "ShPoolRl"), (25, "ShPoolRlSmall")):
            rp = pool[(pool["attacker"] == "rl") & (pool["n_sybils"] == b) & (pool["pool_evidence"])]
            if len(rp):
                m[key] = pct(rp["wrong_used"].mean())
                m[key + "Muted"] = _int(rp["honest_reporters_muted"].mean())
    gas = _csv(RES / "21_onchain_blocklist" / "gas.csv")
    chk = RES / "21_onchain_blocklist" / "checks.json"
    if gas is not None and chk.exists():
        g = gas.groupby("committee_size")["total_gas"].mean()
        m["BlGasFifteen"] = f"{g[15] / 1e6:.2f}"
        m["BlGasFortyFive"] = f"{g[45] / 1e6:.2f}"
        c = json.loads(chk.read_text())
        m["BlDrawsBad"] = f"{c['draws_with_accused_or_duplicates']}/{c['draws_checked']}"
    lim = _csv(RES / "14_digital_twin" / "committee_limits.csv")
    plan = _csv(RES / "14_digital_twin" / "committee_planning.csv")
    if lim is not None and plan is not None:
        l3 = lim[lim["anchors"] == 3].set_index("chain")
        m["TcEthLimit"] = _int(l3.loc["ethereum", "sybils_at_rho_half"])
        m["TcBaseLimit"] = _int(l3.loc["base", "sybils_at_rho_half"])
        m["TcShareLimit"] = pct(l3.loc["base", "sybil_share_at_rho_half"])
        pb = plan[(plan["chain"] == "base") & (plan["anchors"] == 3)].set_index("sybil_share")
        m["TcSeatsTen"] = str(pb.loc[pb.index[np.isclose(pb.index, 0.1, atol=1e-3)][0], "committee_size_needed"])
        m["TcSeatsTwenty"] = str(pb.loc[pb.index[np.isclose(pb.index, 0.2, atol=1e-3)][0], "committee_size_needed"])
        m["TcGasTen"] = f"{pb.loc[pb.index[np.isclose(pb.index, 0.1, atol=1e-3)][0], 'gas_per_judgement'] / 1e6:.1f}"
    return n


def verifiable(m: dict) -> int:
    """Deposits (twin), study 22 (verifiable answers), study 17 (real answers)."""
    n = 0
    for k in ["DpRepEth", "DpAncEth", "DpRatioEth", "DpRatioBase", "VfLocalBig", "VfBanBig", "VfBanHuge",
              "VfLocalHuge", "VfCostBig", "VfHonestBans", "VfSleeperBig", "VfSleeperBigHigh", "VfRlBig",
              "VfWorstBigFive", "VfWorstBigTwenty", "VfWorstHugeTwenty", "VfAllBanned", "RaExpertAcc",
              "RaGenericAcc", "RaHonestOnly", "RaNoneBig", "RaAnchorsBig", "RaSoBig", "RaVerifyBig",
              "RaNoneMid", "RaCommMid", "RaLowRateNone", "RaLowRateComm", "RaHonestBanned", "RaCommSmall",
              "RaSoMid", "RaVerifyMid"]:
        m[k] = NA
    dep = _csv(RES / "14_digital_twin" / "deposit.csv")
    if dep is not None:
        d = dep.set_index(["chain", "case"])
        m["DpRepEth"] = _int(d.loc[("ethereum", "reputation discovery, no anchors"), "identities_needed"])
        m["DpAncEth"] = _int(d.loc[("ethereum", "3 anchors, eclipse probability 0.5"), "identities_needed"])
        m["DpRatioEth"] = _int(d.loc[("ethereum", "3 anchors, eclipse probability 0.5"), "identities_vs_no_anchors"])
        m["DpRatioBase"] = _int(d.loc[("base", "3 anchors, eclipse probability 0.5"), "identities_vs_no_anchors"])
    s = _csv(RES / "22_verifiable" / "summary.csv")
    if s is not None:
        n += len(s)
        f = s[s["attacker"] != "rl"]
        g = f.groupby(["n_sybils", "verify_prob", "mode", "attacker"])["wrong_used"].mean()
        worst = g.unstack("attacker").max(axis=1)
        m["VfLocalBig"] = pct(worst[(200, 0.0, "correct")])
        m["VfLocalHuge"] = pct(worst[(400, 0.0, "correct")])
        m["VfWorstBigFive"] = pct(worst[(200, 0.05, "ban+rep")])
        m["VfWorstBigTwenty"] = pct(worst[(200, 0.2, "ban+rep")])
        m["VfWorstHugeTwenty"] = pct(worst[(400, 0.2, "ban+rep")])
        m["VfBanBig"] = pct(g[(200, 0.05, "ban", "lie1.0")])
        m["VfBanHuge"] = pct(g[(400, 0.05, "ban", "lie1.0")])
        m["VfSleeperBig"] = pct(g[(200, 0.05, "ban+rep", "sleeper")])
        m["VfSleeperBigHigh"] = pct(g[(200, 0.2, "ban+rep", "sleeper")])
        c = f[(f["n_sybils"] == 200) & (f["verify_prob"] == 0.05) & (f["mode"] == "ban")]
        m["VfCostBig"] = f"{c['verifications_per_task'].mean():.2f}"
        m["VfHonestBans"] = _int(f[f["mode"] != "correct"]["global_blocks_honest"].max())
        al = f[(f["n_sybils"] == 200) & (f["verify_prob"] == 0.05) & (f["mode"] == "ban") & (f["attacker"] == "lie1.0")]
        m["VfAllBanned"] = _int(al["global_blocks_sybil"].mean())
        r = s[(s["attacker"] == "rl") & (s["n_sybils"] == 200)]
        if len(r):
            m["VfRlBig"] = pct(r["wrong_used"].mean())
    st = RES / "17_real_answers" / "bundle_stats.json"
    if st.exists():
        b = json.loads(st.read_text())
        m["RaExpertAcc"] = pct(b["expert_accuracy"])
        m["RaGenericAcc"] = pct(b["generic_accuracy"])
    ra = _csv(RES / "17_real_answers" / "summary.csv")
    if ra is not None:
        n += len(ra)
        g = ra.groupby(["n_sybils", "defense", "attacker"])["wrong_used"].mean()
        m["RaHonestOnly"] = pct(g[(0, "none", "lie1.0")])
        m["RaNoneBig"] = pct(g[(200, "none", "lie1.0")])
        m["RaAnchorsBig"] = pct(g[(200, "anchors", "lie1.0")])
        m["RaSoBig"] = pct(g[(200, "second_opinions_u1", "lie1.0")])
        m["RaVerifyBig"] = pct(g[(200, "verify_0.2", "lie1.0")])
        m["RaNoneMid"] = pct(g[(50, "none", "lie1.0")])
        m["RaCommMid"] = pct(g[(50, "committees", "lie1.0")])
        m["RaLowRateNone"] = pct(g[(50, "none", "lie0.3")])
        m["RaLowRateComm"] = pct(g[(50, "committees", "lie0.3")])
        h = ra.groupby(["n_sybils", "defense", "attacker"])["global_blocks_honest"].mean()
        m["RaHonestBanned"] = _int(h[(200, "verify_0.2", "lie1.0")])
        m["RaCommSmall"] = pct(g[(25, "committees", "lie1.0")])
        m["RaSoMid"] = pct(g[(50, "second_opinions_u1", "lie1.0")])
        m["RaVerifyMid"] = pct(g[(50, "verify_0.2", "lie1.0")])
    return n


def _plain(v: str) -> str:
    return (v.replace("{=}", "=").replace("^{", "^").replace("\\%", "%").replace("{,}", ",").replace("$", "").replace("\\times10^{", "e")
            .replace("}", "").replace("--", "–"))


def write_results_md(m: dict) -> None:
    p = {k: _plain(v) for k, v in m.items()}
    lines = [
        "# SybAgent: all numbers",
        "",
        "Generated by `analysis/make_numbers.py` from `results/`. The same values are used in the paper",
        "(`paper/numbers.tex`). N = 200 honest agents unless stated; tail = mean over the last 20 rounds; 5 seeds.",
        "",
        "## 1. Attack (no defense, Sybil service 0.85, single target)",
        "",
        "| setting | Sybil share of target's peers | target eclipsed (rounds) |",
        "|---|---|---|",
        f"| B = {p['AtkB']}, static cards, attractiveness 0.5 | {p['AtkStaticShare']} | {p['AtkStaticEcl']} |",
        f"| B = {p['AtkB']}, query-adaptive cards | {p['AtkAdaptShare']} | {p['AtkAdaptEcl']} |",
        f"| B = {p['AtkB']}, static cards, attractiveness 0 | {p['AtkNoAttrShare']} | |",
        "",
        f"Proposition 2 check: {p['AtkPropTwoB']} adaptive Sybils eclipse the target in round 0 for every"
        f" N in 100..{p['AtkPropTwoMaxN']} (`results/7_theory_check/prop2.csv`).",
        "",
        "## 2. Defenses (query-adaptive cards, 5 targets)",
        "",
        f"At B = {p['DefB']}: no defense {p['DefNoneShare']} Sybil share, eclipsed {p['DefNoneEcl']};"
        f" every classical baseline eclipsed {p['DefBaselineEclRange']};"
        f" anchors {p['DefOursShare']} share, eclipsed {p['DefOursEcl']} ({p['DefOursEclCount']} target instances,"
        f" Theorem 1 bound {p['DefOursBound']}, one-sided binomial p = {p['DefOursEclP']});"
        f" all honest agents without an honest anchor {p['DefOursNoAnchorAll']}.",
        f"With B <= {p['DefSmallB']}: eclipsed {p['DefSmallNoneEcl']} without defense, {p['DefSmallOursEcl']} with anchors.",
        f"Service quality: {p['DefOursQuality']} with anchors vs {p['DefNoneQuality']} without.",
        "",
        "## 3. Theorem 1 bound (anchors a = 1..4, B = 100..800)",
        "",
        f"Measured fraction of honest agents without an honest anchor <= bound + 2 SE in {p['AnchorsWithinSE']}"
        f" configurations; largest excess {p['AnchorsMaxExcess']} percentage points."
        f" a = 3, B = 800: bound {p['AnchorsBoundThreeEight']}, measured {p['AnchorsMeasThreeEight']}.",
        "",
        "## 4. Attacks on the defense",
        "",
        f"- Flooding, B = 1600 (before the switch): {p['FloodNoAnchor']} without an honest anchor (bound {p['FloodBound']}).",
        f"- Biased sampling, beta = 0.5, B = 200: {p['BiasNoAnchor']} (beta = 0: {p['BiasZeroNoAnchor']}; bound {p['BiasBound']}).",
        f"- Sleeper Sybils (switch at round 75), B = 200: targets' Sybil share {p['SleeperBefore']} -> {p['SleeperAfter']};"
        f" no-honest-anchor fraction {p['SleeperNoAnchorBefore']} -> {p['SleeperNoAnchorAfter']}.",
        "",
        "## 5. Second opinions (B = 200, Sybils lie with probability 0.3)",
        "",
        f"Wrong answers used: no defense {p['AuditNoneWrong']}, anchors {p['AuditAnchorsWrong']},"
        f" u = 0.3 {p['AuditFullWrong']} ({p['AuditCost']} extra queries/task), u = 1 {p['AuditAlwaysWrong']}"
        f" ({p['AuditCostAlways']} extra queries/task). Blocked pairs at u = 0.3: {p['AuditSybilBlocks']} Sybil,"
        f" {p['AuditHonestBlocks']} honest. Wrong answers served (before correction) at u = 0.3: {p['AuditServedWrongFull']}.",
        "",
        "## 5b. Colluding anchors and category anchors (B = 200, u = 0.3, skill-dependent honest accuracy)",
        "",
        f"Coordinated Sybils lying 30%: bad corrections per round {p['RbBadMaj']} (majority) -> {p['RbBadUna']}"
        f" (unanimity); with category anchors {p['RbBadMajCat']} -> {p['RbBadUnaCat']}. Good corrections"
        f" {p['RbGoodMaj']} -> {p['RbGoodUna']}; wrong answers used {p['RbWrongMaj']} -> {p['RbWrongUna']}."
        f" Honest peers blocked (max over all configurations): {p['RbHonestBlocksMax']}.",
        f"Honest agents without an honest anchor: {p['RbNoAnchorZero']} (global anchors),"
        f" {p['RbNoAnchorTwoThirds']} (2 of 3 category), {p['RbNoAnchorAll']} (3 of 3 category)."
        f" Always-lying Sybils: wrong answers used {p['RbAlwaysLieGlob']} (global) vs {p['RbAlwaysLieCat']} (category).",
        f"Anchors without second opinions: {p['RbAnchorsWrong']} wrong answers used vs {p['RbNoneWrong']} without a defense.",
        "",
        "## 6. LLM selector (Qwen2.5-3B, N = 20, B = 10, 3 targets, 3 seeds)",
        "",
        "| cards | defense | Sybil share | min honest peers of a target |",
        "|---|---|---|---|",
        f"| adaptive | none | {p['LlmAdaptNoneShare']} | {p['LlmAdaptNoneMin']} |",
        f"| adaptive | anchors | {p['LlmAdaptOursShare']} | {p['LlmAdaptOursMin']} |",
        f"| static | none | {p['LlmStaticNoneShare']} | {p['LlmStaticNoneMin']} |",
        f"| static | anchors | {p['LlmStaticOursShare']} | {p['LlmStaticOursMin']} |",
        "",
        "## 7. Real data",
        "",
        f"- ERC-8004 Ethereum: agents with a single reviewer {p['EthSingleRater']}, top-10 reviewers write"
        f" {p['EthTopTen']} of feedback, review graph mixes {p['EthMixRatio']}x slower than log n.",
        f"- ERC-8004 Base: {p['BaseSingleRater']}, {p['BaseTopTen']}, {p['BaseMixRatio']}x.",
        f"- Moltbook: {p['MoltAgents']} agents, {p['MoltEdges']} edges, top 1% of agents in {p['MoltTopOnePct']}"
        f" of interactions, mixing {p['MoltMixRatio']}x slower than log n.",
        "",
        "## 10. On-chain anchor sampling (contracts/AnchorSampler.sol, local EVM)",
        "",
        f"Gas at a = 3: request {p['OcReqGas']}, VRF callback {p['OcCbGasThree']} (+{p['OcPerAnchorGas']} per extra"
        f" anchor); unsafe block-hash sampler {p['OcBlockhashGasThree']}. Callback gas varies by at most"
        f" {p['OcGasSpread']} between N = 100 and N = 1e9."
        f" {p['OcRequests']} on-chain draws: {p['OcMismatch']} mismatches with the Python reference, chi-square"
        f" uniformity p = {p['OcChiP']}. Anchors drawn from identities registered after the request: {p['OcLate']}."
        f" Second request while one is pending: {p['OcReroll']}.",
        f"Grinding (B = 50, N = 200, a = 3): one seed {p['OcQThree']} (exact), 2 seeds {p['OcGrindOne']},"
        f" 101 seeds {p['OcGrindHundred']}; a = 4 with 1001 seeds {p['OcGrindThousandAFour']}."
        f" Largest deviation from Proposition 6: {p['OcGrindMaxDev']}.",
        "",
        "## 11. Strategic attacker and Stackelberg games",
        "",
        f"{p['StDefenders']} defender commitments. The RL attacker's damage exceeds the best fixed strategy by at most"
        f" {p['StRlMaxGain']}; it chose gamma = 0 in {p['StRlGammaZero']} commitments and learned to lie in (almost)"
        f" every state: always lying is the best response (Proposition 7). Damage is affine in u (min R^2"
        f" {p['StLinRtwo']}).",
        f"B = 200, majority: wrong answers used {p['StBigWrongUzero']} (u = 0) -> {p['StBigWrongUone']} (u = 1);"
        f" B = 50, u = 1: {p['StSmallWrongUone']}. Stackelberg (commitments that keep Theorem 1, majority): auditing"
        f" everything pays below a price per query of {p['StPriceSmall']} (B = 50) and {p['StPriceBig']} (B = 200)"
        " wrong answers, auditing nothing above.",
        f"Auditing anchors too (B = 200, u = 1): wrong answers used {p['StAaWrong']}, but honest agents without an"
        f" honest anchor rise from {p['StNoAaNoAnchor']} to {p['StAaNoAnchor']} (Theorem 1 bound {p['StBoundBig']})"
        f" and {p['StAaHonestBlocks']} honest pairs are blocked: colluding anchors evict honest ones.",
        f"Identity game (N = 200, rho = 0.1): the attacker's best response at a = 3 is B = {p['IgBNOneRho']} N;"
        f" from a = {p['IgAnchorsDeter']} anchors no attack pays.",
        "",
        "## 12. Pareto analysis (4 objectives)",
        "",
        f"{p['PaConfigs']} configurations per threat level; Pareto-optimal: {p['PaFrontSmall']} (B = 50),"
        f" {p['PaFrontBig']} (B = 200). Knee at B = 200: {p['PaKneeBig']} (damage {p['PaKneeBigDamage']},"
        f" {p['PaKneeBigCost']} queries/task, {p['PaKneeBigNoAnchor']} without an honest anchor)."
        f" Knee among configurations that keep Theorem 1: {p['PaKneeSafe']} (damage {p['PaKneeSafeDamage']},"
        f" {p['PaKneeSafeCost']} queries/task, {p['PaKneeSafeNoAnchor']} without an honest anchor)."
        f" Least damage of any configuration at B = 200: {p['PaMinDamageBig']}."
        f" Share of front points that audit anchors: {p['PaFrontAaShare']}.",
        "",
        "## 13. Dense subgraphs in ERC-8004",
        "",
        f"Greedy peeling {p['DsGreedy']} vs exact (max-flow) {p['DsExact']} on Ethereum. Densest Base block:"
        f" {p['DsBaseTopRaters']} raters x {p['DsBaseTopAgents']} agents, pair density {p['DsBaseTopDensity']},"
        f" z = {p['DsBaseTopZ']} against degree-preserving random graphs.",
        f"Rated agents inside significant blocks (z > 3): Base {p['DsBaseSigDens']} (density) / {p['DsBaseSigW']}"
        f" (Fraudar weights); Ethereum {p['DsEthSigDens']} / {p['DsEthSigW']}. The 'miner-vouch' block carries"
        f" {p['DsMinerShare']} of Base feedback at pair density {p['DsMinerDensity']}.",
        f"Planted Sybil blocks: recall {p['DsRecallLow']} at pair density 0.05, {p['DsRecallTwo']} at 0.2,"
        f" {p['DsRecallFive']} at 0.5.",
        "",
        "## 14. Digital twin of ERC-8004",
        "",
        "| | Ethereum | Base |",
        "|---|---|---|",
        f"| registered agents | {p['TwEthRegistered']} | {p['TwBaseRegistered']} |",
        f"| next-week activity AUC: twin / recent / popularity | {p['TwEthAucTwin']} / {p['TwEthAucRecent']} /"
        f" {p['TwEthAucPop']} | {p['TwBaseAucTwin']} / {p['TwBaseAucRecent']} / {p['TwBaseAucPop']} |",
        f"| identities for a full top-5 eclipse, reputation discovery (median tag) | {p['TwEthRepMedian']} |"
        f" {p['TwBaseRepMedian']} |",
        f"| same with 3 anchors, P = 0.5 | {p['TwEthAncNeeded']} | {p['TwBaseAncNeeded']} |",
        f"| boosted top-5 entries: AUC of low cost / of activity | {p['TwEthAucCost']} / {p['TwEthAucActivity']} |"
        f" {p['TwBaseAucCost']} / {p['TwBaseAucActivity']} |",
        "",
        f"Live sync: median lag {p['TwSyncLag']} s behind the chain head, {p['TwSyncSeconds']} s per sync.",
        "",
        "## 15. Robustness and tightness",
        "",
        f"20 seeds, B = N, a = 3: {p['RbTwentyEclK']} target instances eclipsed ({p['RbTwentyEcl']}), exact"
        f" {p['RbTwentyExact']}, bound {p['RbTwentyBound']} (p = {p['RbTwentyP']}); {p['RbTwentyNoAnchorPrecise']}"
        f" of honest agents without an honest anchor. B = 50: {p['RbTwentySmallEcl']} eclipsed.",
        f"Sensitivity: {p['SensConfigs']} settings; no defense eclipsed at least {p['SensNoneEcl']} of rounds;"
        f" {p['SensNoAnchorMin']}-{p['SensNoAnchorMax']} without an honest anchor.",
        f"Tightness (Proposition 1b): largest |measured - exact| over study 3: {p['TightMaxDev']}."
        f" Dynamic programming: {p['DpSettings']} settings, optimal minus always-lie {p['DpMaxGap']},"
        f" Monte Carlo deviation {p['DpMcMaxDev']}.",
        "",
        "## 16. Real agent cards",
        "",
        f"{p['RcCards']} unique real ERC-8004 descriptions; adaptive Sybil cards match tasks with cosine"
        f" {p['RcAdaptCos']} vs {p['RcHonestCos']} for the best honest card.",
        f"Embedding selector, adaptive cards: B = 50 share {p['RcAdaptShareSmall']}, eclipsed {p['RcAdaptEclSmall']}"
        f" -> anchors {p['RcOursEclSmall']}; B = 200 share {p['RcAdaptShareBig']}, eclipsed {p['RcAdaptEclBig']} ->"
        f" anchors {p['RcOursEclBig']} ({p['RcOursNoAnchorBig']} without an honest anchor). Static cards, B = 200:"
        f" eclipsed {p['RcStaticEclBig']}. Quality {p['RcQualOurs']} vs {p['RcQualNone']}.",
        f"LLM selector on real cards (N = 20, B = 10): adaptive {p['RcLlmAdaptNone']} -> {p['RcLlmAdaptOurs']};"
        f" static {p['RcLlmStaticNone']} -> {p['RcLlmStaticOurs']}; min honest peers {p['RcLlmMinNone']} ->"
        f" {p['RcLlmMinOurs']}.",
        "",
        "## 20. Sharing evidence (worst case over attackers, false reports on)",
        "",
        f"Naive/threshold sharing blocks {p['ShNaiveFramed']} honest agents; alerts block at most {p['ShAlertsFramed']}.",
        f"Alerts + committees: B = 25 {p['ShWorstLocalSmall']} -> {p['ShWorstBestSmall']}; B = 50"
        f" {p['ShWorstLocalMid']} -> {p['ShWorstBestMid']} (probes: {p['ShProbeWorstSmall']}, {p['ShProbeWorstMid']}).",
        f"Above the threshold (quorum 2/3): {p['ShFramedHundred']} honest blocked at B = 100, {p['ShFramedTwoHundred']}"
        f" at B = 200; quorum 0.9 with 45 seats at B = 200: {p['ShQuorumSafeBig']}.",
        f"Design history (RL attacker, B = 50 unless stated): probes + one-sided {p['ShVOneRl']} ({p['ShVOneMuted']}"
        f" honest reporters muted; always lie {p['ShVOneAlways']}); probes + two-sided {p['ShProbeRl']} (always lie"
        f" {p['ShProbeAlways']}); probes + pooling, B = 25: {p['ShProbePoolRl']} ({p['ShProbePoolMuted']} muted);"
        f" transcripts + pooling: {p['ShPoolRlSmall']} (B = 25), {p['ShPoolRl']} (B = 50). Transcripts without pooling"
        f" (5 committees/round): the RL attacker lies in {p['ShRlLieFrac']} of answers, i.e. learns always-lie; its"
        f" runs give {p['ShRlCommitteeRange']} (mean {p['ShRlCommittee']}; fixed always-lie runs {p['ShRlAlways']})"
        " because committee-queue congestion varies between runs.",
        f"Final design with pooling (10 committees/round): worst {p['ShPoolWorstSmall']} (B = 25), {p['ShPoolWorst']}"
        f" (B = 50); without pooling {p['ShNoPoolWorstSmall']}, {p['ShNoPoolWorst']}; budget Sybils"
        f" {p['ShBudgetPool']} vs {p['ShBudgetLocal']} without sharing; 5 committees/round: always lie"
        f" {p['ShPoolCapFiveAlways']}; at most {p['ShPoolHonestBlocked']} honest blocked in a run (B <= 50).",
        f"On-chain blocklist: {p['BlGasFifteen']}M gas per judgement (15 seats), {p['BlGasFortyFive']}M (45);"
        f" draws with the accused or a duplicate: {p['BlDrawsBad']}. Real registries: committees impossible from"
        f" {p['TcShareLimit']} Sybils ({p['TcEthLimit']} on Ethereum, {p['TcBaseLimit']} on Base); seats needed at 10%:"
        f" {p['TcSeatsTen']} ({p['TcGasTen']}M gas), at 20%: {p['TcSeatsTwenty']}.",
        "",
        "## 22. Verifiable answers (worst case over four attackers)",
        "",
        f"B = 200: {p['VfLocalBig']} without verification -> {p['VfWorstBigFive']} at v = 0.05 and {p['VfWorstBigTwenty']}"
        f" at v = 0.2 (proof ban + verified reputation); always-lying Sybils at v = 0.05: {p['VfBanBig']},"
        f" {p['VfAllBanned']} of 200 banned, {p['VfCostBig']} verifications per task. B = 400: {p['VfLocalHuge']} ->"
        f" {p['VfWorstHugeTwenty']} at v = 0.2. Sleepers at B = 200: {p['VfSleeperBig']} (v = 0.05),"
        f" {p['VfSleeperBigHigh']} (v = 0.2). Honest agents banned: at most {p['VfHonestBans']} per run. RL attacker"
        f" (B = 200, v = 0.05): {p['VfRlBig']}.",
        f"Deposits: without anchors {p['DpRepEth']} identities take all top-5 places (Ethereum); with 3 anchors an eclipse"
        f" with probability 1/2 needs {p['DpAncEth']}; deterring deposit {p['DpRatioEth']}x smaller (Base"
        f" {p['DpRatioBase']}x).",
        "",
        "## 17. Real questions, real answers (MMLU, Qwen2.5-3B)",
        "",
        f"Honest accuracy: expert {p['RaExpertAcc']}, non-specialist {p['RaGenericAcc']}; wrong answers used without"
        f" Sybils: {p['RaHonestOnly']}.",
        f"200 always-lying Sybils: none {p['RaNoneBig']}, anchors {p['RaAnchorsBig']}, second opinions u = 1"
        f" {p['RaSoBig']}, verification v = 0.2 {p['RaVerifyBig']} (all Sybils banned; on average {p['RaHonestBanned']}"
        f" honest agents banned). 50 Sybils: none {p['RaNoneMid']}, second opinions {p['RaSoMid']}, committees"
        f" {p['RaCommMid']}, verification {p['RaVerifyMid']}; 25 Sybils, committees {p['RaCommSmall']}.",
        f"Sybils lying 30% answer better than honest non-specialists: {p['RaLowRateNone']} wrong answers used without a"
        f" defense at B = 50 (committees {p['RaLowRateComm']}).",
        "",
        f"Total simulation runs: {p['TotalRuns']}.",
        "",
    ]
    tables = ROOT / "tables"
    for name in ["attack", "defenses", "anchors", "adaptive", "llm_local", "prop2", "erc8004_structure",
                 "moltbook_structure"]:
        f = tables / f"{name}.md"
        if f.exists():
            lines += [f"## Detail: {name}", "", f.read_text(encoding="utf-8"), ""]
    (tables / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    m: dict[str, str] = {}
    total = (attack(m) + prop2(m) + defenses(m) + anchors(m) + adaptive_attacks(m) + audit(m) + robust_audit(m)
             + llm(m) + strategic(m) + pareto(m) + robustness(m) + real_cards(m) + shared(m) + verifiable(m))
    real_data(m)
    onchain(m)
    dense(m)
    twin(m)
    m["TotalRuns"] = f"{total:,}".replace(",", "{,}")
    missing = sorted(k for k, v in m.items() if v == NA)
    out = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())) + "\n"
    PAPER.mkdir(exist_ok=True)
    (PAPER / "numbers.tex").write_text(out, encoding="utf-8")
    write_results_md(m)
    print(out)
    if missing:
        print("MISSING:", ", ".join(missing))


if __name__ == "__main__":
    main()
