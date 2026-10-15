"""Paper figures and tables from results/. Missing inputs are skipped."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
TAB = ROOT / "tables"
DEF_ORDER = ["none", "degree_bound", "random_mix", "sybilrank", "sumup", "proposed"]
DEF_LABEL = {"none": "no defense", "degree_bound": "degree bound", "random_mix": "random mix",
             "sybilrank": "SybilRank", "sumup": "SumUp", "proposed": "anchors (ours)"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def theorem1_bound(n_honest: int, n_sybils: int, a: int, beta: float = 0.0) -> float:
    p = beta + (1 - beta) * n_sybils / (n_honest + n_sybils - a)
    return p ** a


def _save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=200)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("figure", name)


def _table(df: pd.DataFrame, name: str) -> None:
    TAB.mkdir(exist_ok=True)
    df.to_csv(TAB / f"{name}.csv")
    (TAB / f"{name}.md").write_text(df.to_markdown(), encoding="utf-8")
    print("table", name)


def fig_attack() -> None:
    p = RES / "1_attack" / "summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    s85 = s[s["adversary.service_quality"] == 0.85]
    for ax, adaptive in zip(axes, [False, True]):
        d = s85[s85["adversary.adaptive_cards"] == adaptive]
        piv = d.pivot_table(index="adversary.n_sybils", columns="adversary.attractiveness",
                            values="tail_target_sybil_share")
        im = ax.imshow(piv.values, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.set_xlabel("card attractiveness" if not adaptive else "card attractiveness (ignored: per-query cards)")
        ax.set_title("static cards" if not adaptive else "query-adaptive cards")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.5 else "black")
    axes[0].set_ylabel("Sybil identities")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Sybil share of target's peers")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "attack_heatmap.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "attack_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("figure attack_heatmap")
    tab = s.pivot_table(index=["adversary.adaptive_cards", "adversary.service_quality", "adversary.n_sybils"],
                        columns="adversary.attractiveness",
                        values=["tail_target_sybil_share", "tail_targets_eclipsed_frac"]).round(3)
    _table(tab, "attack")


def fig_defenses() -> None:
    p = RES / "2_defenses" / "summary.csv"
    if not p.exists():
        return
    s_all = pd.read_csv(p)
    s = s_all[s_all["adversary.adaptive_cards"] == True]  # noqa: E712  (the figure shows the adaptive attacker)
    metrics = [("tail_target_sybil_share", "Sybil share of target's peers"),
               ("tail_targets_eclipsed_frac", "fraction of rounds targets eclipsed"),
               ("tail_mean_quality", "mean service quality (honest view)")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, (m, label) in zip(axes, metrics):
        for d in DEF_ORDER:
            g = s[s["defense.kind"] == d].groupby("adversary.n_sybils")[m].agg(["mean", "std"])
            if g.empty:
                continue
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", ms=3, capsize=2, label=DEF_LABEL[d],
                        lw=2.2 if d == "proposed" else 1.0)
        ax.set_xscale("log")
        ax.set_xlabel("Sybil identities (N = 200 honest)")
        ax.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, "defenses")
    tab = s_all.pivot_table(index=["adversary.adaptive_cards", "adversary.n_sybils"], columns="defense.kind",
                            values=["tail_target_sybil_share", "tail_targets_eclipsed_frac",
                                    "tail_target_min_honest_neighbors", "tail_mean_quality"]).round(3)
    _table(tab, "defenses")


def fig_anchors() -> None:
    p = RES / "3_anchor_bound" / "summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    n_honest = 200
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for a in sorted(s["defense.params.n_anchors"].unique()):
        g = s[s["defense.params.n_anchors"] == a].groupby("adversary.n_sybils")["tail_honest_no_honest_anchor_frac"].mean()
        line = ax.plot(g.index, g.values, marker="o", ms=3, label=f"a = {a} (measured)")
        bound = [theorem1_bound(n_honest, b, a) for b in g.index]
        ax.plot(g.index, bound, ls="--", color=line[0].get_color(), lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sybil identities (N = 200 honest)")
    ax.set_ylabel("honest agents with no honest anchor")
    ax.legend(frameon=False, fontsize=7, title="dashed: Theorem 1 bound", title_fontsize=7)
    _save(fig, "anchors_bound")
    tab = s.pivot_table(index="adversary.n_sybils", columns="defense.params.n_anchors",
                        values=["tail_honest_no_honest_anchor_frac", "tail_target_sybil_share", "tail_mean_quality"]).round(4)
    _table(tab, "anchors")


def fig_adaptive() -> None:
    d = RES / "4_adaptive_attacks"
    p = d / "summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    import json
    runs = []
    for f in sorted((d / "runs").glob("run_*.csv")):
        cfg = json.loads(f.with_suffix(".json").read_text())
        runs.append((cfg, pd.read_csv(f)))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    # left: before the sleeper switch (rounds 55-74), anchors only, vs. Theorem 1 with bias beta
    for bias in sorted(s["adversary.registry_bias"].unique()):
        rows = {}
        for cfg, df in runs:
            a_ = cfg["adversary"]
            if cfg["defense"]["kind"] == "proposed" and a_["registry_bias"] == bias:
                v = df[(df["round"] >= 55) & (df["round"] < 75)]["honest_no_honest_anchor_frac"].mean()
                rows.setdefault(a_["n_sybils"], []).append(v)
        bs = sorted(rows)
        line = axes[0].plot(bs, [np.mean(rows[b]) for b in bs], marker="o", ms=3, label=f"β = {bias}")
        axes[0].plot(bs, [theorem1_bound(200, b, 3, bias) for b in bs], ls="--", lw=0.9, color=line[0].get_color())
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Sybil identities (N = 200 honest)")
    axes[0].set_ylabel("honest agents with no honest anchor")
    axes[0].legend(frameon=False, fontsize=7, title="dashed: bound with bias β", title_fontsize=7)
    # right: sleeper timeline, B = 200, beta = 0, mean over seeds
    for kind, color in [("none", "tab:red"), ("proposed", "tab:green")]:
        dfs = [df for cfg, df in runs if cfg["adversary"]["n_sybils"] == 200
               and cfg["adversary"]["registry_bias"] == 0.0 and cfg["defense"]["kind"] == kind]
        if dfs:
            mean = pd.concat(dfs).groupby("round")["target_sybil_share"].mean()
            axes[1].plot(mean.index, mean.values, lw=1.2, color=color, label=DEF_LABEL[kind])
    axes[1].axvline(75, color="grey", ls=":", lw=1)
    axes[1].text(77, 0.08, "Sybils switch\nto bad service", fontsize=7)
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("Sybil share of targets' peers")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    _save(fig, "adaptive")
    tab = s.pivot_table(index="adversary.n_sybils", columns="adversary.registry_bias",
                        values=["tail_honest_no_honest_anchor_frac", "tail_target_sybil_share"]).round(4)
    _table(tab, "adaptive")


def _llm_bars(s: pd.DataFrame, model_col: str | None, name: str) -> None:
    metrics = [("tail_target_sybil_share", "Sybil share of targets' peers"),
               ("tail_target_min_honest_neighbors", "min honest peers of a target")]
    groups = [(m, d) for m in ([None] if model_col is None else sorted(s[model_col].unique()))
              for d in ["none", "proposed"]]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, (metric, label) in zip(axes, metrics):
        x, pos, labels = 0.0, [], []
        for adaptive in [False, True]:
            for model, d in groups:
                sel = (s["adversary.adaptive_cards"] == adaptive) & (s["defense.kind"] == d)
                if model_col:
                    sel &= s[model_col] == model
                v = s.loc[sel, metric]
                if v.empty:
                    continue
                color = "tab:green" if d == "proposed" else "tab:red"
                ax.bar(x, v.mean(), yerr=v.std(), color=color, alpha=0.8, capsize=2)
                pos.append(x)
                labels.append(f"{'adaptive' if adaptive else 'static'} cards\n{DEF_LABEL[d]}"
                              + (f"\n{model}" if model else ""))
                x += 1.0
            x += 0.5
        ax.set_xticks(pos, labels, fontsize=6)
        ax.set_ylabel(label)
    _save(fig, name)


def fig_llm() -> None:
    p = RES / "6_llm_selector" / "summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        _llm_bars(s, None, "llm_local")
        tab = s.pivot_table(index="adversary.adaptive_cards", columns="defense.kind",
                            values=["tail_target_sybil_share", "tail_target_min_honest_neighbors",
                                    "tail_honest_mean_sybil_share", "tail_mean_quality"],
                            aggfunc=["mean", "std"]).round(3)
        _table(tab, "llm_local")
    p = RES / "9_llm_colab" / "summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        _table(s.groupby(["model", "adaptive", "defense"]).mean(numeric_only=True).round(3), "llm_colab")


def fig_audit() -> None:
    p = RES / "5_second_opinions" / "summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    s = s[(s["adversary.n_sybils"] == 200) & (s["defense.kind"] == "proposed")]
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for r in sorted(s["adversary.answer_wrong_prob"].unique()):
        g = s[s["adversary.answer_wrong_prob"] == r].groupby("defense.params.audit_prob")[
            ["tail_wrong_used_rate", "final_audit_calls"]].mean()
        x = g["final_audit_calls"] / (200 * 100)
        ax.plot(x, g["tail_wrong_used_rate"], marker="o", ms=3, label=f"Sybil lying prob. {r:g}")
        for u, xi, yi in zip(g.index, x, g["tail_wrong_used_rate"]):
            ax.annotate(f"u={u:g}", (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("extra anchor queries per task")
    ax.set_ylabel("wrong answers used by honest agents")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, "second_opinions")


def fig_onchain() -> None:
    g, r = RES / "10_onchain_sampler" / "gas.csv", RES / "10_onchain_sampler" / "grinding.csv"
    if not (g.exists() and r.exists()):
        return
    gas, grind = pd.read_csv(g), pd.read_csv(r)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
    d = gas[gas["registry_size"] == gas["registry_size"].max()]
    axes[0].plot(d["anchors"], (d["request_gas"] + d["callback_gas"]) / 1e3, "o-", label="VRF sampler (request + callback)")
    axes[0].plot(d["anchors"], d["blockhash_gas"] / 1e3, "s--", label="block-hash sampler (unsafe)")
    axes[0].set_xlabel("anchors a")
    axes[0].set_ylabel("gas (thousands)")
    axes[0].set_title("gas (same for N = 1e2 .. 1e9)")
    axes[0].legend(frameon=False, fontsize=7)
    sub = grind[grind["sybils"] == 50]
    for a, gg in sub.groupby("anchors"):
        x = gg["rerolls"] + 1
        line, = axes[1].plot(x, gg["prop6_prediction"], "-", lw=1)
        axes[1].plot(x, gg["p_no_honest_anchor"], "o", color=line.get_color(), ms=3, label=f"a = {a}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("seeds the adversary can try (r + 1)")
    axes[1].set_ylabel("P(no honest anchor)")
    axes[1].set_title("grinding, B = 50, N = 200")
    axes[1].legend(frameon=False, fontsize=7)
    _save(fig, "onchain")


def fig_strategic() -> None:
    p = RES / "11_strategic_attacker" / "summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    lab = {"lie0.3": "lie 30%", "lie1.0": "always lie", "budget": "budget (knows defense)", "rl": "RL (Q-learning)"}
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7), sharey=True)
    for ax, b in zip(axes, [50, 200]):
        d = s[(s["n_sybils"] == b) & (s["audit_rule"] == "majority") & (~s["audit_anchors"].astype(bool))]
        for att in ["lie0.3", "lie1.0", "budget", "rl"]:
            m = d[d["attacker"] == att].groupby("audit_prob")["wrong_used"].agg(["mean", "sem"])
            ax.errorbar(m.index, m["mean"], yerr=m["sem"], marker="o" if att != "rl" else "x", ms=4,
                        ls="-" if att != "rl" else ":", label=lab[att])
        ax.set_xlabel("audit rate u")
        ax.set_title(f"B = {b}")
    axes[0].set_ylabel("wrong answers used")
    axes[1].legend(frameon=False, fontsize=7)
    _save(fig, "strategic")


def fig_pareto() -> None:
    p = RES / "12_pareto" / "config_means.csv"
    f = RES / "12_pareto" / "pareto_front.csv"
    if not (p.exists() and f.exists()):
        return
    m, front = pd.read_csv(p), pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, b in zip(axes, [50, 200]):
        d, fr = m[m["adversary.n_sybils"] == b], front[front["adversary.n_sybils"] == b]
        sc = ax.scatter(d["cost"], d["damage"], c=d["defense.params.n_anchors"], cmap="viridis", s=10, alpha=0.6)
        ax.scatter(fr["cost"], fr["damage"], facecolors="none", edgecolors="red", s=30, lw=0.8,
                   label="Pareto-optimal (3 objectives)")
        ax.set_xlabel("extra queries per task")
        ax.set_title(f"B = {b}")
    axes[0].set_ylabel("wrong answers used")
    axes[1].legend(frameon=False, fontsize=7)
    fig.colorbar(sc, ax=axes, label="anchors a", shrink=0.8)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "pareto.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "pareto.pdf", bbox_inches="tight")
    plt.close(fig)
    print("figure pareto")


def fig_dense() -> None:
    p = RES / "13_dense_subgraph" / "injection.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
    for ax, chain in zip(axes, ["ethereum", "base"]):
        for (meth, camo), g in d[d["chain"] == chain].groupby(["method", "camouflage"]):
            m = g.groupby("density")["recall"].mean()
            ax.plot(m.index, m.values, "o-" if camo == 0 else "s--", ms=3,
                    label=f"{'Fraudar weights' if meth == 'weighted' else 'density'}, camouflage {camo}")
        ax.set_xscale("log")
        ax.set_xlabel("pair density of the planted Sybil block")
        ax.set_title(f"ERC-8004 {chain.capitalize()}")
    axes[0].set_ylabel("recall of planted nodes")
    axes[1].legend(frameon=False, fontsize=6)
    _save(fig, "dense_injection")


def fig_twin() -> None:
    base = RES / "14_digital_twin"
    chains = [c for c in ("ethereum", "base") if (base / c / "risk_over_time.csv").exists()]
    if not chains:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for chain in chains:
        r = pd.read_csv(base / chain / "risk_over_time.csv")
        g = r.groupby("t").agg(rep=("reputation_identities", "median"), anc=("anchors_identities", "median"))
        days = (g.index - g.index.min()) / 86400
        line, = axes[0].plot(days, g["rep"], "-", label=f"{chain}: reputation top-5 (median tag)")
        axes[0].plot(days, g["anc"], "--", color=line.get_color(), label=f"{chain}: with 3 anchors, P = 0.5")
    a = pd.read_csv(base / chains[-1] / "activity_backtest.csv")
    names = {"twin_logistic": "twin model", "recent_7d": "recent activity", "popularity": "popularity"}
    for model in ["twin_logistic", "recent_7d", "popularity"]:
        gg = a[a["model"] == model]
        axes[1].plot((gg["t"] - gg["t"].min()) / 86400, gg["auc"], marker=".", lw=1, label=names[model])
    axes[1].set_title(f"{chains[-1].capitalize()}: weekly backtest")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("days since first origin")
    axes[0].set_ylabel("identities for a full eclipse")
    axes[0].legend(frameon=False, fontsize=6)
    axes[1].set_xlabel("days since first origin")
    axes[1].set_ylabel("AUC, next-week activity")
    axes[1].legend(frameon=False, fontsize=6)
    _save(fig, "twin")


def fig_shared() -> None:
    p = RES / "20_shared_defense" / "main.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d["false_reports"] == 1]
    lab = {"local": "local only", "naive": "naive", "threshold": "threshold", "alerts": "alerts",
           "committee": "committees", "alerts+committee": "alerts + committees"}
    style = {"local": "k-", "naive": "r:", "threshold": "m:", "alerts": "b--", "committee": "g-.",
             "alerts+committee": "g-"}
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    worst = d.groupby(["share", "n_sybils", "attacker"])["wrong_used"].mean().unstack("attacker").max(axis=1)
    framed = d.groupby(["share", "n_sybils"])["global_blocks_honest"].mean()
    marks = {"local": "o", "naive": "v", "threshold": "^", "alerts": "s", "committee": "D", "alerts+committee": "o"}
    offset = {"naive": -0.06, "threshold": 0.06}
    for share in lab:
        w = worst.loc[share]
        x = np.arange(len(w)) + offset.get(share, 0.0)
        axes[0].plot(x, w.values, style[share], marker=marks[share], ms=3, label=lab[share])
        f = framed.loc[share]
        axes[1].plot(x, f.values, style[share], marker=marks[share], ms=3, label=lab[share])
    for ax in axes:
        ax.set_xticks(range(4), ["25", "50", "100", "200"])
        ax.set_xlabel("Sybil identities B (N = 200)")
    axes[0].set_ylabel("wrong answers used (worst attacker)")
    axes[1].set_ylabel("honest agents blocked network-wide")
    axes[1].legend(frameon=False, fontsize=6)
    _save(fig, "shared")


def fig_verify() -> None:
    p = RES / "22_verifiable" / "summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    s = s[s["attacker"] != "rl"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7), sharey=True)
    for ax, mode in zip(axes, ["correct", "ban+rep"]):
        for b, style in [(50, "o-"), (200, "s-"), (400, "^-")]:
            d = s[(s["n_sybils"] == b) & ((s["mode"] == mode) | (s["verify_prob"] == 0.0))]
            w = d.groupby(["verify_prob", "attacker"])["wrong_used"].mean().unstack("attacker").max(axis=1)
            ax.plot(w.index, w.values, style, ms=3, label=f"B = {b}")
        ax.set_xlabel("fraction of answers verified v")
        ax.set_title("verified answers corrected only" if mode == "correct" else "+ proof ban and verified reputation")
    axes[0].set_ylabel("wrong answers used (worst attacker)")
    axes[1].legend(frameon=False, fontsize=7)
    _save(fig, "verify")


def main() -> None:
    fig_attack()
    fig_defenses()
    fig_anchors()
    fig_adaptive()
    fig_audit()
    fig_llm()
    fig_onchain()
    fig_strategic()
    fig_pareto()
    fig_dense()
    fig_twin()
    fig_shared()
    fig_verify()


if __name__ == "__main__":
    main()
