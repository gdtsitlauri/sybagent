# SybAgent

**Can a handful of fake agents cut an AI agent off from every honest peer, and how do we stop it?**

AI agents increasingly discover and hire other agents on open networks: A2A directories, on-chain identity and
reputation standards such as ERC-8004, agent-only social networks. They usually choose collaborators
*semantically*. An LLM or an embedding search reads each candidate's self-advertised description and ranks it.

SybAgent shows that this makes **eclipse attacks** cheap. A few Sybil identities whose descriptions win the
semantic comparison can isolate a target agent from every honest peer, independently of network size. We propose
**semantic-blind anchors** as a defense, prove guarantees for it, and evaluate it in simulation, with a real LLM
selector and against real reputation data. We then deploy the sampler as a smart contract, attack the defense with a
learning adversary and solve the resulting Stackelberg games, tune it by multi-objective (Pareto) analysis, look for
colluding raters with dense-subgraph detection, and build a live digital twin of ERC-8004 that estimates eclipse
costs on the real network.

The defense has three parts:
- **Anchors.** Each agent reserves a few slots for peers sampled uniformly *without reading their descriptions*.
- **Sticky slots.** Anchors are pinned against recency eviction and replaced only on recent bad service.
- **Second opinions.** Anchors also serve as auditors: their majority answer can check, and correct, the answers of
  other peers.

| # | study | question | scale |
|---|---|---|---|
| 1 | attack | how many Sybils eclipse a target, static vs query-adaptive descriptions | 480 runs |
| 2 | defenses | anchors vs degree bound, random mix, SybilRank, SumUp | 300 runs |
| 3 | anchor bound | Theorem 1 measured against its bound, a = 1..4 anchors | 80 runs |
| 4 | adaptive attacks | identity flooding, biased sampling, sleeper Sybils | 120 runs |
| 5 | second opinions | wrong answers used vs audit cost | 240 runs |
| 5b | colluding and category anchors | coordinated lies, majority vs unanimity, anchors by skill category | 160 runs |
| 6 | LLM selector | the attack and the defense with Qwen2.5-3B choosing collaborators | 12 runs |
| 7 | theory check | Proposition 2: O(m) Sybils eclipse independently of N | 40 runs |
| 8 | real data | ERC-8004 (Ethereum, Base) and Moltbook graphs vs Sybil-defense assumptions | 571k feedback events, 34k agents |
| 10 | on-chain sampler | anchors sampled by a VRF smart contract: gas, uniformity, front-running, grinding | local EVM, 2,000 draws |
| 11 | strategic attacker | RL (Q-learning) and knowledge-based attackers vs audit commitments; Stackelberg games | 1,440 runs |
| 12 | Pareto | four objectives over anchors, audit rate, rule, anchor audits | 600 runs |
| 13 | dense subgraphs | coordinated raters in ERC-8004 (Charikar, Fraudar, exact max-flow, null model, planted blocks) | 2 chains |
| 14 | digital twin | live mirror of ERC-8004, forecasts, eclipse cost, what-if, backtests, committee planning | 152k agents, 30+ weekly origins |
| 15 | robustness | 20 seeds for the key results; sensitivity to m, k, dimension, selector, temperature | 278 runs |
| 16 | real agent cards | the attack and the defense on 1,217 real ERC-8004 descriptions, LLM-written tasks and Sybil cards; embedding and LLM selectors | 72 runs |
| 20 | sharing evidence | naive, threshold, alerts, anchor committees; false reports, strategic votes, RL attacker; quorum threshold; evidence pooling | ~880 runs |
| 21 | on-chain blocklist | committee contract: gas, rules, committee draws | local EVM |
| 22 | verifiable answers | proof-carrying bans, verified reputation, sleepers, RL; B up to 2N | 474 runs |
| 17 | real answers | 360 MMLU questions answered by Qwen2.5-3B as expert / non-specialist; all defenses | ~130 runs |

## Main findings

All numbers come from `tables/RESULTS.md`. N = 200 honest agents and 5 seeds unless stated.

1. **Semantic selection is eclipsed with a constant number of identities.**
   - Proposition 2: with query-adaptive descriptions, O(m) Sybils suffice, where m is the working-set size,
     whatever the network size. In simulation, 8 Sybils eclipse a target in round 0 for every N from 100 to 800.
   - With 200 Sybils, a target's peers are 100% Sybil with adaptive descriptions, and 87% with static ones (eclipsed in
     26% of rounds).
   - Even descriptions with no attractiveness at all take 62% of a target's peers through sheer numbers.
2. **Classical Sybil and eclipse defenses do not help here.**
   - With 200 adaptive Sybils, degree bounds, random mixing, SybilRank and SumUp all leave targets eclipsed in 100% of
     rounds.
   - Real data shows why. On ERC-8004, 92% (Ethereum) and 54% (Base) of rated agents have a single reviewer.
   - The review graphs mix 275x and 4,482x slower than the log n that random-walk defenses assume. On Moltbook the
     factor is 39x.
3. **Semantic-blind anchors restore a provable guarantee (Theorem 1).**
   - The probability that an agent has no honest anchor is at most `(B/(N+B-a))^a`, whatever the descriptions say.
     An eclipse therefore needs Theta(a N) identities instead of O(m).
   - The bound is nearly tight: measured values stay within two standard errors in 16/16 configurations.
   - With up to 50 adaptive Sybils, targets are eclipsed in 0% of rounds (93% without a defense).
   - With 200 Sybils (B = N), 24% of target instances are eclipsed (6/25, consistent with the 13% bound,
     binomial p = 0.09). Over all honest agents, 12% have no honest anchor.
   - The cost in service quality is 0.848 vs 0.850.
4. **Adaptive attacks on the defense behave as the theory says.**
   - Flooding with 1,600 identities (8N) leaves 68% of agents without an honest anchor (bound 71%).
   - Biased sampling (beta = 0.5) raises 12% to 39% (bound 43%). Uniform sampling is a real requirement.
   - Sleeper Sybils that turn bad are evicted by recency-weighted eviction: the targets' Sybil share drops from 85% to
     62% and agents without an honest anchor from 12% to 0%.
5. **Second opinions bound damage per Sybil and lower wrong answers by correction (Proposition 3).**
   - Sybils lying 30% of the time give 30% wrong answers used without a defense, 28% with anchors, 22% with
     second opinions at u = 0.3 (1.9 extra queries per task) and 8.8% at u = 1 (6.4 extra queries per task).
   - No honest peer was ever blocked by mistake.
6. **A unanimity rule stops colluding anchors (Proposition 4); anchors by skill category do not pay off
   (Proposition 5).** Study 5b uses coordinated Sybils and skill-dependent honest accuracy.
   - Unanimity cuts bad corrections (right answers flipped to wrong) from 8.7 to 0.4 per round. It also cuts good
     corrections, from 28 to 10 per round, so wrong answers used move from 28% to 29%.
   - No honest peer was blocked in any configuration.
   - Category anchors raise agents without an honest anchor from 12% to 43% (two of three) and 83% (three of three),
     because the adversary claims every category.
7. **The attack and the defense hold with a real LLM choosing collaborators** (Qwen2.5-3B).
   - With adaptive descriptions, the targets' Sybil share is 100% without a defense and 67% with anchors. The minimum
     number of honest peers of a target goes from 0.0 to 0.7.
   - With static descriptions the share is 55% without a defense and 42% with anchors.
8. **The sampler can be secured on-chain (study 10, Proposition 6).**
   - `contracts/AnchorSampler.sol` draws anchors from a VRF output with a registry snapshot and one pending request
     per agent. With a = 3 a request costs 151,458 gas and the callback 106,973, the same for 10^2 to 10^9 agents.
   - 2,000 on-chain draws match the simulator's sampler bit for bit and are uniform (chi-square p = 0.84). Late
     identities were never drawn (0/200) and re-requests are rejected.
   - A block-hash seed can be ground: with B = 50, N = 200, a = 3, choosing the best of 101 seeds raises the chance of
     no honest anchor from 0.8% to 55%, exactly as Proposition 6 predicts.
9. **A learning attacker can do no better than always lying (study 11, Propositions 7 and 8).**
   - Against 40 audit commitments, the RL attacker never beats the best fixed strategy by more than 0.006. It chose a
     myopic discount in 36/40 cases and learned to lie in every state.
   - Damage is affine in the audit rate (R^2 >= 0.997), so the defender's Stackelberg optimum is all or nothing:
     audit every answer while a query costs less than 0.13 (B = 50) or 0.055 (B = 200) wrong answers.
   - In the identity game, from a = 5 anchors no attack pays (rho = 0.1), for any network size.
10. **Anchors must never audit anchors (studies 11 and 12).** Cross-checking anchors lowers wrong answers used (57% to
    51% at B = 200, u = 1), but colluding Sybil anchors evict honest ones: agents without an honest anchor rise from
    13% to 48%, far above Theorem 1's bound. The Pareto knee that keeps Theorem 1 is a = 6, u = 0.6, majority.
11. **Collusion detection finds dense groups only (study 13).** On Base, 32 raters and 25 agents form an 86%-dense
    block (z = 54). Significant blocks cover 3–54% of rated agents depending on chain and weighting. The largest
    campaign (`miner-vouch`, 43% of Base feedback) is star-shaped and invisible to density. Planted Sybil blocks are
    found at pair density >= 0.2 (recall 93%) but not at 0.05 (45%).
12. **A digital twin of ERC-8004 puts the separation in real numbers (study 14).**
    - The twin mirrors 52,557 (Ethereum) and 99,185 (Base) registered agents and syncs live (4 s behind the chain
      head).
    - Taking all top-5 places of reputation-based discovery needs a median of 8 (Ethereum) and 9 (Base) identities
      per category. With 3 anchors, a 50% eclipse needs 201,688 and 377,501. The what-if engine confirms the costs.
    - Next-week agent activity is predicted with AUC 0.91 (Ethereum) and 0.79 (Base), above recent-activity and
      popularity baselines.
13. **The guarantees are tight and the results robust (studies 3, 11, 15; Propositions 1b, 2b, 7).**
    - Theorem 1 is tight (Proposition 1b): measured values differ from the exact probability by at most 0.027.
    - With 20 seeds at B = N, 13/100 target instances are eclipsed (13.0%), against an exact 12.5% and a bound of
      12.8%. The earlier 24% (6/25) was sampling noise.
    - Across 33 settings of m, k, dimension, selector and temperature, targets without a defense are always eclipsed,
      and 11–15% of agents have no honest anchor with a = 3.
    - Semantic blindness is necessary (Proposition 2b): any selector that reads adversary-controlled content, LLMs
      included, can be captured with one identity per slot.
    - The attacker's problem was solved exactly by dynamic programming in 24 settings. Always lying is optimal even
      when identities cannot be replaced, and the RL attacker matches the optimum.
14. **The attack and the defense hold on real agent cards (study 16).** On 1,217 real ERC-8004 descriptions with
    LLM-written tasks and Sybil cards, adaptive Sybil cards match tasks better than the best honest card (cosine 0.77
    vs 0.71). 50 Sybils eclipse targets in 52% of rounds and 200 in 72%; anchors bring this to 0% and 11%, with 12.7%
    of agents without an honest anchor (bound 12.8%). With Qwen2.5-3B reading the real cards (N = 20, B = 10), the targets' Sybil
    share falls from 100% to 69% and their minimum number of honest peers rises from 0 to 0.7.
15. **Honest agents can share evidence safely, but only below a threshold (studies 20, 21; Theorem 2).**
    - Naive and threshold sharing let false reports block all 200 honest agents. Alerts never block anyone.
    - Anchor committees in the final design (semantic-blind seats, quorum, judge once, two-sided accountability,
      signed transcripts checked with a binomial test, evidence pooling, 10 committees per round) cut the worst-case
      wrong answers used from 69% to 5.2% (B = 25) and from 76% to 19% (B = 50). Budget Sybils fall from 21% to 3.1%.
      At most 3 of 200 honest agents were blocked in any run.
    - Above about a third of the identities they cannot be both safe and live (Corollary 2): at quorum 2/3 they block
      27 honest agents at B = 100 and 152 at B = 200; at quorum 0.9 with 45 seats they are safe but convict no Sybil.
    - On the real registries this threshold is 25,833 Sybil identities (Ethereum) and 48,753 (Base).
    - A learning attacker broke two intermediate designs. Against one-sided strikes it got 62 honest reporters muted
      (73% wrong answers used, against 24% for always lying). Against probe-based committees with pooling, it learned to
      answer strangers truthfully and passed the probes (70% at B = 25). Against the final design it reaches 5.7%
      (B = 25) and 18% (B = 50), the same as the worst fixed attacker.
    - On-chain, one judgement costs 1.63M gas with 15 seats.
16. **Verifiable answers work beyond the threshold, and select good agents (study 22, Propositions 15–16).**
    - If a fraction v of answers can be checked with certainty, a statistically implausible count of verified errors
      bans an agent network-wide with no vote. A false accuser cannot forge verified errors.
    - At B = 200, where committees fail, the worst case over four attackers falls from 89% to 7.6% at v = 0.2, and all
      200 Sybils are banned; always-lying Sybils are contained already at v = 0.05 (7.8%, 0.15 verifications per task).
      At B = 400 the worst case falls from 94% to 18%.
    - Damage scales with B/v (Proposition 15). Sleeper Sybils are the hardest attacker. Verified reputation (rank only
      from verified correct work) lowers their damage at B = 200, v = 0.05 from 48% to 30%.
    - At most 1 honest agent was banned in any run; the RL attacker learns to always lie (8.4%).
17. **Anchors make deposits tiny (digital twin).** An attack of value V does not pay when each identity costs more than
    V / (identities needed). Without anchors, taking all top-5 places of a category needs 8 identities on Ethereum;
    with 3 anchors, an eclipse with probability 1/2 needs 202,192. The deterring deposit is 25,274x smaller with
    anchors (42,398x on Base).
18. **Real questions, real answers (study 17).** On 360 MMLU questions answered by Qwen2.5-3B (expert 64%,
    non-specialist 58% accurate), honest agents alone use 41% wrong answers. With 200 always-lying Sybils: 100% without
    a defense, 95% with anchors, 90% with second opinions, and 38% with verification at v = 0.2, which bans all Sybils
    but also, on average, 6 of the 200 least accurate honest agents. With 50 Sybils, committees reach 60% and
    verification 36%. Sybils lying 30% answer better than honest non-specialists here and are not a threat on this
    metric.

## Negative and limiting results (reported as such)

- **Anchors cap Sybil presence, they do not remove it.** With B = N, 24% of target instances were still eclipsed,
  as the bound allows. Sybils that serve well stay as anchors.
- **Blocking alone barely lowers wrong answers served** (28% at u = 0.3). A blocked identity is replaced by a fresh
  one. Proposition 3 bounds damage *per identity*, and with 200 identities the aggregate bound exceeds the horizon.
  The gain comes from correction, at a cost linear in the audit rate.
- **Anchors cost accuracy when honest skill matters.** With skill-dependent accuracy, anchors alone give 31% wrong
  answers used against 30% without a defense: unrelated honest anchors answer worse than Sybils that lie 30% of the
  time. Their value is robustness, not accuracy.
- **Anchors by skill category are a negative result** under a query-adaptive adversary (see finding 6). They help
  only if category membership is verified.
- **Unanimity is conservative.** It rules out framing by a Sybil anchor majority but makes far fewer good corrections.
- **Uniform sampling is an assumption (A1).** A biased sampler degrades the guarantee, as the experiments show.
  So do free identities (Douceur).
- **First versions had weaknesses that were fixed during the study.** Eviction by the plain running mean could not
  evict sleeper Sybils. With static descriptions, Proposition 2's precondition failed in our vector model, so the
  query-adaptive attacker was added. See "Protocol discipline".
- **Second opinions cannot fix correctness at B = N with realistic skills.** Against always-lying, coordinated Sybils
  with skill-dependent honest accuracy, no configuration of study 12 brings wrong answers used below 67% at B = 200.
  Anchors keep contact with honest peers, but their second opinions are too weak there.
- **Pacing is not tested where it could pay.** Proposition 7 needs replaceable identities. With scarce identities or
  shared blocking evidence the game changes; we did not study that regime.
- **The twin's risk score does not predict observed boosted entries.** Low eclipse cost predicts new top-5 entries
  by fresh raters worse than chance (AUC 0.27 Ethereum, 0.43 Base); category activity predicts them (0.88, 0.74).
  Without ground truth, manipulation and organic growth cannot be told apart.
- **Aggregate forecasts are weak.** Holt's smoothing is the best one-week forecast for 1 of 4 weekly series on
  Ethereum and 0 of 4 on Base; feedback arrives in bursts.
- **Committees have a hard limit.** Beyond about a third of the identities (effective corruption 1/2), no quorum
  makes committees both safe and live. This is a theorem, not a tuning problem; only identity costs that keep the
  Sybil share low move it.
- **Budget Sybils evade reports.** A Sybil that tells each victim at most κ − 1 lies is never blocked locally, hence
  never reported; evidence pooling (study 20b) addresses it.
- **Scope.** Experiments are a stylized simulator plus a small LLM (3B) on a small network (20 agents). The attack
  was not run against real A2A deployments, by design. Larger models (7–8B) are prepared as a Colab notebook but not
  yet run. The on-chain sampler ran on a local EVM with a mock VRF coordinator, not on a public chain; the twin only
  reads public logs.

## Folder map

```
sybagent/
  README.md, LICENSE (MIT), CITATION.cff
  docs/theory.md                  Part I: Theorem 1, Propositions 1-8; Part II: Theorem 2, Propositions 9-13
  docs/internal_review.md         a strict reviewer's objections and how each was addressed
  docs/literature_review.md       literature review and novelty check
  analysis/                       run_all.ps1, make_figures.py, make_numbers.py, validate_theory.py,
                                  analyze_erc8004.py, analyze_moltbook.py, games.py, pareto.py,
                                  dense_subgraph.py, dp_attacker.py, build_real_cards.py,
                                  twin_committee.py, repro_check.py
  figures/                        png + pdf
  tables/RESULTS.md               every number quoted above, plus detailed tables
  environment/                    exact package versions
  results/
    1_attack/ ... 7_theory_check/ per-run CSV + config JSON, summary.csv per study (5b has baselines/ and main/)
    8_real_data/                  ERC-8004 and Moltbook graph statistics (aggregates only)
    10_onchain_sampler/           gas, grinding, uniformity and front-running checks
    11_strategic_attacker/        runs, learned Q-tables and policies, games/ (best responses, equilibria)
    12_pareto/                    sweep + Pareto front, knees, hypervolume, epsilon-constraint table
    13_dense_subgraph/            blocks with z-scores, null model, planted-block test
    14_digital_twin/              per chain: weekly series, backtests, eclipse cost over time, what-if checks;
                                  sync_log.csv (live syncs); committee_planning.csv, committee_limits.csv
    15_robustness/                20 seeds, sensitivity sweeps
    16_real_cards/                bundle statistics; embedding/ and llm/ sweeps on real agent cards
    20_shared_defense/            main.csv, threshold.csv, rl.csv, pooling/ (evidence pooling); the first version is
                                  in 20_shared_defense_v1_superseded/
    21_onchain_blocklist/         gas and rule checks of Blocklist.sol
    repro_check.json              result of analysis/repro_check.py
  code/sybagent/              simulator package (src/sybagent, incl. strategy.py, defenses/shared.py and twin/), tests,
                                  configs, contracts/ (AnchorSampler.sol, Blocklist.sol, mocks), scripts (run_experiment.py,
                                  run_strategic.py, run_shared.py, run_pooling.py, onchain_sampler.py,
                                  onchain_blocklist.py, twin.py, crawl_erc8004.py, fetch_agent_cards.py), Colab notebook
```

Proposition numbers follow `docs/theory.md`. The paper numbers the same results in its own order.

Raw data (`data/`) is not part of the repository. The ERC-8004 crawler rebuilds it from public logs. The Moltbook
archive is on Zenodo (DOI 10.5281/zenodo.19594803).

## Reproducing

Python 3.12; packages in `environment/requirements_frozen.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r code\sybagent\requirements.txt -e code\sybagent
.\analysis\run_all.ps1 -Workers 5     # studies 1-5b, 7, 10-12 (13-14 if the crawl exists), figures, numbers, paper
                                      # (about 5 h on 6 cores; study 11 trains 960 RL episodes)
.\.venv\Scripts\python analysis\repro_check.py
```

| study | command |
|---|---|
| 6 (LLM) | Ollama with `qwen2.5:3b`, then `python code\sybagent\scripts\run_experiment.py --sweep code\sybagent\configs\sweep_llm.yaml --out results\6_llm_selector` (about 2 h on a GTX 1650) |
| 6 (larger LLMs) | `code/sybagent/notebooks/colab_llm.ipynb` on a Colab T4; put the returned files in `results/9_llm_colab/` |
| 8 (ERC-8004) | `python code\sybagent\scripts\crawl_erc8004.py --chain ethereum --out data\raw\erc8004\ethereum_feedback.csv` (and `--chain base`), then `python analysis\analyze_erc8004.py` |
| 13, 14 (identities) | the same crawler with `--kind identity --out data\raw\erc8004\<chain>_identities.csv`; long ranges can be split with `--start/--end` and joined by `scripts\merge_crawl_parts.py` |
| 14 (live twin) | `python code\sybagent\scripts\twin.py sync --chain base` (read-only; updates `data\twin`), `twin.py whatif --chain base --tag trust` |
| 10 (tools) | `pip install -e code\sybagent[chain]`; the Solidity compiler is fetched by `py-solc-x` (0.8.24) |
| 8 (Moltbook) | download the Zenodo archive to `data\raw\moltbook\extracted`, then `python analysis\analyze_moltbook.py` |

The paper is kept out of this public repository; every number it reports is in `tables/RESULTS.md`, and the
proofs are in `docs/theory.md`.

## Status and what remains

All studies above are complete and final. Left for future work:
- larger LLM selectors (notebook ready);
- deploying verifiable answers for real tasks (proofs, tests) and measuring their real cost;
- recommendations through anchors' own working sets;
- deploying the anchor sampler on a testnet with a production VRF service;
- a fuller paper on the digital twin (ground-truth labels for manipulation, richer discovery models).

## Protocol discipline

Every number in the paper and in this README is generated from `results/` by `analysis/make_numbers.py`.
`analysis/repro_check.py` re-runs stored runs of studies 1–5b, 11 (including a frozen RL policy), 12, 20 and 22 with the final
code. All 20 checked runs are identical (`results/repro_check.json`). 49 automated tests pass. The LLM study cannot be rerun bit-for-bit, because LLM outputs are not deterministic
across runtimes.

Changes made during the study, all before the final runs:
1. **Card jitter.** Static Sybil descriptions got a small per-identity jitter, because identical descriptions made
   attractiveness 1.0 *weaker* than 0.5. The effect is reported.
2. **Query-adaptive attacker.** The first check of Proposition 2 failed: static descriptions outrank the best honest
   one in only 10–18% of tasks. We added the query-adaptive attacker that the proposition assumes, and report both.
3. **Recency-weighted eviction.** Anchor eviction first used the running mean of all outcomes, which could not evict
   sleeper Sybils. It now uses an exponentially weighted mean. The adaptive study was rerun.
4. **Correction in second opinions.** The first second-opinion run only blocked peers and did not lower wrong
   answers. Correction (use the anchors' majority on disagreement) was added and the study rerun. The negative
   result is kept in the text.
5. **Answer randomness.** Task answers use a separate random stream, so adding them did not change any earlier
   result. Verified by `repro_check.py`.
6. **Study 5b was added after review of our own design.** We identified two weaknesses: anchors are skill-agnostic,
   and colluding Sybil anchors could flip second opinions. This led to the skill-dependent answer model, coordinated
   lies, the unanimity rule and category anchors. These are new options. Their defaults reproduce all earlier
   studies exactly (12/12 runs identical).
7. **Superseded runs.** Sweeps run with intermediate code were deleted and rerun with the final code.
8. **Extensions (studies 10–14) were added after the first version was final.** The strategy hook, the anchor-audit
   option and the twin are new code paths; their defaults reproduce all earlier studies exactly (repro check above).
9. **One snapshot of ERC-8004.** Studies 8, 13 and 14 use the crawl of 10 October 2026. Study 8 was first computed on
   a crawl of 8 October (`results/8_real_data/erc8004_structure_crawl_2026-10-08.csv`). Ethereum had a burst of
   feedback in between, so its values moved: single-reviewer agents 87% -> 92%, mixing 416x -> 275x slower than log n.
   Base did not change. The conclusions are the same.
10. **First RL design.** The first RL attacker valued each (Sybil, victim) pair as if it were irreplaceable and became
    timid (damage 0.22 vs 0.67 for always lying). The final attacker chooses its discount factor by training damage
    and defaults to lying in unvisited states. Both choices are documented in `strategy.py` and `run_strategic.py`.
11. **The Pareto analysis gained a fourth objective** (agents without an honest anchor) after study 11 showed that
    anchor audits break Theorem 1. With three objectives, anchor-audit configurations looked free.
12. **Committee design changed five times, each time because an experiment broke it.**
    - (a) Re-reporting allowed the adversary to resample committees until one framed an honest agent: judge once per
      epoch.
    - (b) False reports flooded the committee queue: reporters are muted after cleared reports.
    - (c) A learning attacker turned one-sided strikes (any failure to convict clears) against honest reporters:
      verdicts became two-sided. Kept in `results/20_shared_defense_v1_superseded/` (`verdict: one_sided`).
    - (d) With evidence pooling, the learning attacker learned to answer strangers truthfully, passed every probe, and
      got honest reporters muted: members now judge signed transcripts instead of probes. Kept in
      `results/20_shared_defense/pooling_probes/` and `probes/` (`evidence_mode: probes`).
    - (e) The first transcript version submitted only disputed answers and convicted honest agents for ordinary
      errors: reporters now submit all audited answers, and members test the error rate.

    The intermediate results are kept because they are findings of the paper.
13. **Probe evidence.** Committee members first required unanimity of their anchors on probes; with a Sybil anchor
    breaking unanimity, honest members rarely voted correctly. They now use the anchors' unique majority with at
    least 2 of 3 probes disagreeing.
14. **One research, one theory.** The shared-defense theory was first written separately and is now Part II of
    `docs/theory.md` (Theorem 2, Corollary 2, Propositions 9–13).

## Citation and license

George David Tsitlauri, Department of Informatics and Telecommunications, University of Thessaly, Greece, 2026. MIT license ([LICENSE](LICENSE)); citation metadata in
[CITATION.cff](CITATION.cff).
