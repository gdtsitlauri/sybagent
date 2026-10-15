# Literature review: Sybil and eclipse attacks on open AI-agent networks

**Working title at the time:** *Semantic Eclipse: Sybil and Eclipse Attacks on Open AI-Agent Networks, and Graph-Theoretic Defenses*

**Date:** 2026-10-08. The review is based on web searches (arXiv, IACR ePrint, conferences) up to that date.
It is not an exhaustive SoK-style review. Newer publications must be checked again before any submission.

---

## 1. Conclusion at a glance

| Question | Answer |
|---|---|
| Does the problem exist in practice? | **Yes, at large scale.** In ERC-8004, 73.5% / 59.2% / 90.6% of the reviewers (Ethereum / BSC / Base) show coordinated Sybil behaviour [E1]. Moltbook had about 1.5 million "agents" under one entity. |
| Has **eclipse** been studied in agent networks? | **No.** Eclipse attacks have been studied only in blockchains and overlays [B1–B5]. |
| Has the manipulation of LLM **selection** been studied? | **Yes, for a single selection:** of a tool [D1–D4] or a peer agent [D5]. Not as a systematic, lasting isolation of a node inside the network. |
| Is there a **graph-based defence** against Sybils in agent networks? | **No.** The existing proposals are deposits/staking [E2], reputation memory [E3] and cryptographic signing [C2]. |
| Is there a theory of robust topology for LLM agents? | **Partly.** SAC [F1] gives (F+1)-robustness conditions, but for a **given, fixed** graph. It does not study an attacker who **shapes** the graph. |

**Verdict:** the direction is **original** as long as the contribution is at the **network level**: lasting isolation (eclipse), a model of cheap attack edges and defences with proven guarantees on graphs. A paper on a single manipulated selection would **not** be original.

---

## 2. Map of the literature

### A. Classic graph-based Sybil defences (foundation)

| # | Work | What it does | Why it is not enough for agents |
|---|---|---|---|
| A1 | Douceur, *The Sybil Attack*, IPTPS 2002 | Defines the attack and shows that without a central authority it cannot be fully prevented. | A fundamental limit, so a cost model is needed. |
| A2 | Yu et al., SybilGuard (SIGCOMM 2006) and SybilLimit (S&P 2008) | Random walks. About O(log n) Sybils per attack edge. | Assume **few attack edges** and a fast-mixing graph of humans. |
| A3 | Danezis & Mittal, SybilInfer (NDSS 2009). Tran et al., SumUp (NSDI 2009) | Bayesian inference. Voting with **max-flow**. | SumUp is a close model for votes and reviews with flows. |
| A4 | Cao et al., SybilRank (NSDI 2012) | Early-terminated random walks with degree normalisation. | Same assumption on attack edges. |
| A5 | Alvisi et al., *SoK: The Evolution of Sybil Defense via Social Networks* (IEEE S&P 2013) [link](https://cs.utexas.edu/~lorenzo/papers/Alvisi13SoK.pdf) | Connects Sybil defence, random walks and community detection. | The key SoK. The theoretical framework starts here. |
| A6 | Yang et al., *Uncovering Social Network Sybils in the Wild* (IMC 2011) | Real Sybils do **not** form tight communities. | First sign that the assumption breaks. In agent networks it breaks even more. |
| A7 | Cheng & Friedman, *Sybilproof Reputation Mechanisms* (P2PEcon 2005) [link](https://or.stanford.edu/seminars/friedman_111605.html) | Impossibility: no **symmetric** reputation function is Sybil-proof. Flow/path-based ones (MaxFlow, PathRank) are value-Sybilproof. | **Key for the theory:** the defence must be asymmetric or flow-based. |
| A8 | MeritRank (2022) [link](https://arxiv.org/pdf/2207.09950). Sybil-proof accounting with transitive trust (Parkes et al.) | Sybil-tolerant reputation and limits of transitive trust. | Useful as baselines. |
| A9 | Graph-based Fake Account Detection: A Survey (2025) [link](https://arxiv.org/pdf/2507.06541) | Recent survey. | Does not cover networks made **only** of agents. |

### B. Eclipse attacks (foundation)

| # | Work | What it does | Relation |
|---|---|---|---|
| B1 | Singh, Castro, Druschel, Rowstron, *Defending against Eclipse Attacks on Overlay Networks* (2004/INFOCOM 2006) [link](https://www.mpi-sws.org/~druschel/publications/Eclipse-Infocom.pdf) | Defines eclipse. Defence: a **bound on node degree** through auditing. Eclipse is more general than Sybil: a Sybil defence does not prevent eclipse. | Fundamental. The degree-bound defence is **graph-theoretic** and can be carried over. |
| B2 | Heilman et al., *Eclipse Attacks on Bitcoin's P2P Network* (USENIX Security 2015) [link](https://www.usenix.org/conference/usenixsecurity15/technical-sessions/presentation/heilman) | Poisoning of the peer table and isolation at restart. | A template for how an eclipse study is set up. |
| B3 | Marcus et al., *Low-Resource Eclipse Attacks on Ethereum* (2018) [link](https://eprint.iacr.org/2018/236.pdf) | Eclipse with few resources. | |
| B4 | *Eclipse Attacks on Ethereum's P2P Network* (2026) [link](https://arxiv.org/abs/2601.16560) | First end-to-end eclipse on execution-layer nodes (28 IPs, DNS-list poisoning). | Shows the topic is active in 2026. |
| B5 | Survey of DHT security techniques (Urdaneta et al.) [link](https://research.vu.nl/en/publications/a-survey-of-dht-security-techniques/). Active Sybil attack in the IPFS DHT (2025/2026) [link](https://arxiv.org/pdf/2505.01139) | Sybil, eclipse and routing attacks on DHTs. | Relevant, because AGNTCY ADS runs on a Kademlia DHT [C1]. |
| B6 | Bortnikov et al., *Brahms* (PODC 2008 / Computer Networks 2009). Auvolat et al., *Basalt* (2021) | Byzantine-resilient random peer sampling, robust to eclipse. | **Defence baselines.** They do not consider peer selection by semantics (LLM). |

### C. Agent-network infrastructure (where the attack happens)

| # | Work | Note |
|---|---|---|
| C1 | AGNTCY Agent Directory Service (2025) [link](https://arxiv.org/pdf/2509.18787) | Discovery on a Kademlia DHT. Handles **Sybil index poisoning** with artifact hosting and rate limiting. Does not handle semantic selection. |
| C2 | Agent Name Service (ANS), IETF draft / arXiv 2505.10609 [link](https://arxiv.org/pdf/2505.10609) | Registry poisoning, handled with signatures and RA validation. Covers **authenticity**, not "who is useful/honest". |
| C3 | ERC-8004 *Trustless Agents* (draft) [link](https://eips.ethereum.org/EIPS/eip-8004) | On-chain identity, reputation and validation registries. The standard itself acknowledges that Sybils can inflate reputation. |
| C4 | SemDHT (2026) [link](https://arxiv.org/pdf/2609.23539). Agent Identity URI (2026) [link](https://arxiv.org/pdf/2601.14567). Agent registries evolution (2025) [link](https://arxiv.org/pdf/2508.03095). Agentic P2P networks (2026) [link](https://arxiv.org/pdf/2603.03753) | Discovery and naming. Useful for the system model. |

### D. Manipulation of LLM selection (the **closest** risk to originality)

| # | Work | What it does | What it does **not** do |
|---|---|---|---|
| D1 | ToolHijacker, *Prompt Injection Attack to Tool Selection in LLM Agents* (NDSS 2026) [link](https://arxiv.org/pdf/2504.19793) | Optimised malicious tool documents that win the selection. | One tool selection, not a network of peers, not isolation over time. |
| D2 | ToolTweak (2025) [link](https://arxiv.org/pdf/2510.02554). Attractive Metadata Attack (2025) | Manipulation of tool metadata. | Same: one selection. |
| D3 | MemMorph (2026) [link](https://arxiv.org/pdf/2605.26154) | Hijacking of tool selection through memory poisoning. | Shows that selection **with memory** can be attacked, so it is closer to "lasting". Still one agent, not a network. |
| D4 | Implicit Manipulation for Skill Selection (2026) [link](https://arxiv.org/pdf/2609.02035) | Manipulation of semantic matching. | One selection. |
| D5 | Trustwave SpiderLabs, *Agent-in-the-Middle* (blog, 21/04/2025) [link](https://www.levelblue.com/blogs/spiderlabs-blog/agent-in-the-middle-abusing-agent-cards-in-the-agent-2-agent-protocol-to-win-all-the-tasks) | An inflated A2A agent card wins **one** task assignment (4 agents, PoC). | Not a paper. No network, no Sybils, no defence. |
| D6 | Agent Name Collision Attacks (2026) [link](https://arxiv.org/pdf/2609.27624). Security analysis of agent protocols (2025) [link](https://arxiv.org/pdf/2511.03841) | Name collisions, card spoofing. | Identity level, not eclipse. |

> **Position with respect to D:** D1–D5 are **the basic tool of the attack** (how to win one selection).
> This work is the **next level**:
> - What happens when that tool is used **repeatedly by many Sybil identities**, in a network with discovery and memory, so that the target ends up **seeing only the attacker**?
> - How much does that cost?
> - How can it be defended **with guarantees**?
>
> **Risk:** the group behind ToolHijacker (N. Gong, Duke) or Trustwave may take that step, so a quick preprint is needed.

### E. Reputation and Sybils in agent markets (the motivation)

| # | Work | Note |
|---|---|---|
| E1 | Xiong et al., *Can Trustless Agents Be Trusted? An Empirical Study of the ERC-8004 Ecosystem* (2026) [link](https://arxiv.org/abs/2606.26028) | **Central motivation.** 73.5% / 59.2% / 90.6% of the reviewers are coordinated Sybils. After removing them, 15.8% / 77.9% / 86.8% of the agents are left without valid feedback. Reputation "manipulated at minimal cost". |
| E2 | LEGIT: Credentialing Protocol for Trustworthy AI Agent Marketplaces (2026) [link](https://arxiv.org/html/2609.21325v1) | Discounted Beta and a **deposit per identity**. An economic defence, not a graph-based one. **Baseline.** |
| E3 | Reputation as Community Memory for the Agentic Web (2026) [link](https://arxiv.org/pdf/2609.19502) | Reputation as memory. No graph-theoretic Sybil defence. |
| E4 | Skill-Conditional Reputation in Agent Swarms (2026) [link](https://arxiv.org/pdf/2606.14200) | Attacks on skill-dependent reputation. Adjacent. |
| E5 | Agent Bazaar (2026) [link](https://arxiv.org/html/2605.17698v1). Agent-to-Agent Finance (2026) [link](https://arxiv.org/pdf/2607.00245). TessIndex (2026) [link](https://arxiv.org/pdf/2608.21942) | The ecosystem of agent markets. |

### F. Graph-level robustness for LLM agents

| # | Work | What it does | Gap |
|---|---|---|---|
| F1 | *Robust Multi-Agent LLMs under Byzantine Faults* (SAC, 2026) [link](https://arxiv.org/abs/2605.09076) | Filter-and-refine. Proves that an **(F+1)-robust** graph suffices. | The graph is **given**. The attacker does not shape it. There are no Sybils (F is fixed). **Eclipse is exactly the attack that breaks its assumption.** |
| F2 | LeBlanc et al., r-robust graphs / W-MSR (IEEE JSAC 2013) | Classic theory of resilient consensus. | A theoretical tool for the defences. |
| F3 | DecentLLMs (2025) [link](https://arxiv.org/pdf/2507.14928). Byzantine Cheap Talk (2026) [link](https://arxiv.org/abs/2606.07790). AAAI Byzantine MAS [link](https://arxiv.org/pdf/2511.10400) | Byzantine agents on fixed topologies. | Same gap as F1. |
| F4 | G-Safeguard (2025) [link](https://arxiv.org/pdf/2502.11127). Topology-Guided Adversarial Propagation (2025) [link](https://arxiv.org/pdf/2512.04129) | Detection and propagation of attacks on a given topology, edge pruning. | No model of the attacker forming the topology. |

### G. Data and empirical background

| # | Source | Use |
|---|---|---|
| G1 | The Moltbook Observatory Archive (2026) [link](https://arxiv.org/pdf/2605.13860) | **Dataset** of activity in an agent-only social network. Useful for realistic graphs. |
| G2 | *Humans welcome to observe* (2026) [link](https://arxiv.org/abs/2602.10127). Persuasive content on Moltbook (2026) [link](https://arxiv.org/pdf/2603.18349). *When Agents Talk* (2026) [link](https://arxiv.org/pdf/2606.00067). Structural Divergence Moltbook vs human networks (Advanced Science) [link](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.77665) | The structure of agent networks **differs** from human ones. This supports the argument that the assumptions of A2–A4 do not hold. |
| G3 | swarm-ai-research PR #626 [link](https://github.com/swarm-ai-research/swarm/pull/626) | Densest subgraph flags 100% of real agent conversations as collusion. A GitHub PR, not peer-reviewed. Useful as anecdotal evidence. |
| G4 | BotSim (AAAI 2025) [link](https://ojs.aaai.org/index.php/AAAI/article/view/33575/35730). Breaking & Defending LLM Bot Detection (2026) [link](https://arxiv.org/pdf/2608.15893) | LLM bots **inside human** networks evade detectors. An adjacent field in a different setting: there, a human trust graph exists. |
| G5 | ERC-8004 on-chain registries (Ethereum, BSC, Base) | Public data for evaluating the defence on a real review graph. |

---

## 3. The gap, stated precisely

Three pieces exist, but **nobody has joined them**:

1. **Selection attacks** (D): an LLM is persuaded to choose the attacker, **once**.
2. **Sybil and eclipse** (A, B): classic theory that assumes **few, expensive attack edges** and peer selection **without semantics**.
3. **Robust graphs for LLM agents** (F): guarantees **for a given graph**.

**This work:** in open agent networks the graph is **formed by LLM choices**. The attacker can create Sybil identities cheaply **and** manipulate the choices, and can therefore:
- (a) break the assumptions of A2–A4 (cheap attack edges);
- (b) achieve a **semantic eclipse** of a target;
- (c) destroy the (F+1)-robustness on which F1 relies.

So a **new threat model, a new cost measure and new algorithms for forming and checking neighbourhoods, with proven guarantees**, are needed.

---

## 4. Research questions (first formulation)

- **RQ1 (Model):** How is an open agent network modelled when edges come from semantic selection? What is the **cost of an attack edge** when the attacker can optimise agent cards (D1–D5)?
- **RQ2 (Attack):** How many Sybil identities, steps and tokens does the eclipse of a target need in realistic networks (Moltbook, A2A, ERC-8004)? How does this break SybilRank/SumUp and the (F+1)-robustness of SAC?
- **RQ3 (Defence):** A neighbourhood-formation rule that **guarantees** that every honest agent keeps at least k honest neighbours, or r-robustness, even against an adaptive attacker who knows the algorithm. Candidate building blocks:
  - a degree bound (B1);
  - Byzantine-resilient sampling (B6);
  - flow-based asymmetric reputation (A3, A7);
  - separating semantic ranking from randomness (e.g. a share of the peers chosen at random, not by the LLM).
- **RQ4 (Theory):** Impossibility results. Does the Cheng & Friedman theorem (A7) carry over when reputation comes from LLM evaluations? What is the minimal randomness peer selection needs to rule out eclipse?
- **RQ5 (Evaluation):** Real data (G1, G5) and simulation with the A2A SDK and real LLMs.

---

## 5. First steps (plan at the time)

1. **Month 1:** Read A5, B1, B2, D1, E1 and F1 in full. Reproduce D5 (Agent-in-the-Middle) with the A2A SDK.
2. **Months 2–3 (Part I, measurement and attack):** Extend to a network of N agents with discovery and memory. Measure the cost of eclipse. Show that SybilRank, SumUp and SAC break. Targets: USENIX Security, CCS, NDSS, or a workshop for an early marker.
3. **Months 4–9 (Part II, defence):** An algorithm with a proof, evaluated on Moltbook and ERC-8004.
4. **Later:** Theory (RQ4), a distributed implementation, game-theoretic analysis.

---

## 6. Risks

- **Competition:** the groups behind D1 and D5 are one step away. A quick preprint is needed for Part I.
- **Defining "honest":** in agent networks "honest" is fuzzier than in human networks. The definition must be made carefully in RQ1.
- **Ethics:** no experiments on live third-party networks, only simulation and public historical data. Responsible disclosure where needed.
- **Unconfirmed figures:** the Moltbook number (1.5 million agents under one entity) comes from press sources and must be confirmed before it goes into a paper. The ERC-8004 numbers come from the abstract of E1.
