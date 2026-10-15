# Internal review of SybAgent

A strict reviewer's objections, written before submission, and what was done about each. "Open" marks what remains
a stated limitation.

| # | Objection | Response | Status |
|---|---|---|---|
| 1 | The simulator is stylized: skills are vectors, not real agents. | Study 16 runs the attack and the defense on real ERC-8004 agent cards (1,217 unique descriptions fetched from the identity registry), with tasks and Sybil cards written by an LLM and compared with a real embedding model (bge-small). A second run lets Qwen2.5-3B read the real card texts. | addressed |
| 2 | The LLM is small (3B) and the LLM network tiny (20 agents). | The Colab notebook for 7–8B models is ready but needs the author's Colab session. The embedding study (N = 200, real text) covers scale. | open (author) |
| 3 | Theorem 1 may be loose. | Proposition 1b proves it tight up to ((B−a+1)/B)^a. Measured values differ from the exact probability by at most 0.027 in 16 settings. | addressed |
| 4 | Proposition 2 is about one scoring rule. | Proposition 2b: any selector that reads adversary-controlled content can be captured with one identity per slot. Content-independent slots are necessary. | addressed |
| 5 | 24% eclipsed vs a 13% bound looks like a violation. | 20 seeds: 13/100 = 13.0% vs exact 12.5% and bound 12.8% (p = 0.52). The 5-seed figure was noise. | addressed |
| 6 | Results may depend on m, k, dimension or the selector. | 33 settings in study 15: no defense always 100% eclipsed; 11–15% without an honest anchor, around the bound. | addressed |
| 7 | The RL attacker may simply be weak. | The attacker's problem is solved exactly by dynamic programming (belief MDP). Always lying is optimal in all 24 settings, and Proposition 7 now has a pathwise proof. The RL attacker matches the optimum. | addressed |
| 8 | Proposition 7 assumed replaceable identities. | Claim 1 now holds without that assumption; only the bang-bang claim needs it, and the text says so. | addressed |
| 9 | Second opinions do not fix correctness at B = N with realistic skills. | Reported as a limit (67% minimum in study 12). Part II (studies 20–22) studies shared defenses for this regime. | open (stated) |
| 10 | The on-chain sampler ran only on a local EVM with a mock VRF. | Stated. Gas of the consumer contract is exact; the VRF fee is excluded and named. | open (stated) |
| 11 | No test on a real A2A deployment. | Excluded on ethical grounds; only public logs and public registration files are read. | open (by design) |
| 12 | The twin's risk score does not predict observed manipulation. | Reported as a negative result; no ground-truth labels exist. | open (stated) |
| 13 | Data drift between crawls. | Single snapshot (10 October 2026) for studies 8, 13, 14; the earlier values are kept and the change is documented. | addressed |
| 15 | Local blocking barely helps; why not share evidence? | Part II: anchor committees with a safety/liveness theorem and a threshold (Theorem 2, Corollary 2); naive sharing shown to frame every honest agent. | addressed |
| 16 | Committees could be gamed by a smart adversary. | The RL attacker was run against every design; it broke two (muting honest reporters; passing probes by being honest to strangers). Fixed by two-sided verdicts and signed transcripts with a binomial test; it no longer beats the worst fixed attacker. | addressed |
| 17 | Budget Sybils never trigger reports. | Evidence pooling (Proposition 13): 21% → 3.1% at B = 50. | addressed |
| 18 | Committees beyond a third of Sybils. | Impossible by Corollary 2; stated, with the real-registry thresholds from the twin. | open (theorem) |
| 19 | A few honest agents are still blocked by committees. | At most 3 of 200 per run; accounted for by corrupt seats in Theorem 2. | open (stated) |
| 20 | Nothing works beyond a third of Sybils. | Part III: verifiable answers with proof bans work at any Sybil share (B = 2N: 94% → 18%); the cost scales with B/v. | addressed |
| 21 | Answers are a coin flip, not real. | Study 17: 360 MMLU questions answered by Qwen2.5-3B as expert or non-specialist; all defenses re-evaluated. | addressed |
| 22 | Verification with noisy honest agents. | With 42% honest error, verification bans on average 6 of 200 of the least accurate honest agents; stated. | open (stated) |
| 14 | Reproducibility. | Every number is generated; repro check re-runs stored runs of studies 1–5b, 11, 12, 20 and 22 (20/20 identical); tests cover the new code. | addressed |
