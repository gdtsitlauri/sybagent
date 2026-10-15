# Theory: semantic selection is cheap to eclipse; semantic-blind anchors are not

## 1. Model

* **Agents.** `N` honest agents and `B` adversarial (Sybil) identities are registered in a registry `R`.
* **Cards.** Every identity publishes a card (free text or a vector). The adversary chooses its cards
  arbitrarily and adaptively, with full knowledge of every honest agent's algorithm.
* **Working set.** Each honest agent `i` keeps a working set `W_i` of peers it collaborates with.
  `|W_i| ≤ m`. In the baseline, `W_i` is managed by recency: the `k` peers chosen for each task are
  inserted, and the oldest peers are evicted.
* **Semantic selection.** For a task `x`, agent `i` scores candidates with a function `score_i(c, x)`
  that depends on the card of `c`, and optionally on `i`'s past experience with `c` and on public
  reputation. Then it picks the top `k`, or samples them with a softmax.
* **Service.** When `i` uses peer `j` for a task, `i` observes an outcome quality `q ∈ [0, 1]`.
* **Eclipse.** Agent `i` is *eclipsed* at time `t` if `W_i(t)` contains no honest peer.

### Semantic-blind anchors (proposed)

Agent `i` reserves `a` of its `m` slots as **anchors** `A_i ⊆ W_i`:

1. **Refill.** Whenever `|A_i| < a`, missing anchors are drawn from a sampling oracle `S`, excluding
   `i` and the current anchors. Cards are never read in this step.
2. **Stickiness.** Anchors are never evicted by recency.
3. **Eviction by evidence.** An anchor `j` is removed only if `i` has observed at least `μ` outcomes from
   `j` and the exponentially weighted mean of these outcomes (weight `λ` on the newest) is below a threshold
   `τ`. Weighting recent outcomes evicts a "sleeper" Sybil promptly once its service degrades. A plain
   running mean would hide the change behind a long good history; the first version of the simulator had
   exactly this weakness.

The remaining `m − a` slots are managed exactly as in the baseline.

### Assumptions

* **A1 (sampling).** Each draw of `S` returns a uniformly random registered identity among those not
  excluded. With probability at most `β` per draw, the adversary may replace it with a Sybil of its choice
  (`β = 0` is an honest sampling service).
* **A2 (honest service).** The exponentially weighted mean of outcomes that any honest peer delivers to any
  honest agent stays at or above `τ`. In the simulator, honest quality is `½(1 + ⟨cap, x⟩) + noise ≥ ½ − noise` on
  non-negative skill vectors, while `τ = 0.3`.
* **A3 (identities).** The adversary controls at most `B` registered identities.

---

## 2. Results

### Proposition 1 (card independence)

For every honest agent `i` and every time `t`, the anchor set `A_i(t)` is a function of the draws of `S`
and of the outcomes `i` observed from its anchors only. It does not depend on any card. Hence no choice of
card content, including adversarially optimized text that wins every semantic comparison, changes the
distribution of the number of honest anchors.

*Proof.* Refill (step 1) reads only `S`. Stickiness (step 2) involves no data. Eviction (step 3) reads
only outcomes `i` observed from `j ∈ A_i`. Card contents enter only the scoring function of the
non-anchor slots. ∎

### Theorem 1 (anchor eclipse bound)

Under A1–A3, for every honest agent `i` and every time `t`, against any adaptive adversary,

```
Pr[ A_i(t) contains no honest anchor ]  ≤  p^a ,   where   p = β + (1 − β) · B / (N + B − a).
```

*Proof.*

* **Honest anchors are never removed.** By A2, the running mean of an honest anchor never drops below
  `τ`, so step 3 never removes it. Step 2 forbids any other removal. So once a slot holds an honest
  identity, it holds an honest identity forever.
* **Which slots can hold a Sybil.** At time `t`, slot `s` holds a Sybil only if the **most recent** draw
  that filled `s` returned a Sybil.
* **Probability of one such draw.** Fix such a draw and condition on the entire history before it. The
  pool excludes `i` and at most `a − 1` current anchors. In the worst case those excluded anchors are
  honest, so the pool has at most `B` Sybils among at least `N + B − a` identities. Under A1, the draw is
  a Sybil with probability at most `β + (1 − β)·B/(N + B − a) = p`.
* **All `a` slots.** Order the `a` most-recent filling draws of the slots by time. Apply the chain rule.
  Each factor is at most `p` given everything before it. So the probability that all of them returned
  Sybils is at most `p^a`. ∎

### Proposition 1b (Theorem 1 is tight)

For every `N, B ≥ a` there is an adversary against which, for every honest agent and every `t`,

```
Pr[no honest anchor] = C(B, a) / C(N + B − 1, a)  ≥  ((B − a + 1)/(N + B − a))^a .
```

The ratio to the bound `p^a` (with `β = 0`) is at least `((B − a + 1)/B)^a → 1` when `B ≫ a`. No analysis of this
defense can therefore give a better bound than Theorem 1 by more than this factor. With `β > 0`, an adversary that
uses every biased draw also attains `p` per draw, and the bound is again tight up to the same factor.

*Proof.* Let every Sybil deliver service at least `τ`, so no anchor is ever evicted. Each agent's anchors are then
its first `a` draws: a uniform sample without replacement of `a` identities among the `N + B − 1` others. All are
Sybils with probability `C(B,a)/C(N+B−1,a) = Π_{i<a} (B − i)/(N + B − 1 − i)`. Each factor is at least
`(B − a + 1)/(N + B − a)`. ∎

### Corollary 1 (cost of eclipsing an anchored agent)

Take `β = 0`. To make `Pr[eclipse] ≥ ½` for an agent, the adversary needs

```
B ≥ (N − a) · 2^{−1/a} / (1 − 2^{−1/a})   ≈   (a / ln 2) · N    for large a.
```

The number of identities needed grows **linearly in the network size** `N` and in `a`. If every identity
costs `c` (a deposit, proof of work or credential), the adversary's budget must be `Ω(a·N·c)`.

### Proposition 2 (semantic selection is eclipsed with O(m) identities, independent of N)

Consider the baseline (no anchors). Selection is deterministic top-`k` (temperature 0) with
experience-weighted scores `score_i(c, x) = sim(card_c, x) + w·(q̄_ic − ½)`.

Suppose the adversary deploys `B ≥ m` Sybils. Suppose that, for every task `x` of the target `i`, every
Sybil's score exceeds every honest peer's score. Under A2 with Sybil service at least the honest mean,
this happens when, for all honest `h` and all Sybils `s`,
`sim(card_s, x) − sim(card_h, x) > w·(q̄_ih − q̄_is)`.

Then `i` is eclipsed after at most `⌈m / k⌉` tasks and stays eclipsed.

*Proof.*
* Every task selects `k` Sybils.
* Recency eviction removes the oldest entries, so after `⌈m/k⌉` tasks all `m` slots hold Sybils.
* From then on no honest peer can outrank a Sybil, so none re-enters. ∎

### Proposition 2b (semantic blindness is necessary)

Consider any rule that fills a slot of agent `i` for query `x` by a function `g` of the candidates' adversary-
controllable content: cards, free text read by an LLM, or reviews the adversary can post. The function may be
randomized and may also use `i`'s private experience. Suppose the adversary can present, for each query, content
that `g` weakly prefers to the content of every honest candidate: e.g. a copy of the best honest card for `x` plus
an improvement, or a prompt-injected description. Suppose also that Sybil service keeps the experience term at least
the honest average. Then one Sybil identity per slot captures every such slot with probability at least that of
the best honest candidate. With strict preference, it captures the slot with probability 1. Hence `m` identities
suffice, independently of `N`, for **any** selector of this kind, LLM selectors included.

*Proof.* For each query, the Sybil's content is at least as preferred as any honest candidate's, and the
experience term does not favour honest peers. So the Sybil wins every comparison that an honest candidate wins.
Apply the argument of Proposition 2 slot by slot. ∎

The only slots this argument does not reach are those filled independently of adversary-controlled content.
Anchors are exactly such slots (Proposition 1). In this sense semantic blindness is not merely one possible design:
some content-independent slots are **necessary** for any guarantee that does not grow with the cost of content
manipulation.

**Separation.** The baseline can be eclipsed with `B = m` identities, a constant independent of `N`.
The anchored design needs `B = Θ(a·N)` for a constant eclipse probability (Corollary 1). Card
manipulation, which is what LLM-driven selection is vulnerable to (prompt-injection-style "attractive
metadata"), is neutralized for the anchor slots (Proposition 1).

### Proposition 3 (cross-checking bounds the damage per Sybil)

**Mechanism (second opinion).**
* Each task has a correct answer.
* After a non-anchor peer `j` answers a task for agent `i`, `i` flips a private coin with probability `u`.
* On heads, `i` asks all its anchors the same task and takes their answer if it is a unique majority. Ties
  give no evidence.
* If `j`'s answer differs from that majority, a disagreement is recorded for `(i, j)`.
* After `κ` disagreements, `i` blocks `j`.

Let `h_i` be the probability that `i`'s anchor majority exists and is correct on a task.

**Claim.** For every Sybil `j`, and every strategy that chooses on which tasks to lie (adaptively, but
without seeing `i`'s coin, which is flipped after `j` has answered),

```
E[ number of wrong answers j delivers to i before being blocked ]  ≤  κ / (u · h_i).
```

The bound does **not** depend on how often `j` lies. Lying rarely only spreads the same bounded damage over
more time.

*Proof.*
* Consider any wrong answer of `j`. It is recorded as a disagreement if (a) the coin is heads (probability
  `u`, independent of everything `j` knows) and (b) the anchor majority is unique and correct (probability
  `h_i`, independent of the coin). A correct majority disagrees with a wrong answer.
* So each wrong answer independently yields a recorded disagreement with probability at least `u·h_i`.
* The number of wrong answers until the `κ`-th recorded disagreement is a sum of `κ` geometric variables
  with success probability at least `u·h_i`. Its mean is at most `κ/(u·h_i)`.
* Disagreements caused by correct answers only make blocking earlier. ∎

**Consequences.**
* With `B` Sybils, the total expected number of wrong answers that an honest agent receives from Sybils it
  uses is at most `B·κ/(u·h_i)`. Damage is bounded by the identity budget, not by time.
* `h_i` is high when most anchors are honest. With `a = 3` and honest anchors correct with probability
  0.95, a majority of two honest anchors is correct with probability ≥ 0.90. Theorem 1 bounds the chance
  that this majority is missing.
* **False positives.** An honest peer is blocked only if it accumulates `κ` disagreements. These come from
  its own errors or from a wrong anchor majority. The experiments measure how often this happens. A decay
  of the disagreement counts reduces false positives, at the cost of a weaker damage bound.
* **Cost.** Each audit costs `a` extra queries, so the expected overhead is `u·a` queries per answer from a
  non-anchor peer.
* **Correction.** On a recorded disagreement the agent uses the anchors' majority answer. In the experiments
  this, not blocking, is what lowers the aggregate wrong-answer rate. With many fresh identities, the
  per-identity bound `κ/(u·h_i)` times `B` exceeds the number of interactions in the horizon.

### Proposition 4 (unanimity prevents framing by colluding anchors)

**The attack.** Sybils that happen to be anchors can **coordinate**: they return the *same* wrong answer to a
task. Under the majority rule, a Sybil majority among `i`'s anchors (two of three) is enough to:
* "correct" an honest peer's right answer into the agreed wrong one, and
* record disagreements against that honest peer, eventually blocking it (framing).

For uniformly sampled anchors with Sybil probability `p` per slot, a Sybil majority of three occurs with probability
`3p²(1−p) + p³`. That is `≈ 0.5` when `B = N`.

**The rule.** Act (correct, or record a disagreement) only if **all** anchors (at least two) return the same answer.

**Claim.** Let the honest anchors err with probability at most `ε` on the task, uniformly over the `n−1` wrong
classes, and independently. Then, for any coordinated adversary,

```
Pr[ an audit corrects a right answer, or records a disagreement against a right answer ]
      ≤ Pr[no honest anchor] + ε/(n−1)
      ≤ p^a + ε/(n−1).
```

*Proof.*
* Unanimity on a wrong answer requires every honest anchor to output that same wrong answer.
* If at least one anchor is honest, this happens with probability at most `ε/(n−1)` per honest anchor. With `k`
  honest anchors it is `(ε/(n−1))^k ≤ ε/(n−1)`.
* Otherwise all anchors are Sybils, which Theorem 1 bounds by `p^a`. ∎

Framing an honest peer needs `κ` such events, so its probability is far smaller.

**Cost.** Unanimity also needs the honest anchors to agree when they are right. With skill-dependent accuracy, honest
anchors that are poor matches for the task disagree more often. So unanimity corrects fewer wrong answers than the
majority rule. The experiments quantify this trade-off.

### Proposition 5 (category anchors: relevance vs. robustness)

Anchors sampled from the agent's own task category give better second opinions, because they are closer to the task.
But the category has only `N_c ≈ N/K` honest members, while a query-adaptive adversary can claim every category. The
per-slot Sybil probability for a category anchor is therefore `p_c = B/(N_c + B − a)`, which is much larger than
`p = B/(N + B − a)`.

**Claim.** With `a_g` global and `a_c` category anchors (`a = a_g + a_c`), sampled as in Theorem 1,

```
Pr[no honest anchor] ≤ p^{a_g} · p_c^{a_c}   ≤ p^{a_g}.
```

*Proof.* Theorem 1's argument applies slot by slot. Each slot's most recent filling draw is a Sybil with conditional
probability at most `p` (global) or `p_c` (category). ∎

**Design rule.** Keep at least one global anchor. The eclipse guarantee then never falls below `p^{a_g}`, while
category anchors raise the quality of second opinions.

### Proposition 6 (grinding the sampling seed)

Let the anchor draw be a deterministic function of a seed `s` (as in `contracts/AnchorSampler.sol`), modelled as a
random oracle, and let `q` be the probability that a uniformly random seed gives no honest anchor (`q ≤ p^a` by
Theorem 1). Suppose a party can obtain `r + 1` independent seeds, see the outcome of each, and pick one. Then

```
Pr[no honest anchor] = 1 − (1 − q)^{r+1}   ≤   (r + 1) · q .
```

*Proof.* The `r + 1` outcomes are independent. The party fails only if every one has an honest anchor, which has
probability `(1 − q)^{r+1}`. ∎

**Consequences for an on-chain sampler.**
* *Block-hash seeds.* Whoever produces the block whose hash seeds the draw can try many block variants (ordering,
  extra data), so `r` is limited only by hashing speed and `Pr → 1`. If that party produces a share `f` of the
  blocks, `Pr[no honest anchor] ≥ f · (1 − (1 − q)^{r+1}) + (1 − f) q`, regardless of `a`.
* *Commit–reveal* with a last revealer who may abort gives `r = 1`, which doubles the bound.
* *VRF.* The output is unique for the request and unpredictable before it, so `r = 0` provided the requester cannot
  re-request after seeing an unwanted output and the oracle cannot force a new request by withholding. The contract
  enforces one pending request per agent, and draws only identities registered before the request (a snapshot of
  the registry size), so fresh Sybils cannot be added after the outcome is known. Theorem 1 then holds with `β = 0`.

### Proposition 7 (with local blocking, lying is a dominant strategy; the audit rate is bang-bang)

**Setting.** Second opinions as in Proposition 3: each answer is audited with a private coin `u`; a lie is caught
(corrected and recorded as a disagreement) with probability `q = u·h`, where `h` is the probability that the anchors
return a unique correct majority; a truthful answer is wrongly flagged with probability `f ≥ 0`; `κ` disagreements
block the Sybil. The Sybil does not see the coin.

**Claim 1.** For one Sybil identity and a horizon of `T` answers, lying on every answer maximizes the expected number
of wrong answers used, against every commitment `(u, κ, rule)`. This holds whether or not a blocked identity can be
replaced, and the expected damage per identity before blocking is at most `κ(1 − q)/q`.

*Proof.* Index the identity's lies `1, 2, …` and give lie number `j` its own catch coin `C_j ~ Bernoulli(q)`. Let
`K` be the index of the `κ`-th caught lie. Under any policy the identity tells at most `min(T, K)` lies before it
is blocked or the horizon ends (truthful answers can only add flags, when `f > 0`, and never add damage); always
lying attains exactly `min(T, K)` lies. Each lie adds damage `1 − C_j`. Pathwise, the always-lie policy therefore
dominates every other policy, so it maximizes the expectation, which is at most `E[K](1 − q) = κ(1 − q)/q`. With
replacement the same argument applies to each identity in turn. ∎

`analysis/dp_attacker.py` confirms this numerically. It solves the attacker's problem exactly, as a belief MDP over
(time, age of the identity, lies told) by backward induction, for 24 settings (replaceable or scarce identities, `u`,
`κ`). The optimal value equals the always-lie value in all of them, and Monte Carlo agrees. Pacing lies never pays
when blocking counts only caught lies. It could pay only if truthful service bought the Sybil something, such as
selection by answer accuracy, or if audits were triggered by observable events.

**Claim 2.** Against this best response the damage is affine in `u`, `D(u) = D₀ (1 − u·h_eff)`, and the extra
queries per task are linear, `Q(u) = u · Q₁`. For a price `c` per query, the defender's loss `D(u) + c·Q(u)` is
affine in `u`. Its minimum over `u ∈ [0, 1]` is therefore at an end point: `u* = 1` if `c < D₀ h_eff / Q₁`, and
`u* = 0` otherwise.

*Scope.* Claim 2 needs damage to be affine in `u`, which holds when the identities that serve a victim do not depend
on its audit rate, as with replaceable identities. When blocking removes Sybils for good (scarce identities, or
blocking shared across the network), auditing also shrinks the Sybils' total exposure, and `D(u)` becomes convex
rather than affine; the bang-bang conclusion then need not hold. Claim 1 still holds.

### Proposition 8 (Stackelberg identity game)

The adversary buys `B` identities at price `ρ/N` each, where `ρ` is the price of `N` identities in units of the value
of one eclipse. Its utility is `U(B) = P_a(B) − ρ B / N` with `P_a(B) = (B/(N + B − a))^a` (Theorem 1). The defender
picks `a` first.

* With `x = B/N` and `N ≫ a`, `P_a ≈ (x/(1+x))^a`. The best response `x*(a, ρ)` and the attacker's best utility
  therefore depend only on `a` and `ρ`, not on `N`. In identities, the attack price grows linearly in `N`
  (Corollary 1).
* Without anchors, `O(m)` identities give `P = 1` (Proposition 2). So `U = 1 − ρ m/N`, which tends to 1 as `N`
  grows: the attack always pays.
* With anchors, `max_x U` decreases in `a`. The defender's Stackelberg choice is the smallest `a` with
  `max_x U ≤ 0`, or with `P_a(x*) ≤ ε`. `analysis/games.py` computes it for the simulator and for the real
  registry sizes.

---

## 3. What the guarantees do *not* cover (stated limits)

1. **Sampling integrity (A1).** If the adversary can bias the sampling oracle (`β > 0`), for example by
   eclipsing a DHT-based random walk, the bound degrades to `p ≥ β`. The adaptive experiments measure
   this. Uniform sampling therefore needs its own protection (e.g. Brahms/Basalt-style samplers,
   verifiable random functions, or a registry with sampling audits). Proposition 6 and
   `contracts/AnchorSampler.sol` give one such protection for on-chain registries.
2. **Identity cost (A3).** With free identities `B` is unbounded. No defense can then prevent Sybil
   majorities (Douceur, 2002). The bound makes the needed identity cost explicit.
3. **Anchor quality vs. utility.** Anchors are chosen without regard to skills, so they are often poor
   matches for the agent's tasks. Their role is to guarantee contact with honest peers, which can then
   serve as cross-checks or as recommenders. The experiments report the service-quality cost of
   reserving `a` slots.
4. **Sleeper Sybils.** Sybils that serve well never get evicted from anchor slots. This does not violate
   the bound (it already counts them as Sybils), but it means anchors cap, rather than remove, Sybil
   presence.
5. **A2 in heterogeneous networks.** If honest peers can genuinely deliver means below `τ` for some
   agents, then honest anchors can be evicted. The bound must then be multiplied by the number of
   refills. A lower `τ` keeps this rare.

---

# Part II. Sharing evidence against Sybils without a false-accusation channel

Part I lets every honest agent block a lying peer only for itself. Against `B` replaceable identities this barely
lowers the damage (study 5), and a lying Sybil can mislead each honest agent separately. Here honest agents share
what their second opinions reveal. The difficulty is that Sybils can share too: false reports, and votes.

## II.1 Model

* `N` honest agents, `B` Sybils, `p_s = B/(N + B)`. Each honest agent has `a` anchors (Theorem 1); a
  single anchor draw is a Sybil with probability `p ≤ B/(N + B − a)`.
* An honest agent that records `κ` disagreements against a peer blocks it locally and publishes a **report**. Every
  Sybil may publish false reports against honest agents.
* **Committee.** A report about `x` convenes a committee of `c` identities drawn uniformly from the registry
  (semantic-blind, like anchors; on-chain by the VRF sampler of Proposition 6). Each honest member sends `x` `r` probe tasks
  that look like ordinary tasks, compares each answer with the unique majority answer of its own anchors, and votes
  *block* if at least `e` answers disagree. Sybil members vote strategically. `x` is blocked network-wide if at least
  `⌈Qc⌉` members vote *block* (quorum `Q`).
* **Judge once.** An agent is judged at most once per epoch.
* **Accountability, two-sided verdicts.** A quorum of *block* votes convicts; a quorum of *keep* votes clears the
  accused and gives each reporter a strike; anything else is inconclusive and penalises nobody. A reporter with `s`
  strikes is muted.
* **Evidence pooling** (optional). Every single disagreement is shared, not only full blocks; once `t` distinct
  unmuted agents hold evidence against `x`, `x` goes to a committee.
* **Alerts** (separately or in addition). A report raises an alert. For a while, every honest agent audits the
  reported agent with a high probability, but blocks it only on its own disagreements.

## II.2 Effective corruption

An honest member can be misled by its own anchors. If the majority of its anchors are Sybils, they answer the
probes with the coordinated lie, and the member sees an honest `x` as disagreeing and a lying Sybil `x` as agreeing.
Let

```
π = Pr[ Bin(a, p) > a/2 ]            (an honest member's anchors have a Sybil majority)
ρ = p_s + (1 − p_s) π                (effective corruption of a committee seat)
```

Let `ε` be the probability that an honest member with an honest anchor majority still votes *block* on an honest `x`
(honest errors: at most `C(r, e) (1 − α)^e`-like for answer accuracy `α`). Let `δ` be the probability that it votes
*block* on a lying Sybil.

## II.3 Results

### Theorem 2 (safety and liveness of anchor committees)

For one judgement, with seats drawn independently:

```
Pr[honest x blocked]        ≤ Pr[ Bin(c, ρ_s) ≥ ⌈Qc⌉ ] ≤ exp(−c · KL(Q ‖ ρ_s)),     ρ_s = ρ + (1 − ρ) ε,
Pr[lying Sybil x not blocked] ≤ Pr[ Bin(c, ρ_l) < ⌈Qc⌉ ] ≤ exp(−c · KL(Q ‖ ρ_l)),   ρ_l = (1 − ρ) δ,
```

where the Chernoff forms hold for `ρ_s < Q < ρ_l`. Both error probabilities fall exponentially in the committee
size if and only if `ρ_s < Q < ρ_l`.

*Proof.* A seat votes *block* on an honest `x` only if it is a Sybil (probability `p_s`), or an honest member with a
Sybil anchor majority (`(1 − p_s)π`), or an honest member with an honest majority that errs (`(1 − ρ)ε`). Seats are
independent, so the number of *block* votes is dominated by `Bin(c, ρ_s)`. For a lying Sybil, a seat votes *block*
at least when it is an honest member with an honest anchor majority that detects the lies, which has probability
`(1 − ρ)δ`. Sybil seats never do. Apply the Chernoff–Hoeffding bound in KL form. ∎

### Corollary 2 (a threshold)

Since `δ ≤ 1` and `ε ≥ 0`, a quorum that makes the committee both safe and live exists only if `ρ < 1/2`. In
particular it requires `p_s < 1/2`. More anchors help, because `π` falls in `a` when `p < 1/2`. With `p_s ≥ 1/2` no
anchor count helps, because `π → 1` as `a` grows. A designer must then choose: a high quorum keeps the committee safe
but makes it useless, and a low quorum makes it useful but lets Sybils frame honest agents.

### Proposition 9 (why judge once)

If an honest agent may be judged again after every new report, the adversary can report it until a committee
convicts. After `j` judgements the framing probability is `1 − (1 − P)^j`, with `P` the per-judgement bound of
Theorem 2. This is the committee version of seed grinding (Proposition 6). Judging each agent at most once
per epoch caps it at `P`.

### Proposition 10 (accountability bounds the flood, with two-sided verdicts)

With muting after `s` cleared reports, Sybils can trigger at most `s · B` committees on honest agents in total. A
false report also costs the reported honest agent nothing beyond the committee's probes, as long as the committee is
safe. Without muting, `B` Sybils that file `f` reports per round keep the committee queue full and delay the
judgement of real reports indefinitely. This is a denial-of-service attack, and it appears in the experiments.

Strikes must be two-sided. If every failure to convict counted as clearing (one-sided verdicts), an honest reporter
of a lying Sybil would be struck with the committee's miss probability, which is large near the threshold. The
adversary can then mute honest reporters by keeping its Sybils just hard enough to convict. A learning attacker found
exactly this (study 20, first version). With two-sided verdicts an honest reporter of a lying Sybil is struck only
if a quorum votes *keep*:

```
Pr[strike an honest reporter of a lying Sybil] ≤ Pr[ Bin(c, 1 − ρ_l) ≥ ⌈Qc⌉ ],
```

which falls exponentially in `c` whenever `1 − ρ_l < Q`, the same condition as liveness.

### Proposition 11 (alerts are safe at any Sybil share)

With alerts only, a report never blocks anyone. It only raises the reported agent's audit rate, and blocking still
requires the agent's own `κ` disagreements. The probability that an honest peer is blocked is therefore the same as
without sharing (Proposition 4 bounds it under unanimity). False reports cost extra queries but cannot frame
anyone.

### Proposition 12 (network-wide damage per identity)

Under Theorem 2's conditions, a lying Sybil is blocked network-wide after its first report has been judged. Each
identity's damage is then bounded across the whole network, not per victim. It is at most the damage done before
the first local block (Proposition 3) times the number of victims it reaches before the judgement, plus the
queue delay. Total damage scales with `B`, not with `B · N`.

### Proposition 13 (evidence pooling catches budget Sybils)

A *budget* Sybil tells each victim at most `κ − 1` lies, so no victim ever blocks it and, without pooling, it is never
reported. With pooling at threshold `t`, each caught lie gives its victim evidence, and a victim can catch at most
`κ − 1` of this Sybil's lies. Hence the Sybil reaches a committee after at most `t(κ − 1)` caught lies, that is,
after an expected `t(κ − 1)/q` lies in total. Its expected damage before judgement is at most `t(κ − 1)(1 − q)/q`,
network-wide, whatever the number of victims. Without pooling it is `(κ − 1)(1 − q)` per victim, unbounded in the
number of victims. False evidence is judged and struck like false reports (Proposition 10).

### Proposition 14 (judge signed transcripts, not probes)

With probes, a member's verdict depends on how the accused answers the member's *new* tasks. A Sybil that answers
strangers truthfully and lies only to agents it already serves passes every probe. Committees then clear it, and with
any strike rule its honest reporters are eventually muted. The assumption that probes are indistinguishable from
ordinary tasks fails exactly here: a committee member is a stranger. (A learning attacker found this, study 20b.)

Suppose instead that agents sign their answers (A2A messages can be signed), and reporters submit **all** the
accused's signed `(task, answer)` items they audited, agreements included. Each member re-checks a random sample of
`m` of them against its own anchors. It votes *block* if its count `k` of disagreements rejects the hypothesis "the
accused errs no more often than an honest agent", i.e. if `Pr[Bin(m, 1 − α) ≥ k] < ℓ` for honest accuracy `α` and
level `ℓ`. Then:
* the verdict depends only on answers the accused **already gave**, so behaving well towards the committee is useless;
* a false accuser cannot forge an honest agent's signature, so it can submit only the honest agent's genuine answers;
  an honest member with an honest anchor majority then votes *block* with probability at most `ℓ`, so `ε ≤ ℓ` in
  Theorem 2;
* a Sybil that lies at rate `λ > 1 − α` is detected by each such member with probability `δ → 1` as `m` grows.
So Theorem 2 holds with `ε` and `δ` defined on submitted items, and its probe-indistinguishability premise is no
longer needed.

The test must use **all** audited items. If reporters submitted only the items behind their disagreements, an honest
agent's occasional genuine errors would look exactly like lies (selection bias). In our first transcript design they
did, and honest agents were convicted for ordinary errors (study 20b).

### Strategic Sybils

The counting argument of Proposition 7 still show that pacing does not raise the damage per identity
*before detection*. But with sharing, a new strategy appears. A *budget* Sybil that tells each victim at most `κ − 1`
lies never triggers a local block, hence never a report, hence never a committee. Its damage is `κ − 1` lies per
victim. That is bounded, but it multiplies with the number of victims; Proposition 13 closes this gap. Study 20
measures this strategy and a Q-learning attacker against every variant.

## II.4 Stated limits of Part II

1. With probe evidence, probe tasks must be indistinguishable from ordinary tasks; a Sybil that is honest towards
   strangers passes them. Transcript evidence (Proposition 14) removes this premise but needs signed answers.
2. Seats are drawn uniformly (A1). On-chain this is the VRF sampler. Grinding committee draws breaks
   Theorem 2 exactly as it breaks Theorem 1.
3. Theorem 2 treats seats as independent. Sampling without replacement only helps.
4. The threshold `ρ < 1/2` is a property of the model (members judge through their own anchors). Members with
   verifiable answers would not need anchors to judge, and the threshold would move to `p_s < 1/2` directly.

---

# Part III. Verifiable answers: beyond the threshold, and selecting good agents

Corollary 2 says that no vote can share evidence safely and usefully once about a third of the identities are
Sybils. The way past it is evidence that needs no vote: answers whose correctness can be checked with certainty
(a cryptographic proof of the computation, a test suite, a recomputation, a checkable certificate). Checking costs
something, so an agent verifies each served answer only with probability `v`.

## III.1 Model

* A verified answer is corrected if wrong, and the verification is public and unforgeable: anyone can re-check it.
* **Proof ban.** An agent is blocked network-wide when its verified errors among its last `W` verified answers are
  implausible for an honest agent: `Pr[Bin(m, ε_h) ≥ k] < ℓ_b` for honest error rate `ε_h`, `k` errors in the last
  `m ≤ W` verified answers, and a small level `ℓ_b`. No committee and no vote are involved.
* **Verified reputation.** Card-based selection adds `w · (s_c − 1/2)` to a candidate's score, where `s_c` is its
  verified success rate (Laplace prior), so rank can be earned only by verified correct work.

## III.2 Results

### Proposition 15 (proof-carrying evidence works at any Sybil share)

* *Safety.* A ban needs verified errors, which a Sybil cannot forge for an honest agent. An honest agent is banned
  only if its own genuine errors fail the test. This happens with probability at most `ℓ_b` per test (times the
  number of tests over the horizon, a union bound), whatever `B` is.
* *Liveness and damage.* Let `k*` be the smallest number of errors that fails the test within the window
  (for `ε_h = 0.05` and `ℓ_b = 10⁻⁴`: `k* = 4` for an identity whose verified answers are all lies, rising to 7 when
  its window holds 20 verified answers). A Sybil lying at rate `λ` is banned after an expected `k*/(vλ)` served
  answers. It makes honest agents use an expected `k*(1 − v)/v` wrong answers before the ban (for `λ = 1`), counted
  over the whole network. Total damage is therefore at most `B · k*(1 − v)/v`, independent of `N` and of the
  Sybil share.

*Proof.* Verified outcomes are independent of the Sybil's choices given the coin `v`, which the Sybil does not see.
Each lie is caught with probability `v`, and is then corrected (no damage) and counted. The ban test fires at the
latest when `k*` caught lies fall in one window. The number of uncaught lies before the `k*`-th caught one has
expectation `k*(1 − v)/v` (negative binomial). Safety follows from the unforgeability of verifications and the level
of the test. ∎

**Consequence.** Verification replaces the one-third threshold with a *cost* trade-off. Damage scales with `B/v`.
To keep total damage below `D`, verify a fraction `v ≥ B k*/(D + B k*)` of answers.

### Proposition 16 (verified reputation selects good agents, but cannot stop sleepers alone)

A Sybil earns verified reputation only by giving verified-correct answers. If it lies at rate `λ`, its verified
success rate converges to `1 − λ`, below an honest agent's `1 − ε_h`, so reputation-weighted selection prefers
honest agents among similar cards. A *sleeper* that answers truthfully until it is trusted earns reputation
legitimately and is harmless while it does. Once it lies, reputation alone reacts only as fast as its record
decays. The proof ban, which looks only at the recent window, is what removes it.

## III.3 Stated limits of Part III

1. Verifiable answers are modelled, not implemented. Cryptographic proofs of LLM inference are currently slow and
   costly for large models; tests and recomputation apply only to some tasks. The study reports the verification
   rate needed, which is the quantity a deployment must pay for.
2. The ban test assumes a known upper bound `ε_h` on honest error rates; honest agents that are worse than that
   bound are banned, which is a correct verdict on unreliable service rather than framing.
3. Verification protects answers. Eclipse of the *working set* (who an agent talks to) is still governed by
   Theorem 1. Verified reputation shifts selection towards proven-good agents but does not replace anchors.
