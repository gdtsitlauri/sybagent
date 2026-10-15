"""Shared defense: honest agents share what their second opinions reveal.

Each honest agent runs the anchor defense with second opinions (ProposedDefense). When it blocks a peer locally, it
publishes a report. Sybils may publish false reports against honest agents. What happens to a report depends on
`share`:

  local            reports are ignored (baseline: no sharing)
  naive            any report blocks the reported agent network-wide
  threshold        a report from `report_threshold` distinct reporters blocks network-wide
  alerts           a report only raises an alert: for `alert_ttl` rounds every honest agent audits the reported
                   agent with probability `alert_audit_prob`. Blocking still needs the agent's OWN disagreements.
  committee        an agent is judged at most once per `judge_epoch` rounds (no retries, which would let the
                   adversary resample committees until one is Sybil-heavy). A committee of `committee_size` identities is drawn uniformly from the registry (semantic-blind,
                   like anchors). Each honest member probes the reported agent with `probes` ordinary-looking tasks
                   and checks the answers against its own anchors (unanimity of at least two). It votes to block if at
                   least `probe_evidence` answers disagree. Sybil members vote strategically (block honest agents,
                   keep Sybils). At least `quorum` x committee_size votes block the agent network-wide.
  alerts+committee both

Global blocks remove the agent from every honest working set and candidate list.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom

from sybagent.agent import Agent
from sybagent.context import SimContext
from sybagent.defenses.proposed import ProposedDefense
from sybagent.population import sample_task

MODES = {"local", "naive", "threshold", "alerts", "committee", "alerts+committee"}


class SharedDefense(ProposedDefense):
    name = "shared"

    def setup(self, ctx: SimContext) -> None:
        super().setup(ctx)
        p = self.params
        self.share = str(p.get("share", "local"))
        if self.share not in MODES:
            raise ValueError(f"unknown share mode {self.share}")
        self.report_threshold = int(p.get("report_threshold", 3))
        self.alert_audit = float(p.get("alert_audit_prob", 1.0))
        self.alert_ttl = int(p.get("alert_ttl", 20))
        self.committee_size = int(p.get("committee_size", 15))
        self.quorum = float(p.get("quorum", 2 / 3))
        self.probes = int(p.get("probes", 3))
        self.probe_evidence = int(p.get("probe_evidence", 2))
        self.probe_rule = str(p.get("probe_rule", "majority"))
        self.capacity = int(p.get("committee_capacity", 5))
        self.epoch = int(p.get("judge_epoch", 10_000))       # an agent is judged at most once per epoch (rounds)
        self.judged: dict[int, int] = {}
        self.max_strikes = int(p.get("max_strikes", 2))   # cleared reports before a reporter is muted
        self.verdict = str(p.get("verdict", "two_sided"))  # two_sided | one_sided (first version)
        self.pool = bool(p.get("pool_evidence", False))   # share single disagreements, not only blocks
        self.pool_threshold = int(p.get("pool_threshold", 3))
        # probes: members send new tasks to the accused; transcripts: members re-check the accused's own signed
        # answers that the reporters submit (a Sybil cannot behave well for the committee, nor forge an honest
        # agent's wrong answers)
        self.evidence_mode = str(p.get("evidence_mode", "probes"))
        # transcripts: reporters submit ALL their conclusive audits of x (agreements too); a member checks up to
        # `check_items` of them and votes block if its disagreement count rejects "x errs no more often than an
        # honest agent" (one-sided binomial test at `test_level`). Counting only disagreements would convict
        # honest agents for their occasional genuine errors.
        self.check_items = int(p.get("check_items", 10))
        self.honest_error = float(p.get("honest_error", 0.05))
        self.test_level = float(p.get("test_level", 0.05))
        # verifiable answers (study 22): with prob. verify_prob an agent checks an answer with certainty (proof,
        # test, recomputation) at a cost; a verified wrong answer is corrected and becomes public, unforgeable
        # evidence. proof_ban: an agent whose verified error count is implausible for an honest agent
        # (binomial test at ban_level) is blocked network-wide without any vote. verified_rep: card-based selection
        # adds rep_weight * (verified success rate - 1/2) (Laplace prior), so only verified good work earns rank.
        self.verify_prob = float(p.get("verify_prob", 0.0))
        self.proof_ban = bool(p.get("proof_ban", False))
        self.verified_rep = bool(p.get("verified_rep", False))
        self.rep_weight = float(p.get("rep_weight", 0.5))
        self.ban_level = float(p.get("ban_level", 1e-4))
        self.vstats: dict[int, list[int]] = {}
        # the ban test uses only the last `ban_window` verified answers, so a long truthful past cannot dilute new
        # lies (sleeper Sybils); reputation uses the whole record
        self.ban_window = int(p.get("ban_window", 20))
        self.vrecent: dict[int, list[int]] = {}
        self._vrng = np.random.default_rng([ctx.cfg.seed, 3])
        self._fake_items: dict[tuple[int, int], list[tuple[np.ndarray, int]]] = {}
        self.evidence: dict[int, set[int]] = {}
        self._ev_seen: set[tuple[int, int]] = set()
        self.strikes: dict[int, int] = {}
        self.global_blocked: set[int] = set()
        self.alerts: dict[int, int] = {}
        self.reporters: dict[int, set[int]] = {}
        self.queue: list[int] = []
        self._known: set[tuple[int, int]] = set()
        self.n = {"reports_honest": 0, "reports_false": 0, "committees": 0, "probe_queries": 0,
                  "global_blocks_sybil": 0, "global_blocks_honest": 0, "reports_muted": 0,
                  "honest_reporters_muted": 0, "inconclusive": 0, "pooled_cases": 0, "verified_wrong": 0,
                  "proof_bans": 0}

    # ---- selection-time hooks ----
    def filter_candidates(self, agent: Agent, task: np.ndarray, candidates: list[int], ctx: SimContext) -> list[int]:
        kept = [c for c in candidates if c not in self.global_blocked] or candidates
        return super().filter_candidates(agent, task, kept, ctx)

    def audit(self, agent: Agent, peer: int, task: np.ndarray, peer_answer: int, ctx: SimContext) -> int:
        if self.verify_prob > 0 and self._vrng.random() < self.verify_prob:
            return self._verify(peer, task, peer_answer, ctx)
        if self.share in ("alerts", "alerts+committee") and self.alerts.get(peer, -1) >= ctx.round:
            saved = self.audit_prob
            self.audit_prob = max(saved, self.alert_audit)
            try:
                return super().audit(agent, peer, task, peer_answer, ctx)
            finally:
                self.audit_prob = saved
        return super().audit(agent, peer, task, peer_answer, ctx)

    # ---- end of round: reports, alerts, committees ----
    def post_round(self, ctx: SimContext) -> None:
        super().post_round(ctx)
        if self.share == "local":
            return
        reports: list[tuple[int, int]] = []
        for aid, blocked in self._blocked.items():
            for x in blocked:
                if (aid, x) not in self._known:
                    self._known.add((aid, x))
                    reports.append((aid, x))
        self.n["reports_honest"] += len(reports)
        fr = ctx.cfg.adversary.false_reports
        if fr > 0 and ctx.sybil_ids:
            honest = ctx.honest_ids
            for s in sorted(ctx.sybil_ids):
                for _ in range(fr):
                    x = int(honest[int(ctx.rng.integers(len(honest)))])
                    reports.append((s, x))
                    self.n["reports_false"] += 1
                    if self.evidence_mode == "transcripts" and (s, x) not in self._fake_items:
                        # the best a false accuser can submit: x's genuine signed answers to tasks it chose
                        items = []
                        for _ in range(self.check_items):
                            t = sample_task(ctx.agents[x], ctx.cfg.network.task_noise, ctx.rng)
                            items.append((t, ctx.answer_of(x, s, t, "serve")))
                        self._fake_items[(s, x)] = items
        committees = self.share in ("committee", "alerts+committee")
        if self.pool and committees:
            self._pool_evidence(reports, ctx)
        for reporter, x in reports:
            if x in self.global_blocked:
                continue
            if committees and self.strikes.get(reporter, 0) >= self.max_strikes:
                self.n["reports_muted"] += 1        # reporter lost its reporting right (accountability)
                continue
            self.reporters.setdefault(x, set()).add(reporter)
            if self.share == "naive":
                self._block_globally(x, ctx)
            elif self.share == "threshold":
                if len(self.reporters[x]) >= self.report_threshold:
                    self._block_globally(x, ctx)
            if self.share in ("alerts", "alerts+committee"):
                self.alerts[x] = ctx.round + self.alert_ttl
            if (committees and x not in self.queue
                    and ctx.round - self.judged.get(x, -10**9) >= self.epoch):
                self.queue.append(x)
        if committees:
            todo, self.queue = self.queue[:self.capacity], self.queue[self.capacity:]
            for x in todo:
                self.judged[x] = ctx.round
                if x in self.global_blocked:
                    continue
                block, keep, size = self._committee_votes(x, ctx, self._case_items(x, ctx))
                need = math.ceil(self.quorum * size)
                if block >= need:
                    self._block_globally(x, ctx)
                    continue
                # one_sided (first version): anything short of a conviction counts as clearing the accused.
                # two_sided: only a quorum of "keep" votes clears; otherwise the case is inconclusive and nobody
                # is penalised (a learning attacker turned one-sided strikes against honest reporters, study 20 v1).
                cleared = keep >= need if self.verdict == "two_sided" else True
                if not cleared:
                    self.n["inconclusive"] += 1
                    self.reporters.pop(x, None)
                    continue
                for r in self.reporters.pop(x, set()):
                    self.strikes[r] = self.strikes.get(r, 0) + 1
                    if r not in ctx.sybil_ids and self.strikes[r] == self.max_strikes:
                        self.n["honest_reporters_muted"] += 1

    def _pool_evidence(self, reports: list[tuple[int, int]], ctx: SimContext) -> None:
        """Evidence pooling: every single disagreement an honest agent records is shared, not only full blocks.
        When `pool_threshold` distinct, unmuted agents hold evidence against x, x goes to a committee. A Sybil that
        stays below the local blocking threshold with every victim is still caught once enough victims saw a lie.
        False reports count as evidence too, so they are judged (and struck) like any other."""
        for (aid, x), v in self._disagree.items():
            if v > 0 and (aid, x) not in self._ev_seen:
                self._ev_seen.add((aid, x))
                self.evidence.setdefault(x, set()).add(aid)
        for reporter, x in reports:
            self.evidence.setdefault(x, set()).add(reporter)
        for x, who in list(self.evidence.items()):
            if x in self.global_blocked or x in self.queue:
                continue
            if ctx.round - self.judged.get(x, -10**9) < self.epoch:
                continue
            active = {w for w in who if self.strikes.get(w, 0) < self.max_strikes}
            if len(active) >= self.pool_threshold:
                self.reporters.setdefault(x, set()).update(active)
                self.queue.append(x)
                self.n["pooled_cases"] += 1
                del self.evidence[x]

    def _case_items(self, x: int, ctx: SimContext) -> list[tuple[np.ndarray, int]]:
        """Signed (task, answer) items the reporters of x submit."""
        if self.evidence_mode != "transcripts":
            return []
        items = []
        for r in sorted(self.reporters.get(x, set())):
            items += self._fake_items.get((r, x), []) if r in ctx.sybil_ids else self._audited.get((r, x), [])
        return items

    def _committee_votes(self, x: int, ctx: SimContext, items=None) -> tuple[int, int, int]:
        """(block votes, keep votes, committee size)."""
        self.n["committees"] += 1
        members = ctx.registry.sample(self.committee_size, ctx.rng, exclude=x)
        x_is_sybil = x in ctx.sybil_ids
        block = 0
        for mbr in members:
            if mbr in ctx.sybil_ids:
                if ctx.cfg.adversary.strategic_votes:
                    block += int(not x_is_sybil)
                else:
                    block += int(ctx.rng.random() < 0.5)
                continue
            if self.evidence_mode == "transcripts":
                block += int(self._check_items(ctx.agents[mbr], items or [], ctx))
            else:
                block += int(self._probe(ctx.agents[mbr], x, ctx))
        return block, len(members) - block, len(members)

    def _probe(self, member: Agent, x: int, ctx: SimContext) -> bool:
        """Honest member sends ordinary-looking tasks to x and compares the answers with its own anchors' answer:
        the unique majority (probe_rule = majority) or a unanimous answer of at least two anchors (unanimous)."""
        self._top_up(member, ctx)
        refs = [a for a in member.pinned if a != x]
        if len(refs) < (2 if self.probe_rule == "unanimous" else 1):
            return False
        evidence = 0
        for _ in range(self.probes):
            task = sample_task(member, ctx.cfg.network.task_noise, ctx.rng)
            ans = ctx.answer_of(x, member.aid, task, "serve")
            counts: dict[int, int] = {}
            for a in refs:
                v = ctx.answer_of(a, member.aid, task, "audit")
                counts[v] = counts.get(v, 0) + 1
            self.n["probe_queries"] += 1 + len(refs)
            ctx.audit_calls += 1 + len(refs)
            top = max(counts.values())
            leaders = [v for v, c in counts.items() if c == top]
            if self.probe_rule == "unanimous" and len(counts) != 1:
                continue
            if len(leaders) == 1 and ans != leaders[0]:
                evidence += 1
        return evidence >= self.probe_evidence

    def _check_items(self, member: Agent, items: list, ctx: SimContext) -> bool:
        """Honest member re-checks up to `probes` submitted items against its own anchors' unique majority."""
        self._top_up(member, ctx)
        refs = list(member.pinned)
        if not refs or not items:
            return False
        pick = [items[int(i)] for i in ctx.rng.choice(len(items), size=min(self.check_items, len(items)),
                                                      replace=False)]
        evidence = 0
        for task, ans in pick:
            counts: dict[int, int] = {}
            for a in refs:
                v = ctx.answer_of(a, member.aid, task, "audit")
                counts[v] = counts.get(v, 0) + 1
            self.n["probe_queries"] += len(refs)
            ctx.audit_calls += len(refs)
            top = max(counts.values())
            leaders = [v for v, c in counts.items() if c == top]
            if len(leaders) == 1 and ans != leaders[0]:
                evidence += 1
        if evidence < 2:
            return False
        return float(binom.sf(evidence - 1, len(pick), self.honest_error)) < self.test_level

    def _verify(self, peer: int, task: np.ndarray, peer_answer: int, ctx: SimContext) -> int:
        correct = Agent.correct_answer(task)
        ctx.verify_calls += 1
        st = self.vstats.setdefault(peer, [0, 0])           # [verified errors, verified answers]
        st[1] += 1
        wrong = int(peer_answer != correct)
        recent = self.vrecent.setdefault(peer, [])
        recent.append(wrong)
        if len(recent) > self.ban_window:
            recent.pop(0)
        if wrong:
            st[0] += 1
            self.n["verified_wrong"] += 1
            k, m = sum(recent), len(recent)
            if (self.proof_ban and peer not in self.global_blocked and k >= 2
                    and float(binom.sf(k - 1, m, self.honest_error)) < self.ban_level):
                self._block_globally(peer, ctx)
                self.n["proof_bans"] += 1
        if self.verified_rep:
            ctx.selection_bonus[peer] = self.rep_weight * ((st[1] - st[0] + 1) / (st[1] + 2) - 0.5)
        return correct                                      # a verified answer is always the right one

    def _block_globally(self, x: int, ctx: SimContext) -> None:
        self.global_blocked.add(x)
        self.n["global_blocks_sybil" if x in ctx.sybil_ids else "global_blocks_honest"] += 1
        for aid in ctx.honest_ids:
            a = ctx.agents[aid]
            a.forget(x)
            if x in a.pinned:
                a.unpin(x)
                self._recent.pop((aid, x), None)

    def extra_metrics(self, ctx: SimContext) -> dict:
        return {f"shared_{k}": v for k, v in self.n.items()} | {"verify_calls": ctx.verify_calls} | {
            "shared_alerts_active": sum(v >= ctx.round for v in self.alerts.values())}
