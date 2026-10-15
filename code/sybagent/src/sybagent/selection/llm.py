"""LLM-based selector backed by a local Ollama server (no external services)."""

from __future__ import annotations

import json
import re

import numpy as np

from sybagent.agent import Agent
from sybagent.cards import render_card, render_task
from sybagent.config import SelectionConfig
from sybagent.context import SimContext
from sybagent.selection.base import Selector
from sybagent.selection.similarity import SimilaritySelector

PROMPT = (
    "You route tasks to collaborator agents.\n"
    "{task}\n"
    "Candidates:\n{cards}\n"
    "Pick the {k} best candidates for this task. "
    'Answer only with JSON: {{"ids": [<agent ids>]}}'
)


class LLMSelector(Selector):
    def __init__(self, cfg: SelectionConfig) -> None:
        super().__init__(cfg)
        self._fallback = SimilaritySelector(cfg)
        self._client = None

    def _http(self):
        if self._client is None:
            import httpx
            self._client = httpx.Client(base_url=self.cfg.llm_url, timeout=self.cfg.llm_timeout_s)
        return self._client

    def _ask(self, prompt: str) -> str:
        r = self._http().post(
            "/api/generate",
            json={"model": self.cfg.llm_model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        )
        r.raise_for_status()
        return r.json().get("response", "")

    @staticmethod
    def parse_ids(text: str, allowed: list[int], k: int) -> list[int]:
        ids: list[int] = []
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                ids = [int(x) for x in json.loads(m.group(0)).get("ids", [])]
            except (ValueError, TypeError, json.JSONDecodeError):
                ids = []
        if not ids:
            ids = [int(x) for x in re.findall(r"\d+", text)]
        allowed_set = set(allowed)
        out = []
        for i in ids:
            if i in allowed_set and i not in out:
                out.append(i)
            if len(out) == k:
                break
        return out

    def select(self, agent: Agent, task: np.ndarray, candidates: list[int], k: int, ctx: SimContext) -> list[int]:
        if not candidates:
            return []
        order = list(ctx.rng.permutation(candidates))  # avoid position bias from registry ranking
        def card(c: int) -> str:
            if c in ctx.adaptive_ids:
                text = ctx.adaptive_text.get(task.tobytes())   # real-card mode: LLM-written adaptive card
                return f"Agent {c}: {text}" if text else render_card(c, ctx.card_for(c, task))
            return ctx.agents[c].card_text

        cards = "\n".join(card(c) for c in order)
        label = ctx.task_label(task)
        prompt = PROMPT.format(task=("Task: " + label) if label else render_task(task), cards=cards, k=k)
        try:
            ctx.llm_calls += 1
            chosen = self.parse_ids(self._ask(prompt), order, k)
        except Exception:
            chosen = []
        if len(chosen) < k:
            rest = [c for c in candidates if c not in chosen]
            chosen += self._fallback.select(agent, task, rest, k - len(chosen), ctx)
        return chosen
