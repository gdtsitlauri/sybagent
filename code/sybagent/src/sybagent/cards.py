"""Plain-text agent cards rendered from capability vectors (used only by the LLM selector)."""

from __future__ import annotations

import numpy as np

SKILLS = [
    "translation", "summarization", "code review", "data cleaning",
    "currency conversion", "scheduling", "legal drafting", "tax calculation",
    "image captioning", "web search", "SQL querying", "unit testing",
    "travel booking", "sentiment analysis", "report writing", "math solving",
]


def skill_names(dim: int) -> list[str]:
    if dim <= len(SKILLS):
        return SKILLS[:dim]
    return SKILLS + [f"skill-{i}" for i in range(len(SKILLS), dim)]


def render_card(aid: int, card: np.ndarray, top: int = 3) -> str:
    names = skill_names(len(card))
    order = np.argsort(-card)[:top]
    skills = ", ".join(names[i] for i in order)
    return f"Agent {aid}: offers {skills}."


def render_task(task: np.ndarray, top: int = 2) -> str:
    names = skill_names(len(task))
    order = np.argsort(-task)[:top]
    return "Task requiring " + " and ".join(names[i] for i in order) + "."
