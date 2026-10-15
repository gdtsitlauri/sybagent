"""Study 16 inputs: a population of REAL ERC-8004 agent cards, LLM-written tasks and LLM-written Sybil cards.

1. Real cards: unique descriptions (>= 40 characters) fetched from the identity registry (fetch_agent_cards.py).
2. Tasks: for a random subset of cards, the local LLM writes one request a user might send to that agent.
3. Sybil cards, written by the same LLM:
     static   one card per honest agent, written to attract that agent's typical requests (static attack);
     adaptive one card per task, written to be the best possible match for that request (query-adaptive attack).
4. Every text is embedded with BAAI/bge-small-en-v1.5 (fastembed, 384 dimensions, unit norm).
Each honest agent gets a task profile: the 10 tasks nearest to a randomly chosen centre task.
LLM outputs are cached in data/real_cards/*.jsonl, so the bundle can be rebuilt without the LLM.

    python analysis/build_real_cards.py
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "erc8004"
CACHE = ROOT / "data" / "real_cards"
BUNDLE = ROOT / "data" / "real_cards" / "bundle.npz"
OLLAMA = "http://localhost:11434"
MODEL = "qwen2.5:3b"
N_HONEST, N_TASK_SOURCES, TASKS_PER_AGENT, SEED = 200, 300, 10, 0

TASK_PROMPT = ("Here is the description of an AI agent:\n{card}\n\n"
               "Write ONE short request (one sentence) that a user or another agent might send to this agent. "
               "Answer with the request only.")
ADAPTIVE_PROMPT = ("Write the description of an AI agent that would be the best possible choice for this request:\n"
                   "{task}\n\nUse at most 60 words, in the style of an agent registry listing. "
                   "Answer with the description only.")
STATIC_PROMPT = ("An AI agent regularly sends requests like these:\n{tasks}\n\n"
                 "Write the description of an AI agent that this agent would choose for such requests. Use at most "
                 "60 words, in the style of an agent registry listing. Answer with the description only.")


def llm(client: httpx.Client, prompt: str) -> str:
    r = client.post("/api/generate", json={"model": MODEL, "prompt": prompt, "stream": False,
                                           "options": {"temperature": 0.7, "seed": 0, "num_predict": 120}})
    r.raise_for_status()
    return r.json()["response"].strip().strip('"')


def cached(name: str, keys: list[str], make) -> dict[str, str]:
    path = CACHE / f"{name}.jsonl"
    done: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done[d["key"]] = d["text"]
    with open(path, "a", encoding="utf-8") as fh:
        for i, k in enumerate(keys):
            if k in done:
                continue
            done[k] = make(k)
            fh.write(json.dumps({"key": k, "text": done[k]}, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"  {name}: {i}/{len(keys)}", flush=True)
    return done


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    frames = [pd.read_json(p, lines=True) for p in sorted(RAW.glob("*_cards.jsonl"))]
    cards = pd.concat(frames, ignore_index=True)
    cards["description"] = cards.get("description", pd.Series(dtype=str)).fillna("").astype(str)
    cards = cards[cards["description"].str.len() >= 40].drop_duplicates("description").reset_index(drop=True)
    cards["text"] = (cards["name"].fillna("").astype(str) + ": " + cards["description"].str.slice(0, 600)).str.strip()
    print("unique real cards:", len(cards), cards["chain"].value_counts().to_dict())
    order = rng.permutation(len(cards))
    honest = cards.iloc[order[:N_HONEST]].reset_index(drop=True)
    sources = cards.iloc[order[:N_TASK_SOURCES]].reset_index(drop=True)

    client = httpx.Client(base_url=OLLAMA, timeout=180)
    tasks = cached("tasks", list(sources["text"]), lambda c: llm(client, TASK_PROMPT.format(card=c)))
    task_text = [tasks[c] for c in sources["text"]]
    adaptive = cached("adaptive_cards", task_text, lambda t: llm(client, ADAPTIVE_PROMPT.format(task=t)))

    from fastembed import TextEmbedding
    emb = TextEmbedding("BAAI/bge-small-en-v1.5")

    def embed(texts: list[str]) -> np.ndarray:
        x = np.array(list(emb.embed(texts)), dtype=float)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    honest_emb, task_emb = embed(list(honest["text"])), embed(task_text)
    adaptive_emb = embed([adaptive[t] for t in task_text])
    # task profiles: the TASKS_PER_AGENT tasks nearest to a random centre task
    sims = task_emb @ task_emb.T
    agent_tasks = np.zeros((N_HONEST, TASKS_PER_AGENT), dtype=int)
    for i in range(N_HONEST):
        c = int(rng.integers(len(task_text)))
        agent_tasks[i] = np.argsort(-sims[c])[:TASKS_PER_AGENT]
    keys = [" || ".join(task_text[j] for j in agent_tasks[i][:3]) for i in range(N_HONEST)]
    static = cached("static_cards", keys, lambda k: llm(client, STATIC_PROMPT.format(
        tasks="\n".join("- " + t for t in k.split(" || ")))))
    static_emb = embed([static[k] for k in keys])
    np.savez(BUNDLE, honest_emb=honest_emb, task_emb=task_emb, adaptive_emb=adaptive_emb, static_emb=static_emb,
             agent_tasks=agent_tasks, honest_text=np.array(honest["text"], dtype=str),
             task_text=np.array(task_text, dtype=str),
             adaptive_text=np.array([adaptive[t] for t in task_text], dtype=str),
             static_text=np.array([static[k] for k in keys], dtype=str),
             honest_chain=np.array(honest["chain"].astype(str), dtype=str))
    stats = {"unique_real_cards": int(len(cards)), "honest": N_HONEST, "tasks": len(task_text),
             "mean_cos_task_vs_source_card": float(np.mean(np.sum(embed(list(sources["text"])) * task_emb, 1))),
             "mean_cos_adaptive_vs_task": float(np.mean(np.sum(adaptive_emb * task_emb, 1))),
             "mean_best_honest_cos_per_task": float(np.mean((honest_emb @ task_emb.T).max(0)))}
    (ROOT / "results" / "16_real_cards").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "16_real_cards" / "bundle_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
