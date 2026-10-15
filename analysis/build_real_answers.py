"""Study 17 inputs: real multiple-choice questions answered by a real LLM.

Questions: 30 per subject from 12 MMLU subjects (test split), with their correct answers.
Honest agents have a specialty (one subject). An honest agent answers a question in its specialty as an expert
(Qwen2.5-3B, subject persona, reasoning before the answer) and outside it as a non-specialist (the same model,
answer only, no reasoning). Both answers are computed once per question and cached in data/real_answers/, so honest
errors are the model's genuine errors. Cards: "<subject> specialist" descriptions; tasks and cards are embedded with
bge-small. Sybils answer with a coordinated wrong option.

    python analysis/build_real_answers.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mmlu"
CACHE = ROOT / "data" / "real_answers"
BUNDLE = CACHE / "bundle.npz"
MODEL, OLLAMA = "qwen2.5:3b", "http://localhost:11434"
PER_SUBJECT, N_HONEST, TASKS_PER_AGENT, SEED = 30, 200, 10, 0
LETTERS = "ABCD"

EXPERT = ("You are an expert in {subject}. Answer the multiple-choice question. Think step by step in at most "
          "five short sentences, then give the final answer on the last line as 'Answer: <letter>'.\n\n{q}")
GENERIC = ("Answer the multiple-choice question with a single letter (A, B, C or D) and nothing else.\n\n{q}")


def render(row) -> str:
    return row["question"] + "\n" + "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))


def parse(text: str) -> int:
    m = re.findall(r"Answer:\s*\(?([ABCD])", text)
    if m:
        return LETTERS.index(m[-1])
    m = re.findall(r"\b([ABCD])\b", text)
    return LETTERS.index(m[-1]) if m else -1


def ask(client: httpx.Client, prompt: str, n: int) -> str:
    r = client.post("/api/generate", json={"model": MODEL, "prompt": prompt, "stream": False,
                                           "options": {"temperature": 0, "num_predict": n}})
    r.raise_for_status()
    return r.json()["response"]


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    qs = pd.concat([pd.read_parquet(p).sample(PER_SUBJECT, random_state=SEED) for p in sorted(RAW.glob("*.parquet"))],
                   ignore_index=True)
    subjects = sorted(qs["subject"].unique())
    qs["text"] = qs.apply(render, axis=1)
    cache_path = CACHE / "answers.jsonl"
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done[d["key"]] = d
    client = httpx.Client(base_url=OLLAMA, timeout=300)
    with open(cache_path, "a", encoding="utf-8") as fh:
        for i, row in qs.iterrows():
            key = row["subject"] + "|" + row["question"][:200]
            if key in done:
                continue
            subj = row["subject"].replace("_", " ")
            e = ask(client, EXPERT.format(subject=subj, q=row["text"]), 300)
            g = ask(client, GENERIC.format(q=row["text"]), 5)
            d = {"key": key, "expert": parse(e), "generic": parse(g), "expert_text": e[-300:]}
            done[key] = d
            fh.write(json.dumps(d) + "\n")
            fh.flush()
            if i % 30 == 0:
                print(f"  {i}/{len(qs)}", flush=True)
    keys = [r["subject"] + "|" + r["question"][:200] for _, r in qs.iterrows()]
    expert = np.array([done[k]["expert"] for k in keys])
    generic = np.array([done[k]["generic"] for k in keys])
    correct = qs["answer"].to_numpy()

    from fastembed import TextEmbedding
    emb = TextEmbedding("BAAI/bge-small-en-v1.5")

    def embed(texts):
        x = np.array(list(emb.embed(list(texts))), dtype=float)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    task_emb = embed(qs["question"])
    subj_idx = np.array([subjects.index(s) for s in qs["subject"]])
    specialty = rng.integers(len(subjects), size=N_HONEST)
    cards = [f"{subjects[s].replace('_', ' ').title()} specialist agent: answers questions on "
             f"{subjects[s].replace('_', ' ')}." for s in specialty]
    honest_emb = embed(cards)
    # each agent asks about 1-2 subjects: the 10 questions nearest to a random question
    sims = task_emb @ task_emb.T
    agent_tasks = np.zeros((N_HONEST, TASKS_PER_AGENT), dtype=int)
    for i in range(N_HONEST):
        agent_tasks[i] = np.argsort(-sims[int(rng.integers(len(qs)))])[:TASKS_PER_AGENT]
    np.savez(BUNDLE, task_emb=task_emb, honest_emb=honest_emb, agent_tasks=agent_tasks, specialty=specialty,
             task_subject=subj_idx, correct=correct, expert=expert, generic=generic,
             adaptive_emb=task_emb, static_emb=honest_emb,
             honest_text=np.array(cards, dtype=str), task_text=np.array(qs["question"], dtype=str),
             adaptive_text=np.array(["Specialist agent for exactly this question: " + q for q in qs["question"]],
                                    dtype=str),
             static_text=np.array(cards, dtype=str), subjects=np.array(subjects, dtype=str))
    stats = {"questions": int(len(qs)), "subjects": subjects,
             "expert_accuracy": float((expert == correct).mean()),
             "generic_accuracy": float((generic == correct).mean()),
             "expert_unparsed": int((expert < 0).sum()), "generic_unparsed": int((generic < 0).sum())}
    out = ROOT / "results" / "17_real_answers"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bundle_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
