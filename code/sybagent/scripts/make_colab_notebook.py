"""Generate notebooks/colab_llm.ipynb (larger LLM selectors on a Colab GPU)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    ("markdown", """# LLM-selector experiments on Colab

1. Runtime -> Change runtime type -> **T4 GPU**.
2. Run the first cell and upload `sybagent_colab.zip`.
3. Run all cells. Allow Google Drive access when asked (results are saved in `MyDrive/sybagent_colab_results`).
4. If Colab disconnects, run all cells again: finished runs are skipped.
5. At the end `colab_results.zip` downloads automatically."""),
    ("code", """from google.colab import files
up = files.upload()
!unzip -q -o sybagent_colab.zip -d /content/sybagent
%cd /content/sybagent
!pip -q install -e .
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", """!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time
server = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)
MODELS = ["qwen2.5:7b", "llama3.1:8b"]   # add "qwen2.5:14b" if time allows
for m in MODELS:
    !ollama pull {m}"""),
    ("code", """import copy, json, yaml, time
from pathlib import Path
from sybagent.config import from_dict
from sybagent.simulation import Simulation

spec = yaml.safe_load(open("configs/sweep_llm.yaml", encoding="utf-8-sig"))
base = spec["base"]
SEEDS = [0, 1, 2]

# results are kept on Google Drive so that a disconnect does not lose finished runs;
# re-running this cell continues where it stopped
from google.colab import drive
drive.mount("/content/drive")
out = Path("/content/drive/MyDrive/sybagent_colab_results")
out.mkdir(parents=True, exist_ok=True)
for m in MODELS:
    for adaptive in [True, False]:
        for dname in ["none", "proposed"]:
            for s in SEEDS:
                tag = f"{m.replace(':', '-')}_{'adaptive' if adaptive else 'static'}_{dname}_s{s}"
                if (out / f"{tag}.csv").exists():
                    continue
                cfg = copy.deepcopy(base)
                cfg["seed"] = s
                cfg["selection"]["llm_model"] = m
                cfg["adversary"]["adaptive_cards"] = adaptive
                cfg["defense"]["kind"] = dname
                t = time.time()
                sim = Simulation(from_dict(cfg))
                sim.run()
                sim.save(out / f"{tag}.csv")
                (out / f"{tag}.json").write_text(json.dumps({"model": m, "adaptive": adaptive, "defense": dname, "seed": s}))
                print(tag, f"{time.time() - t:.0f}s", "LLM calls:", sim.ctx.llm_calls)"""),
    ("code", """import pandas as pd
rows = []
for f in sorted(out.glob("*.csv")):
    if f.name == "summary.csv":
        continue
    meta = json.loads(f.with_suffix(".json").read_text())
    t = pd.read_csv(f).tail(5).mean(numeric_only=True)
    rows.append({**meta, "target_sybil_share": t["target_sybil_share"],
                 "target_min_honest_neighbors": t["target_min_honest_neighbors"],
                 "honest_mean_sybil_share": t["honest_mean_sybil_share"], "mean_quality": t["mean_quality"]})
summary = pd.DataFrame(rows)
summary.to_csv(out / "summary.csv", index=False)
summary.groupby(["model", "adaptive", "defense"]).mean(numeric_only=True)"""),
    ("code", """!cd "/content/drive/MyDrive" && zip -q -r /content/colab_results.zip sybagent_colab_results
files.download("/content/colab_results.zip")"""),
]


def main() -> None:
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, src in CELLS
        ],
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = ROOT / "notebooks" / "colab_llm.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
