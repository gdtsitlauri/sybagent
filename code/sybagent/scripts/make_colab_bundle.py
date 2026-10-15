"""Zip the code needed on Colab into notebooks/sybagent_colab.zip (forward-slash paths)."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "configs", "scripts", "tests", "pyproject.toml", "requirements.txt"]


def main() -> None:
    out = ROOT / "notebooks" / "sybagent_colab.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in INCLUDE:
            p = ROOT / item
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
            for f in files:
                if "__pycache__" in f.parts or f.suffix == ".pyc" or f.name.endswith(".egg-info"):
                    continue
                if any(part.endswith(".egg-info") for part in f.parts):
                    continue
                zf.write(f, f.relative_to(ROOT).as_posix())
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
