"""Merge identity crawls run in parallel block ranges (crawl_erc8004.py --start/--end) into one file.

    python scripts/merge_crawl_parts.py data/raw/erc8004 base
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    folder, chain = Path(sys.argv[1]), sys.argv[2]
    parts = sorted(folder.glob(f"{chain}_identities_part*.csv"))
    df = pd.concat([pd.read_csv(p, dtype={"agent": str, "owner": str}) for p in parts], ignore_index=True)
    n = len(df)
    df = df.drop_duplicates(subset=["block", "tx", "agent"]).sort_values(["block", "agent"], kind="stable")
    df.to_csv(folder / f"{chain}_identities.csv", index=False)
    print(f"{chain}: {len(parts)} parts, {n} rows, {len(df)} after removing overlaps, "
          f"blocks {df['block'].min()}..{df['block'].max()}")


if __name__ == "__main__":
    main()
