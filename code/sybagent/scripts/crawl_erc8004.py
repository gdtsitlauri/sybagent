"""Read-only crawl of ERC-8004 registries via public JSON-RPC.

    python scripts/crawl_erc8004.py --chain base --out data/raw/erc8004/base_feedback.csv
    python scripts/crawl_erc8004.py --chain base --kind identity --out data/raw/erc8004/base_identities.csv

feedback: ReputationRegistry NewFeedback / FeedbackRevoked logs; identity: IdentityRegistry mints (ERC-721 Transfer
from the zero address), i.e. agent registrations.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sybagent.data.chain import (IDENTITY, NEW_FEEDBACK, REPUTATION, REVOKED, RPC, RPCS, TRANSFER,  # noqa: E402
                                     ZERO_TOPIC, block_at_time, decode_mint, decode_new_feedback,
                                     decode_revoked, iter_logs)

FEEDBACK_COLS = ["chain", "block", "tx", "rater", "agent", "feedback_index", "score", "tag1", "tag2", "revoked"]
IDENTITY_COLS = ["chain", "block", "tx", "agent", "owner"]


def crawl(chain: str, out: Path, start: int | None, step: int, since: str, kind: str, end: int | None = None) -> None:
    rpc = RPC(RPCS[chain])
    latest = rpc.head() if end is None else end
    if start is None:
        ts = int(datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp())
        start = block_at_time(rpc, ts, latest)
    print(f"{chain} {kind}: blocks {start}..{latest}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    revoked: set[tuple[str, str, int]] = set()
    if kind == "identity":
        address, topics = IDENTITY, [TRANSFER, ZERO_TOPIC]
    else:
        address, topics = REPUTATION, [[NEW_FEEDBACK, REVOKED]]
    for i, (hi, logs) in enumerate(iter_logs(rpc, address, topics, start, latest, step)):
        for lg in logs:
            if kind == "identity":
                rows.append({"chain": chain, **decode_mint(lg)})
            elif lg["topics"][0] == NEW_FEEDBACK:
                rows.append({"chain": chain, **decode_new_feedback(lg)})
            else:
                revoked.add(decode_revoked(lg))
        if (i + 1) % 500 == 0:
            print(f"  {len(rows)} rows", flush=True)
            _write(out, rows, revoked, kind)
    _write(out, rows, revoked, kind)
    print(f"{chain}: {len(rows)} {kind} rows -> {out} (last block {latest})", flush=True)


def _write(out: Path, rows: list[dict], revoked: set, kind: str) -> None:
    cols = IDENTITY_COLS if kind == "identity" else FEEDBACK_COLS
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            if kind == "identity":
                w.writerow(r)
            else:
                w.writerow({**r, "revoked": (r["agent"], r["rater"], r["feedback_index"]) in revoked})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", choices=sorted(RPCS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", choices=["feedback", "identity"], default="feedback")
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--step", type=int, default=1_000)
    ap.add_argument("--since", default="2025-12-01")
    args = ap.parse_args()
    crawl(args.chain, Path(args.out), args.start, args.step, args.since, args.kind, args.end)


if __name__ == "__main__":
    sys.exit(main())
