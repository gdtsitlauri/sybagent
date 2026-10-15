"""Mirror of an ERC-8004 deployment: agent registrations and feedback, kept in sync with the chain."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sybagent.data.chain import (IDENTITY, NEW_FEEDBACK, REPUTATION, REVOKED, RPC, RPCS, TRANSFER, ZERO_TOPIC,
                                     BlockClock, decode_mint, decode_new_feedback, decode_revoked, iter_logs)

DAY = 86_400


@dataclass
class Snapshot:
    """State of the deployment strictly before time `t` (unix seconds)."""
    t: float
    feedback: pd.DataFrame       # rows with ts < t, not revoked
    registered: np.ndarray       # agent ids registered before t (as str)

    @property
    def n_registered(self) -> int:
        return len(self.registered)


class TwinState:
    def __init__(self, chain: str, feedback: pd.DataFrame, identities: pd.DataFrame, clock: BlockClock,
                 last_block: int) -> None:
        self.chain = chain
        self.clock = clock
        self.last_block = last_block
        self.feedback = feedback
        self.identities = identities
        self._stamp()

    # ---- persistence ----
    @classmethod
    def load(cls, raw_dir: Path, twin_dir: Path, chain: str, rpc: RPC | None = None,
             use_mirror: bool = True) -> "TwinState":
        """Load the mirror (events synced so far) or, with use_mirror=False, the frozen crawl in `raw_dir`
        (used by the backtests so that they are reproducible)."""
        src = twin_dir if use_mirror and (twin_dir / f"{chain}_feedback.csv").exists() else raw_dir
        fb = pd.read_csv(src / f"{chain}_feedback.csv", dtype={"agent": str, "rater": str, "tag1": str, "tag2": str})
        ids = pd.read_csv(src / f"{chain}_identities.csv", dtype={"agent": str, "owner": str})
        fb = fb.drop(columns=["ts"], errors="ignore")
        ids = ids.drop(columns=["ts"], errors="ignore")
        clock_path = twin_dir / f"{chain}_clock.json"
        if clock_path.exists():
            meta = json.loads(clock_path.read_text())
            clock = BlockClock(meta["blocks"], meta["times"])
            last = int(meta.get("last_block", max(fb["block"].max(), ids["block"].max())))
        else:
            rpc = rpc or RPC(RPCS[chain])
            lo = int(min(fb["block"].min(), ids["block"].min()))
            # the two files were crawled at different times: resume from the older end, sync() drops duplicates
            last = int(min(fb["block"].max(), ids["block"].max()))
            clock = BlockClock.sample(rpc, lo, int(max(fb["block"].max(), ids["block"].max())))
        st = cls(chain, fb, ids, clock, last)
        if not clock_path.exists():
            st.save(twin_dir, tables=False)
        return st

    def save(self, twin_dir: Path, tables: bool = True) -> None:
        """Persist the clock and sync point; with `tables`, also the mirrored events (after a sync)."""
        twin_dir.mkdir(parents=True, exist_ok=True)
        (twin_dir / f"{self.chain}_clock.json").write_text(json.dumps({**self.clock.to_dict(),
                                                                       "last_block": self.last_block}))
        if tables:
            self.feedback.drop(columns=["ts"]).to_csv(twin_dir / f"{self.chain}_feedback.csv", index=False)
            self.identities.drop(columns=["ts"]).to_csv(twin_dir / f"{self.chain}_identities.csv", index=False)

    def _stamp(self) -> None:
        self.feedback["ts"] = self.clock(self.feedback["block"].to_numpy())
        self.identities["ts"] = self.clock(self.identities["block"].to_numpy())
        self.feedback = self.feedback.sort_values("block", kind="stable").reset_index(drop=True)
        self.identities = self.identities.sort_values("block", kind="stable").reset_index(drop=True)

    # ---- live synchronisation ----
    def sync(self, rpc: RPC | None = None, max_blocks: int | None = None) -> dict:
        """Pull registrations and feedback after `last_block` up to the chain head (read-only RPC)."""
        rpc = rpc or RPC(RPCS[self.chain])
        t0 = time.time()
        head = rpc.head()
        end = head if max_blocks is None else min(head, self.last_block + max_blocks)
        start = self.last_block + 1
        new_fb, new_ids, revoked = [], [], set()
        if end >= start:
            for _, logs in iter_logs(rpc, REPUTATION, [[NEW_FEEDBACK, REVOKED]], start, end, progress_every=0):
                for lg in logs:
                    if lg["topics"][0] == NEW_FEEDBACK:
                        new_fb.append({"chain": self.chain, **decode_new_feedback(lg), "revoked": False})
                    else:
                        revoked.add(decode_revoked(lg))
            for _, logs in iter_logs(rpc, IDENTITY, [TRANSFER, ZERO_TOPIC], start, end, progress_every=0):
                new_ids += [{"chain": self.chain, **decode_mint(lg)} for lg in logs]
            head_ts = rpc.timestamp(end)
            self.clock = BlockClock(list(self.clock.blocks) + [end], list(self.clock.times) + [head_ts])
        else:
            head_ts = rpc.timestamp(head)
        n_fb, n_id = len(self.feedback), len(self.identities)
        if new_fb:
            self.feedback = pd.concat([self.feedback, pd.DataFrame(new_fb)], ignore_index=True).drop_duplicates(
                subset=["block", "tx", "agent", "rater", "feedback_index"], keep="first", ignore_index=True)
        if new_ids:
            self.identities = pd.concat([self.identities, pd.DataFrame(new_ids)], ignore_index=True).drop_duplicates(
                subset=["block", "tx", "agent"], keep="first", ignore_index=True)
        if revoked:
            key = list(zip(self.feedback["agent"], self.feedback["rater"], self.feedback["feedback_index"]))
            self.feedback["revoked"] = self.feedback["revoked"].astype(bool) | pd.Series([k in revoked for k in key])
        self.last_block = max(self.last_block, end)
        self._stamp()
        return {"chain": self.chain, "from_block": start, "to_block": end, "head": head,
                "new_feedback": len(self.feedback) - n_fb, "new_registrations": len(self.identities) - n_id,
                "revocations": len(revoked),
                "rpc_calls": rpc.calls, "seconds": time.time() - t0,
                "state_lag_seconds": time.time() - head_ts}

    # ---- views ----
    @property
    def span(self) -> tuple[float, float]:
        return float(min(self.feedback["ts"].min(), self.identities["ts"].min())), \
            float(max(self.feedback["ts"].max(), self.identities["ts"].max()))

    def at(self, t: float) -> Snapshot:
        fb = self.feedback[(self.feedback["ts"] < t) & ~self.feedback["revoked"].astype(bool)]
        reg = self.identities.loc[self.identities["ts"] < t, "agent"].to_numpy()
        return Snapshot(t=t, feedback=fb, registered=reg)

    def window(self, t0: float, t1: float) -> pd.DataFrame:
        f = self.feedback
        return f[(f["ts"] >= t0) & (f["ts"] < t1) & ~f["revoked"].astype(bool)]
