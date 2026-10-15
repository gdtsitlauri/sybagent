"""Fetch real ERC-8004 agent registration files (agent cards) for a random sample of registered agents.

Read-only: eth_call tokenURI(agentId) on the identity registry, then resolve the URI (data:, https://, ipfs://).
Writes one JSON line per agent with the name, description and declared services/skills.

    python scripts/fetch_agent_cards.py --chain base --n 3000 --out data/raw/erc8004/base_cards.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sybagent.data.chain import IDENTITY, RPC, RPCS  # noqa: E402

TOKEN_URI = "0xc87b56dd"          # tokenURI(uint256)
IPFS = "https://ipfs.io/ipfs/"


def token_uri(rpc: RPC, agent: int) -> str:
    data = TOKEN_URI + agent.to_bytes(32, "big").hex()
    out = bytes.fromhex(rpc.call("eth_call", [{"to": IDENTITY, "data": data}, "latest"])[2:])
    if len(out) < 64:
        return ""
    n = int.from_bytes(out[32:64], "big")
    return out[64:64 + n].decode("utf-8", errors="replace")


def resolve(uri: str, client: httpx.Client) -> dict | None:
    try:
        if uri.startswith("data:"):
            head, _, body = uri.partition(",")
            raw = base64.b64decode(body) if ";base64" in head else urllib.parse.unquote(body).encode()
            return json.loads(raw)
        if uri.startswith("ipfs://"):
            uri = IPFS + uri[len("ipfs://"):]
        if uri.startswith("http"):
            r = client.get(uri, timeout=15, follow_redirects=True)
            if r.status_code == 200 and len(r.content) < 2_000_000:
                return r.json()
    except Exception:
        return None
    return None


def card_text(doc: dict) -> dict:
    services = doc.get("services") or doc.get("endpoints") or []
    skills = []
    for s in services if isinstance(services, list) else []:
        if isinstance(s, dict):
            for k in ("skills", "domains", "capabilities"):
                v = s.get(k)
                if isinstance(v, list):
                    skills += [str(x) for x in v][:20]
    return {"name": str(doc.get("name", ""))[:200], "description": str(doc.get("description", ""))[:2000],
            "skills": skills[:40], "n_services": len(services) if isinstance(services, list) else 0}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[3]
    ids = pd.read_csv(root / "data" / "raw" / "erc8004" / f"{args.chain}_identities.csv", dtype={"agent": str})
    sample = ids["agent"].drop_duplicates().sample(min(args.n, len(ids)), random_state=args.seed).astype(int)
    rpc = RPC(RPCS[args.chain])
    client = httpx.Client(headers={"User-Agent": "SybAgent-research (read-only)"})

    def one(agent: int) -> dict:
        try:
            uri = token_uri(rpc, agent)
        except Exception:
            uri = ""
        kind = uri.split(":", 1)[0] if ":" in uri else ("empty" if not uri else "other")
        doc = resolve(uri, client) if uri else None
        row = {"chain": args.chain, "agent": agent, "uri_kind": kind, "resolved": doc is not None}
        if isinstance(doc, dict):
            row.update(card_text(doc))
        return row

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, sample))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    df = pd.DataFrame(rows)
    print(df["uri_kind"].value_counts().to_dict(), "resolved", int(df["resolved"].sum()),
          "with description", int((df.get("description", pd.Series(dtype=str)).fillna("").str.len() > 20).sum()))


if __name__ == "__main__":
    main()
