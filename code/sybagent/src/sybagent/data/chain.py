"""Read-only access to ERC-8004 registries through public JSON-RPC (used by the crawler and the digital twin)."""

from __future__ import annotations

import time
from typing import Iterator

import httpx
import numpy as np
from Crypto.Hash import keccak

REPUTATION = "0x8004BAa17C55a88189AE136b182e5fdA19dE9b63"
IDENTITY = "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432"   # ReputationRegistry.getIdentityRegistry() on both chains
RPCS = {
    # free endpoints that still serve historical logs (Tenderly public: <= 1000 blocks per eth_getLogs)
    "ethereum": ["https://gateway.tenderly.co/public/mainnet", "https://rpc.flashbots.net"],
    "base": ["https://gateway.tenderly.co/public/base"],
    "bsc": ["https://bsc-rpc.publicnode.com", "https://bsc.drpc.org"],
}


def topic(sig: str) -> str:
    k = keccak.new(digest_bits=256)
    k.update(sig.encode())
    return "0x" + k.hexdigest()


NEW_FEEDBACK = topic("NewFeedback(uint256,address,uint64,int128,uint8,string,string,string,string,string,bytes32)")
REVOKED = topic("FeedbackRevoked(uint256,address,uint64)")
TRANSFER = topic("Transfer(address,address,uint256)")
ZERO_TOPIC = "0x" + "0" * 64


class RPC:
    def __init__(self, urls: list[str]) -> None:
        self.urls = urls
        self.i = 0
        self.calls = 0
        self.client = httpx.Client(timeout=60)

    def call(self, method: str, params: list):
        last = None
        for attempt in range(6):
            url = self.urls[self.i % len(self.urls)]
            try:
                self.calls += 1
                r = self.client.post(url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
                r.raise_for_status()
                body = r.json()
                if "error" in body:
                    raise RuntimeError(str(body["error"]))
                return body["result"]
            except Exception as e:  # rotate endpoint and back off
                last = e
                self.i += 1
                time.sleep(min(0.5 * 2 ** attempt, 10))
        raise RuntimeError(f"{method} failed: {last}")

    def head(self) -> int:
        return int(self.call("eth_blockNumber", []), 16)

    def timestamp(self, block: int) -> int:
        return int(self.call("eth_getBlockByNumber", [hex(block), False])["timestamp"], 16)


def block_at_time(rpc: RPC, ts: int, latest: int) -> int:
    """First block with timestamp >= ts (headers only, works on non-archive nodes)."""
    lo, hi = 0, latest
    while lo < hi:
        mid = (lo + hi) // 2
        if rpc.timestamp(mid) >= ts:
            hi = mid
        else:
            lo = mid + 1
    return lo


def iter_logs(rpc: RPC, address: str, topics: list, start: int, end: int, step: int = 1000,
              progress_every: int = 500) -> Iterator[tuple[int, list[dict]]]:
    """Yield (last block of chunk, logs) over [start, end], halving the chunk on provider errors."""
    b = start
    n = 0
    while b <= end:
        hi = min(b + step - 1, end)
        try:
            logs = rpc.call("eth_getLogs", [{"address": address, "fromBlock": hex(b), "toBlock": hex(hi),
                                             "topics": topics}])
        except RuntimeError:
            if step > 100:
                step //= 2
                continue
            raise
        yield hi, logs
        b = hi + 1
        n += 1
        if progress_every and n % progress_every == 0:
            print(f"  block {hi}/{end}", flush=True)


def _word(data: bytes, i: int) -> bytes:
    return data[32 * i: 32 * (i + 1)]


def _string_at(data: bytes, offset: int) -> str:
    n = int.from_bytes(data[offset: offset + 32], "big")
    return data[offset + 32: offset + 32 + n].decode("utf-8", errors="replace")


def decode_new_feedback(log: dict) -> dict:
    data = bytes.fromhex(log["data"][2:])
    value = int.from_bytes(_word(data, 1), "big", signed=True)
    decimals = int.from_bytes(_word(data, 2), "big")
    return {
        "block": int(log["blockNumber"], 16),
        "tx": log["transactionHash"],
        "agent": str(int(log["topics"][1], 16)),
        "rater": "0x" + log["topics"][2][-40:],
        "feedback_index": int.from_bytes(_word(data, 0), "big"),
        "score": value / (10 ** decimals) if decimals < 40 else float("nan"),
        "tag1": _string_at(data, int.from_bytes(_word(data, 3), "big")),
        "tag2": _string_at(data, int.from_bytes(_word(data, 4), "big")),
    }


def decode_revoked(log: dict) -> tuple[str, str, int]:
    return str(int(log["topics"][1], 16)), "0x" + log["topics"][2][-40:], int(log["topics"][3], 16)


def decode_mint(log: dict) -> dict:
    return {"block": int(log["blockNumber"], 16), "tx": log["transactionHash"],
            "agent": str(int(log["topics"][3], 16)), "owner": "0x" + log["topics"][2][-40:]}


class BlockClock:
    """Block number -> unix time, by linear interpolation between sampled block headers."""

    def __init__(self, blocks: list[int], times: list[int]) -> None:
        order = np.argsort(blocks)
        self.blocks = np.asarray(blocks, dtype=float)[order]
        self.times = np.asarray(times, dtype=float)[order]

    @classmethod
    def sample(cls, rpc: RPC, start: int, end: int, n: int = 64) -> "BlockClock":
        bs = sorted({int(x) for x in np.linspace(start, end, n)})
        return cls(bs, [rpc.timestamp(b) for b in bs])

    def __call__(self, block) -> np.ndarray:
        return np.interp(np.asarray(block, dtype=float), self.blocks, self.times)

    def to_dict(self) -> dict:
        return {"blocks": self.blocks.astype(int).tolist(), "times": self.times.astype(int).tolist()}
