"""Study 10: semantic-blind anchor sampling on-chain (contracts/AnchorSampler.sol).

Runs the contracts on a local EVM (py-evm through eth-tester) and measures
  1. gas of a request and of the VRF callback, for registry sizes N and anchor counts a;
  2. that on-chain draws equal the Python reference sampler bit for bit, and are uniform (chi-square);
  3. that identities registered after a request are never drawn (no front-running);
  4. that a second request while one is pending is rejected (no re-roll);
and computes the grinding advantage against a blockhash-seeded sampler (Proposition 6), with the contract's own
draw function.

    python scripts/onchain_sampler.py --out results/10_onchain_sampler
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Crypto.Hash import keccak
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"
SOLC = "0.8.24"


# ---------------------------------------------------------------- reference sampler (same as AnchorSampler._draw)
def _kec(b: bytes) -> bytes:
    k = keccak.new(digest_bits=256)
    k.update(b)
    return k.digest()


def draw(seed: int, n: int, a: int, skip: int) -> list[int]:
    out: list[int] = []
    ctr = 0
    while len(out) < a:
        h = int.from_bytes(_kec(seed.to_bytes(32, "big") + ctr.to_bytes(32, "big")), "big")
        idx = 1 + h % n
        ctr += 1
        if idx == skip or idx in out:
            continue
        out.append(idx)
    return out


# ---------------------------------------------------------------- chain helpers
def compile_all() -> dict:
    import solcx
    solcx.set_solc_version(SOLC)
    srcs = {p.name: {"content": p.read_text()} for p in CONTRACTS.glob("*.sol")}
    res = solcx.compile_standard({"language": "Solidity", "sources": srcs,
                                  "settings": {"optimizer": {"enabled": True, "runs": 200},
                                               "outputSelection": {"*": {"*": ["abi", "evm.bytecode.object"]}}}},
                                 allow_paths=str(CONTRACTS))
    out = {}
    for file, cs in res["contracts"].items():
        for name, c in cs.items():
            out[name] = (c["abi"], c["evm"]["bytecode"]["object"])
    return out


class Chain:
    def __init__(self) -> None:
        from web3 import EthereumTesterProvider, Web3
        self.w3 = Web3(EthereumTesterProvider())
        self.acct = self.w3.eth.accounts[0]
        self.art = compile_all()

    def deploy(self, name: str, *args):
        abi, code = self.art[name]
        tx = self.w3.eth.contract(abi=abi, bytecode=code).constructor(*args).transact({"from": self.acct})
        addr = self.w3.eth.get_transaction_receipt(tx)["contractAddress"]
        return self.w3.eth.contract(address=addr, abi=abi)

    def send(self, fn, sender=None):
        tx = fn.transact({"from": sender or self.acct, "gas": 10_000_000})
        return self.w3.eth.get_transaction_receipt(tx)


def setup(ch: Chain):
    reg = ch.deploy("MockAgentRegistry")
    vrf = ch.deploy("MockVRFCoordinator")
    smp = ch.deploy("AnchorSampler", reg.address, vrf.address)
    bh = ch.deploy("BlockhashAnchorSampler", reg.address)
    noop = ch.deploy("NoopConsumer")
    ch.send(reg.functions.register())            # agent id 1, owned by the test account
    return reg, vrf, smp, bh, noop


def request_and_fulfill(ch, vrf, smp, a: int, word: int) -> tuple[int, int, list[int], int]:
    r1 = ch.send(smp.functions.requestAnchors(1, a))
    rid = vrf.functions.nextId().call() - 1
    r2 = ch.send(vrf.functions.fulfill(rid, word))
    assert r1["status"] == 1 and r2["status"] == 1
    return r1["gasUsed"], r2["gasUsed"], smp.functions.anchorsOf(rid).call(), rid


# ---------------------------------------------------------------- experiments
def gas_study(ch, reg, vrf, smp, bh, noop) -> list[dict]:
    # gas of the same fulfillment path into a consumer that does nothing = base transaction + mock overhead
    overhead = ch.send(vrf.functions.fulfillTo(noop.address, 1, 12345))["gasUsed"]
    rows = []
    rng = np.random.default_rng(0)
    for n in (100, 10_000, 1_000_000, 1_000_000_000):
        ch.send(reg.functions.setTotal(n))
        for a in (1, 2, 3, 4, 6, 8, 16):
            if a >= n:
                continue
            reqs, fuls, bhs = [], [], []
            for _ in range(5):
                g1, g2, anchors, _ = request_and_fulfill(ch, vrf, smp, a, int(rng.integers(0, 2**63)) << 64)
                assert len(set(anchors)) == a and 1 not in anchors and max(anchors) <= n
                reqs.append(g1)
                fuls.append(g2)
                bhs.append(ch.send(bh.functions.sampleAnchors(1, a))["gasUsed"])
            rows.append({"registry_size": n, "anchors": a, "request_gas": float(np.mean(reqs)),
                         "fulfill_gas": float(np.mean(fuls)), "fulfill_gas_max": int(max(fuls)),
                         "callback_gas": float(np.mean(fuls)) - overhead, "mock_overhead_gas": overhead,
                         "blockhash_gas": float(np.mean(bhs))})
    return rows


def uniformity_study(ch, reg, vrf, smp, n_req: int = 2000) -> dict:
    n, a = 100, 3
    ch.send(reg.functions.setTotal(n))
    counts = np.zeros(n + 1, dtype=int)
    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(n_req):
        word = int.from_bytes(rng.bytes(32), "big")
        _, _, anchors, _ = request_and_fulfill(ch, vrf, smp, a, word)
        mismatches += anchors != draw(word, n, a, 1)
        for x in anchors:
            counts[x] += 1
    obs = counts[2:]                                   # ids 2..n (id 1 is the requester)
    chi2, p = stats.chisquare(obs)
    return {"registry_size": n, "anchors": a, "requests": n_req, "reference_mismatches": int(mismatches),
            "chi2": float(chi2), "dof": int(len(obs) - 1), "p_value": float(p),
            "min_count": int(obs.min()), "max_count": int(obs.max()), "expected": float(obs.mean())}


def frontrun_and_reroll(ch, reg, vrf, smp) -> dict:
    n = 50
    ch.send(reg.functions.setTotal(n))
    late_hits, trials = 0, 200
    rng = np.random.default_rng(2)
    rejected = False
    for t in range(trials):
        ch.send(reg.functions.setTotal(n))
        ch.send(smp.functions.requestAnchors(1, 3))
        rid = vrf.functions.nextId().call() - 1
        if t == 0:
            try:                                          # a reverted transaction has receipt status 0
                rejected = ch.send(smp.functions.requestAnchors(1, 3))["status"] == 0
            except Exception:
                rejected = True
            rejected = rejected and vrf.functions.nextId().call() - 1 == rid
        ch.send(reg.functions.setTotal(n + 10_000))     # 10,000 fresh identities register after the request
        ch.send(vrf.functions.fulfill(rid, int.from_bytes(rng.bytes(32), "big")))
        late_hits += any(x > n for x in smp.functions.anchorsOf(rid).call())
    return {"trials": trials, "late_identities_per_trial": 10_000, "anchors_from_late_identities": late_hits,
            "second_request_while_pending_rejected": rejected}


def grinding_study(n_honest=200, sybil_counts=(50, 200), anchors=(1, 2, 3, 4), rerolls=(0, 1, 10, 100, 1000),
                   trials=4000) -> list[dict]:
    """P(no honest anchor) when whoever fixes the seed can try r+1 seeds (Proposition 6). Registry ids: requester 1,
    honest 2..n_honest, Sybils n_honest+1..n_honest+B."""
    rows = []
    rng = np.random.default_rng(3)
    for b in sybil_counts:
        n = n_honest + b
        for a in anchors:
            q_mc = None
            for r in rerolls:
                hits = 0
                for _ in range(trials if r <= 100 else trials // 8):
                    for _ in range(r + 1):
                        seed = int.from_bytes(rng.bytes(32), "big")
                        if all(x > n_honest for x in draw(seed, n, a, 1)):
                            hits += 1
                            break
                t = trials if r <= 100 else trials // 8
                p_hat = hits / t
                if r == 0:
                    q_mc = p_hat
                # exact q for sampling without replacement among the N-1 other ids
                from math import comb
                q = comb(b, a) / comb(n - 1, a)
                rows.append({"honest": n_honest, "sybils": b, "anchors": a, "rerolls": r, "trials": t,
                             "p_no_honest_anchor": p_hat, "q_exact": q,
                             "prop6_prediction": 1 - (1 - q) ** (r + 1),
                             "theorem1_bound": (b / (n_honest + b - a)) ** a})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ch = Chain()
    reg, vrf, smp, bh, noop = setup(ch)
    gas = pd.DataFrame(gas_study(ch, reg, vrf, smp, bh, noop))
    gas.to_csv(out / "gas.csv", index=False)
    print(gas.to_string(), flush=True)
    uni = uniformity_study(ch, reg, vrf, smp)
    print(uni, flush=True)
    fr = frontrun_and_reroll(ch, reg, vrf, smp)
    print(fr, flush=True)
    grind = pd.DataFrame(grinding_study())
    grind.to_csv(out / "grinding.csv", index=False)
    print(grind.to_string(), flush=True)
    (out / "checks.json").write_text(json.dumps({"solc": SOLC, "optimizer_runs": 200, "uniformity": uni,
                                                 "frontrun_reroll": fr}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
