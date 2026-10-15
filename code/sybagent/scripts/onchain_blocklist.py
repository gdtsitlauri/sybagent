"""Study 21: the committee blocklist on-chain (contracts/Blocklist.sol) on a local EVM.

Measures gas of report / committee draw / vote / close for committee sizes and registry sizes, and checks the rules:
judge once per epoch, muting after cleared reports, quorum, seat-only and single voting, and that the drawn
committee never contains the accused and has distinct members.

    python scripts/onchain_blocklist.py --out results/21_onchain_blocklist
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onchain_sampler import Chain  # noqa: E402


def ok(receipt) -> bool:
    return receipt["status"] == 1


def try_send(ch, fn, sender=None) -> bool:
    try:
        return ok(ch.send(fn, sender))
    except Exception:
        return False


def deploy(ch, c: int, epoch: int = 1000, voting: int = 100, strikes: int = 2):
    reg = ch.deploy("MockOwnedRegistry")
    vrf = ch.deploy("MockVRFCoordinator")
    bl = ch.deploy("Blocklist", reg.address, vrf.address, c, 2, 3, voting, epoch, strikes)
    return reg, vrf, bl


def open_case(ch, reg, vrf, bl, reporter: int, accused: int, word: int):
    r1 = ch.send(bl.functions.report(reporter, accused))
    rid = vrf.functions.nextId().call() - 1
    r2 = ch.send(vrf.functions.fulfill(rid, word))
    return r1, r2, rid


def gas_study(ch) -> list[dict]:
    rows = []
    rng = np.random.default_rng(0)
    for c in (15, 45):
        for n in (1_000, 1_000_000):
            reg, vrf, bl = deploy(ch, c)
            ch.send(reg.functions.setTotal(n, ch.acct))
            r1, r2, rid = open_case(ch, reg, vrf, bl, 1, 2, int(rng.integers(0, 2**63)))
            seats = bl.functions.seatsOf(rid).call()
            assert len(seats) == c and len(set(seats)) == c and 2 not in seats
            votes = [ch.send(bl.functions.vote(rid, s, True))["gasUsed"] for s in seats]
            close = ch.send(bl.functions.close(rid))
            assert bl.functions.blocked(2).call()
            rows.append({"committee_size": c, "registry_size": n, "report_gas": r1["gasUsed"],
                         "draw_gas": r2["gasUsed"], "vote_gas": float(np.mean(votes)), "close_gas": close["gasUsed"],
                         "total_gas": r1["gasUsed"] + r2["gasUsed"] + sum(votes) + close["gasUsed"]})
    return rows


def rule_checks(ch) -> dict:
    out = {}
    reg, vrf, bl = deploy(ch, 15, epoch=1000, voting=5, strikes=2)
    ch.send(reg.functions.setTotal(500, ch.acct))
    _, _, rid = open_case(ch, reg, vrf, bl, 1, 7, 11)
    out["re_report_same_epoch_rejected"] = not try_send(ch, bl.functions.report(3, 7))
    seats = bl.functions.seatsOf(rid).call()
    non_seat = next(i for i in range(1, 500) if i not in seats and i != 7)
    out["non_seat_vote_rejected"] = not try_send(ch, bl.functions.vote(rid, non_seat, True))
    ch.send(bl.functions.vote(rid, seats[0], True))
    out["double_vote_rejected"] = not try_send(ch, bl.functions.vote(rid, seats[0], True))
    out["close_while_open_without_quorum_rejected"] = not try_send(ch, bl.functions.close(rid))
    for s in seats[1:9]:                                       # 9 block votes of 15 < 2/3, no keep quorum either
        ch.send(bl.functions.vote(rid, s, True))
    for _ in range(6):                                         # let the voting window pass
        ch.send(reg.functions.setOwner(999, ch.acct))
    out["close_after_window_ok"] = try_send(ch, bl.functions.close(rid))
    out["below_quorum_not_blocked"] = not bl.functions.blocked(7).call()
    out["inconclusive_no_strike"] = bl.functions.strikes(1).call() == 0
    # a quorum of keep votes clears and strikes the reporter; two clears mute it
    regk, vrfk, blk = deploy(ch, 15, epoch=1000, voting=100, strikes=2)
    ch.send(regk.functions.setTotal(500, ch.acct))
    for accused, word in ((8, 21), (9, 22)):
        _, _, rk = open_case(ch, regk, vrfk, blk, 1, accused, word)
        for s in blk.functions.seatsOf(rk).call()[:10]:
            ch.send(blk.functions.vote(rk, s, False))
        ch.send(blk.functions.close(rk))
        if accused == 8:
            out["reporter_strikes_after_clear"] = blk.functions.strikes(1).call()
    out["muted_after_two_clears"] = not try_send(ch, blk.functions.report(1, 11))
    # votes after the window are rejected (each transaction mines one block; the window is 5 blocks)
    _, _, rid3 = open_case(ch, reg, vrf, bl, 2, 10, 13)
    accepted = sum(try_send(ch, bl.functions.vote(rid3, s, True)) for s in bl.functions.seatsOf(rid3).call()[:10])
    out["votes_accepted_within_5_block_window"] = int(accepted)
    # exactly a 2/3 quorum blocks (10 of 15), with a voting window long enough
    regq, vrfq, blq = deploy(ch, 15, epoch=1000, voting=100, strikes=2)
    ch.send(regq.functions.setTotal(500, ch.acct))
    _, _, rq = open_case(ch, regq, vrfq, blq, 2, 10, 14)
    for s in blq.functions.seatsOf(rq).call()[:10]:
        ch.send(blq.functions.vote(rq, s, True))
    out["quorum_close_ok"] = try_send(ch, blq.functions.close(rq))
    out["quorum_blocks"] = blq.functions.blocked(10).call()
    # seats never include the accused and are distinct, over many draws
    bad = 0
    reg2, vrf2, bl2 = deploy(ch, 15, epoch=0)
    ch.send(reg2.functions.setTotal(40, ch.acct))
    rng = np.random.default_rng(1)
    for i in range(200):
        _, _, r = open_case(ch, reg2, vrf2, bl2, 1, 5, int.from_bytes(rng.bytes(32), "big"))
        s = bl2.functions.seatsOf(r).call()
        bad += int(5 in s or len(set(s)) != 15)
    out["draws_checked"] = 200
    out["draws_with_accused_or_duplicates"] = bad
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ch = Chain()
    gas = pd.DataFrame(gas_study(ch))
    gas.to_csv(out / "gas.csv", index=False)
    print(gas.to_string())
    checks = rule_checks(ch)
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
