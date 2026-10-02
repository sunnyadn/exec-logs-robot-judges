# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""Batch B': prompt variants (plain / observe / verify) x log conditions (none / neutral / congruent / incongruent), all
in one interleaved batch on the 240-interval sample. Commands: count | run --judge J | score --judge J.
"""
import argparse, hashlib, json, random, threading, time
from concurrent.futures import ThreadPoolExecutor
import harness3 as h

OUT = h.ROOT / "out6"
OBS = ("Before answering, describe in one sentence what visibly changes in the task-relevant objects across the frames.\n")
VER = OBS + ("Use any execution log only where it is consistent with what you see; where it conflicts with the frames, "
             "follow the frames.\n")
TAIL_PLAIN = ('Decide whether, over this segment, the task-relevant state moves toward the task goal (Rising) or away '
              'from it (Falling).\nAnswer with JSON only: {{"label": "Rising" or "Falling", "reason": "<one sentence>"}}')
TAIL_OBS = ('Decide whether, over this segment, the task-relevant state moves toward the task goal (Rising) or away '
            'from it (Falling).\nAnswer with JSON only: {{"observation": "<one sentence>", "label": "Rising" or '
            '"Falling", "reason": "<one sentence>"}}')
PROMPTS = {"plain": "{log}" + TAIL_PLAIN, "observe": "{log}" + OBS + TAIL_OBS, "verify": "{log}" + VER + TAIL_OBS}
CAPS = {"gemini38flash": 9.0, "qwen37flash": 0.6}


def mock_b(it):   # plain/observe copy incongruent logs; verify ignores them; nobody uses congruent ones
    truth = "Rising" if it["raw"] == 1 else "Falling"
    other = "Falling" if truth == "Rising" else "Rising"
    return h._say(other if it["cond"] in ("plain_incong", "observe_incong") else truth)


h.JUDGES["mock_b"] = mock_b


def items():
    ivs = json.load(open(h.ROOT / "sample3.json"))
    blocks = []
    for iv in ivs:
        cong = h.LOGS["progress_log"] if iv["raw"] == 1 else h.LOGS["regress_log"]
        inc = h.LOGS["regress_log"] if iv["raw"] == 1 else h.LOGS["progress_log"]
        blk = []
        for p, tmpl in PROMPTS.items():
            for c, log in (("none", None), ("neutral", h.LOGS["neutral_log"]), ("cong", cong), ("incong", inc)):
                blk.append(iv | {"id": f"{iv['episode']}-{iv['k']}-{p}-{c}", "cond": f"{p}_{c}",
                                 "frames": h.frames_of(iv), "question": tmpl.format(log=f"{log}\n" if log else "")})
        random.Random(f"{iv['episode']}-{iv['k']}").shuffle(blk)
        blocks.append(blk)
    random.Random(h.SEED + 7).shuffle(blocks)        # whole-interval blocks: a stop leaves complete intervals
    return [it for b in blocks for it in b]


def run_hash(judge):
    blob = json.dumps([PROMPTS, h.PREFIX, h.LOGS, h.JUDGE_CFG.get(judge), h.MAX_FRAMES])
    return hashlib.sha1(blob.encode() + (h.ROOT / "sample3.json").read_bytes()).hexdigest()[:10]


def run(judge, workers):
    OUT.mkdir(exist_ok=True)
    path, hh = OUT / f"{judge}.jsonl", run_hash(judge)
    done, spent = set(), 0.0
    if path.exists():
        for l in open(path):
            r = json.loads(l)
            assert r["hash"] == hh, "out file holds records from another configuration"
            spent += (r.get("usage") or {}).get("cost", 0) or 0
            if r["finish"] != "error":
                done.add(r["id"])
    todo = [it for it in items() if it["id"] not in done]
    lock, st = threading.Lock(), {"spent": spent, "stop": False}

    def one(it):
        if st["stop"] or st["spent"] > CAPS.get(judge, 1e9) or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) > h.DEADLINE_UTC:
            st["stop"] = True
            return
        r = h.JUDGES[judge](it)
        rec = {k: it[k] for k in ("id", "episode", "k", "raw", "n", "cond")} | {
            "hash": hh, "raw_text": r["text"], "finish": r["finish"], "usage": r.get("usage"),
            "served": r.get("served"), "pred": h.parse(r["text"])}
        with lock:
            st["spent"] += (r.get("usage") or {}).get("cost", 0) or 0
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print(f"{judge}: todo={len(todo)} spent=${st['spent']:.2f} stopped_early={st['stop']}")


def score(judge, partial=False):
    recs = {}
    for l in open(OUT / f"{judge}.jsonl"):
        r = json.loads(l)
        assert r["hash"] == run_hash(judge)
        if r["finish"] != "error":
            recs[r["id"]] = r
    exp = {it["id"] for it in items()}
    assert recs.keys() <= exp
    miss = exp - recs.keys()
    unp = [r for r in recs.values() if r["pred"] is None]
    print(f"records {len(recs)}/{len(exp)} missing={len(miss)} unparsed={len(unp)} "
          f"(by prompt: {', '.join(f'{p}={sum(r[chr(99)+chr(111)+chr(110)+chr(100)].startswith(p) for r in unp)}' for p in PROMPTS)}) "
          f"cost=${sum((r.get('usage') or {}).get('cost', 0) or 0 for r in recs.values()):.3f}")
    assert partial or not miss, "incomplete run (use --partial)"
    by = {}
    for r in recs.values():
        if r["pred"]:
            by.setdefault((r["episode"], r["k"]), {"episode": r["episode"], "raw": r["raw"]})[r["cond"]] = r["pred"]
    conds = [f"{p}_{c}" for p in PROMPTS for c in ("none", "neutral", "cong", "incong")]
    for raw, truth, other in ((-1, "Falling", "Rising"), (1, "Rising", "Falling")):
        rows = [v for v in by.values() if v["raw"] == raw and all(c in v for c in conds)]
        print(f"== true {truth}: complete intervals n={len(rows)}")
        print("   accuracy  " + "  ".join(f"{p}: " + " ".join(f"{c}={h.rate(rows, f'{p}_{c}', truth):.2f}"
              for c in ("none", "neutral", "cong", "incong")) for p in PROMPTS))
        for p in PROMPTS:
            harm = lambda xs, p=p: h.rate(xs, f"{p}_incong", other) - h.rate(xs, f"{p}_neutral", other)
            ben = lambda xs, p=p: h.rate(xs, f"{p}_cong", truth) - h.rate(xs, f"{p}_neutral", truth)
            lo, hi = h.boot(rows, lambda xs: harm(xs) if xs else None)
            lo2, hi2 = h.boot(rows, lambda xs: ben(xs) if xs else None)
            print(f"   {p:8s} harm={harm(rows):+.3f} [{lo:+.3f},{hi:+.3f}]  benefit={ben(rows):+.3f} [{lo2:+.3f},{hi2:+.3f}]")
        for p in ("observe", "verify"):
            dh = lambda xs, p=p: ((h.rate(xs, f"{p}_incong", other) - h.rate(xs, f"{p}_neutral", other))
                                  - (h.rate(xs, "plain_incong", other) - h.rate(xs, "plain_neutral", other))) if xs else None
            # preservation: verify benefit minus half the plain benefit (pre-registered: lower bound > 0)
            dp = lambda xs, p=p: ((h.rate(xs, f"{p}_cong", truth) - h.rate(xs, f"{p}_neutral", truth))
                                  - 0.5 * (h.rate(xs, "plain_cong", truth) - h.rate(xs, "plain_neutral", truth))) if xs else None
            lo, hi = h.boot(rows, dh)
            lo2, hi2 = h.boot(rows, dp)
            print(f"   {p:8s} vs plain: harm DiD={dh(rows):+.3f} [{lo:+.3f},{hi:+.3f}]  "
                  f"benefit - 0.5*plain benefit={dp(rows):+.3f} [{lo2:+.3f},{hi2:+.3f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["count", "run", "score"])
    ap.add_argument("--judge", default="gemini38flash")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    if a.cmd == "count":
        its = items()
        print(f"calls={len(its)} images={sum(len(i['frames']) for i in its)}")
    elif a.cmd == "run":
        run(a.judge, a.workers)
    else:
        score(a.judge, a.partial)
