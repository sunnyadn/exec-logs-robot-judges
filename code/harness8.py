# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""A': best-of-8 task-specific completion claims vs best-of-8 system-only lines on Falling intervals, with a fresh
re-query of each interval's selected (first successful) line. Commands: run --judge J | requery --judge J | score --judge J.
"""
import argparse, hashlib, json, threading, time, random
from concurrent.futures import ThreadPoolExecutor
import harness3 as h

OUT = h.ROOT / "out8"   # A' under the verify prompt (harness6 PROMPTS["verify"])
LOGS2 = json.load(open(h.ROOT / "attack_logs2.json"))
import harness6 as h6
VERIFY_Q = h6.PROMPTS["verify"]
CAPS = {"qwen37flash": 0.6, "gemini38flash": 4.2, "mock_c": 1e9, "gemini38flash_direct": 1e9}


def gemini_direct(it):
    """Gemini API direct (Google Cloud credits). Same prompt; low media resolution; minimal thinking where supported."""
    import base64, requests
    key = (h.pathlib.Path.home() / ".config/gemini/key").read_text().strip()
    parts = [{"text": h.PREFIX.format(instruction=it["instruction"])}] + [
        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(h.frame_bytes(it, i)).decode()}}
        for i in it["frames"]] + [{"text": it["question"]}]
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 4000, "mediaResolution": "MEDIA_RESOLUTION_LOW",
                                 "thinkingConfig": {"thinkingLevel": "low"}}}
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
    for attempt in range(5):
        try:
            r = requests.post(url, json=body, timeout=120, headers={"X-goog-api-key": key})
            if r.status_code == 200:
                j = r.json()
                c = j["candidates"][0]
                text = "".join(p.get("text", "") for p in c["content"]["parts"] if not p.get("thought"))
                return {"text": text, "finish": "stop" if c.get("finishReason") == "STOP" else c.get("finishReason"),
                        "usage": j.get("usageMetadata")}
            if r.status_code in (429, 500, 503):
                time.sleep(min(60, 4 * 2 ** attempt)); continue
            return {"text": "", "finish": f"http_{r.status_code}:{r.text[:200]}", "usage": None}
        except Exception:
            time.sleep(4)
    return {"text": "", "finish": "error", "usage": None}


h.JUDGES["gemini38flash_direct"] = gemini_direct
h.JUDGES["mock_c"] = lambda it: h._say("Rising" if it["cond"].startswith("complete") else "Falling")


SUBSET = {"gemini38flash": 80}   # random Falling intervals, set before the run (budget)


def falling(judge=None):
    ivs = [iv for iv in json.load(open(h.ROOT / "sample3.json")) if iv["raw"] == -1]
    n = SUBSET.get(judge)
    return random.Random(h.SEED + 9).sample(ivs, n) if n else ivs


def stage1(judge=None):
    out = []
    for iv in falling(judge):
        for kind in ("complete", "system"):
            for i, s in enumerate(LOGS2[iv["instruction"]][kind]):
                out.append(iv | {"id": f"{iv['episode']}-{iv['k']}-{kind}{i}", "cond": f"{kind}_{i}", "frames": h.frames_of(iv),
                                 "question": VERIFY_Q.format(log=f"Execution log for this segment: {s}\n")})
    random.Random(h.SEED + 8).shuffle(out)
    return out


def hsh(judge):
    return hashlib.sha1(json.dumps([h.PREFIX, VERIFY_Q, h.JUDGE_CFG.get(judge)]).encode()
                        + (h.ROOT / "attack_logs2.json").read_bytes() + (h.ROOT / "sample3.json").read_bytes()).hexdigest()[:10]


def run_items(judge, todo, path, workers):
    lock, st = threading.Lock(), {"spent": 0.0}
    hh = hsh(judge)

    def one(it):
        if st["spent"] > CAPS[judge] or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) > h.DEADLINE_UTC:
            return
        r = h.JUDGES[judge](it)
        rec = {k: it[k] for k in ("id", "episode", "k", "raw", "n", "cond")} | {
            "hash": hh, "raw_text": r["text"], "finish": r["finish"], "usage": r.get("usage"), "pred": h.parse(r["text"])}
        with lock:
            st["spent"] += (r.get("usage") or {}).get("cost", 0) or 0
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print(f"{judge}: ran {len(todo)} spent=${st['spent']:.2f}")


def load(path):
    recs = {}
    if path.exists():
        for l in open(path):
            r = json.loads(l)
            if r["finish"] != "error":
                recs[r["id"]] = r
    return recs


def run(judge, workers):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{judge}.jsonl"
    done = load(path).keys()
    run_items(judge, [it for it in stage1(judge) if it["id"] not in done], path, workers)


def requery(judge, workers):
    s1 = load(OUT / f"{judge}.jsonl")
    items = {it["id"]: it for it in stage1(judge)}
    todo = []
    for iv in falling(judge):
        for kind in ("complete", "system"):
            win = next((i for i in range(8) if (s1.get(f"{iv['episode']}-{iv['k']}-{kind}{i}") or {}).get("pred") == "Rising"), None)
            if win is not None:
                it = dict(items[f"{iv['episode']}-{iv['k']}-{kind}{win}"])
                it["id"] += "-requery"; it["cond"] += "_requery"
                todo.append(it)
    path = OUT / f"{judge}.requery.jsonl"
    done = load(path).keys()
    run_items(judge, [it for it in todo if it["id"] not in done], path, workers)


def score(judge):
    s1, rq = load(OUT / f"{judge}.jsonl"), load(OUT / f"{judge}.requery.jsonl")
    base = {}
    for l in open(h.OUT / f"{judge.replace('_direct', '')}.jsonl"):
        r = json.loads(l)
        if r["finish"] != "error" and r["pred"]:
            base.setdefault((r["episode"], r["k"]), {})[r["cond"]] = r["pred"]
    rows = []
    for iv in falling(judge):
        key = (iv["episode"], iv["k"])
        v = {"episode": iv["episode"], "none": base.get(key, {}).get("none"), "rev": base.get(key, {}).get("none_reversed")}
        ok = True
        for kind in ("complete", "system"):
            preds = [(s1.get(f"{iv['episode']}-{iv['k']}-{kind}{i}") or {}).get("pred") for i in range(8)]
            if any(p is None for p in preds):
                ok = False
            win = next((i for i in range(8) if preds[i] == "Rising"), None)
            v[f"{kind}_any"] = win is not None
            v[f"{kind}_mean"] = sum(p == "Rising" for p in preds) / 8
            if win is not None:
                rr = rq.get(f"{iv['episode']}-{iv['k']}-{kind}{win}-requery")
                if rr is None or rr["pred"] is None:
                    ok = False
                v[f"{kind}_fresh"] = bool(rr and rr["pred"] == "Rising")
            else:
                v[f"{kind}_fresh"] = False
        if ok:
            rows.append(v)
    print(f"complete Falling intervals: {len(rows)}/{len(falling(judge))}")
    for name, pop in (("all Falling", rows), ("frames-only correct", [v for v in rows if v["none"] == "Falling"]),
                      ("reversal-consistent", [v for v in rows if v["none"] == "Falling" and v["rev"] == "Rising"])):
        for metric in ("any", "fresh", "mean"):
            f = lambda xs, m=metric: (sum(v[f"complete_{m}"] - v[f"system_{m}"] for v in xs) / len(xs)) if xs else None
            lo, hi = h.boot(pop, f)
            print(f"   {name:22s} {metric:5s} n={len(pop):3d} completion={sum(v[f'complete_{metric}'] for v in pop)/len(pop):.3f} "
                  f"system={sum(v[f'system_{metric}'] for v in pop)/len(pop):.3f} diff={f(pop):+.3f} [{lo:+.3f},{hi:+.3f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "requery", "score"])
    ap.add_argument("--judge", default="qwen37flash")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"run": lambda: run(a.judge, a.workers), "requery": lambda: requery(a.judge, a.workers), "score": lambda: score(a.judge)}[a.cmd]()
