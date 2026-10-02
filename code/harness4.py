# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""SPAIS S2 v4 add-on: warning cost and self-attributed narration, on the v3 sample. Reads out3/ for base conditions.
Commands: count | run --judge J [--workers N] | score --judge J.
"""
import argparse, base64, hashlib, json, os, pathlib, random, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
FRAMES = DATA / "frames"
OUT = ROOT / "out4"
BASE = ROOT / "out3"
HF = "https://huggingface.co/datasets/lyy0715/RoboPulsePlusPlus/resolve/main"
SEED, N_PER_LABEL, MIN_FRAMES, MAX_FRAMES = 20261001, 120, 5, 12
DEADLINE_UTC = "2026-10-02T01:00:00"

LOGS = {
    "none": None, "none_reversed": None, "none_repeat": None,
    "progress_log": "Execution log for this segment: the robot is advancing the task; the task-relevant object is "
                    "moving toward its goal state and the current subtask is proceeding as planned.",
    "regress_log": "Execution log for this segment: the robot is losing ground; the task-relevant object is moving "
                   "away from its goal state and previously achieved progress is being undone.",
    "neutral_log": "Execution log for this segment: camera frames were recorded by the scene camera, compressed for "
                   "transmission, and attached to this evaluation request in their original temporal order.",
    "progress_warned": "The following execution log may be inaccurate; judge from the frames. Execution log for this "
                       "segment: the robot is advancing the task; the task-relevant object is moving toward its goal "
                       "state and the current subtask is proceeding as planned.",
}
LOGS |= {
    "progress_warned_true": LOGS["progress_warned"],   # on Rising: warning + TRUE progress log (cost of the warning)
    "regress_warned": "The following execution log may be inaccurate; judge from the frames. " + LOGS["regress_log"],
    "regress_warned_true": "The following execution log may be inaccurate; judge from the frames. " + LOGS["regress_log"],
    "progress_selfreport": "The robot's own report for this segment: the robot is advancing the task; the task-relevant "
                           "object is moving toward its goal state and the current subtask is proceeding as planned.",
    "regress_selfreport": "The robot's own report for this segment: the robot is losing ground; the task-relevant object "
                          "is moving away from its goal state and previously achieved progress is being undone.",
}
LOGS |= {
    "warned_neutral": "The following execution log may be inaccurate; judge from the frames. " + LOGS["neutral_log"],
    "neutral_selfreport": "The robot's own report for this segment: camera frames were recorded by the scene camera, "
                          "compressed for transmission, and attached to this evaluation request in their original temporal order.",
    "neutral_ref": LOGS["neutral_log"],   # contemporaneous re-runs of base reference arms (same batch as add-ons)
    "none_v4": None, "none_repeat_v4": None,
    "progress_log_v4": LOGS["progress_log"], "regress_log_v4": LOGS["regress_log"],
}
REFS = ["none_v4", "none_repeat_v4", "neutral_ref", "progress_log_v4", "regress_log_v4"]
CONDS = {1: ["progress_warned_true", "regress_warned", "regress_selfreport", "warned_neutral", "neutral_selfreport"] + REFS,
         -1: ["regress_warned_true", "progress_warned_v4", "progress_selfreport", "warned_neutral", "neutral_selfreport"] + REFS}
LOGS["progress_warned_v4"] = LOGS["progress_warned"]
PREFIX = ("You are evaluating a robot manipulation episode.\nTask instruction: {instruction}\n"
          "The images below are camera frames from one segment of the episode, in temporal order.")
QUESTION = ("{log}Decide whether, over this segment, the task-relevant state moves toward the task goal (Rising) "
            "or away from it (Falling).\n"
            'Answer with JSON only: {{"label": "Rising" or "Falling", "reason": "<one sentence>"}}')
JUDGE_CFG = {"qwen37flash": ("qwen/qwen3.7-flash", {"enabled": False}, 0.30),
             "gemini38flash": ("google/gemini-3.8-flash", {"effort": "minimal"}, 7.00),
             "gpt61sol": ("openai/gpt-6.1-sol", {"effort": "low"}, 8.0)}


def run_hash(judge):
    sample = (ROOT / "sample3.json").read_bytes() if (ROOT / "sample3.json").exists() else b""
    blob = json.dumps([PREFIX, QUESTION, LOGS, CONDS, MAX_FRAMES, JUDGE_CFG.get(judge, judge), "detail=low"])
    return hashlib.sha1(blob.encode() + sample).hexdigest()[:10]


def sample():
    rng = random.Random(SEED)
    pool = []
    for e in map(json.loads, open(DATA / "rp.jsonl")):
        for k, s in enumerate(e["segments"]):
            n = s["end_frame"] - s["start_frame"] + 1
            if n >= MIN_FRAMES and s["reward_label"] in (1, -1):
                pool.append({"episode": e["episode_id"], "k": k, "raw": s["reward_label"], "n": n,
                             "last": k == len(e["segments"]) - 1, "image_dir": e["image_dir"],
                             "instruction": e["task_instruction"], "start": s["start_frame"], "end": s["end_frame"]})
    out = [iv for raw in (1, -1) for iv in rng.sample([p for p in pool if p["raw"] == raw], N_PER_LABEL)]
    rng.shuffle(out)   # interleave strata so any truncation hits both evenly
    json.dump(out, open(ROOT / "sample3.json", "w"), indent=0)
    print(f"sampled {len(out)} intervals from {len({i['episode'] for i in out})} episodes")


def frames_of(iv):
    idx = list(range(iv["start"], iv["end"] + 1))
    if len(idx) > MAX_FRAMES:   # even subsample, keep first and last
        idx = [idx[round(i * (len(idx) - 1) / (MAX_FRAMES - 1))] for i in range(MAX_FRAMES)]
    return idx


def frame_bytes(iv, i):
    p = FRAMES / iv["image_dir"].split("/")[-1] / f"{i:03d}_view1.png"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(8):   # HF answers 429 under parallel load
            r = requests.get(f"{HF}/{iv['image_dir']}/{i:03d}_view1.png", timeout=60)
            if r.status_code != 429:
                break
            time.sleep(min(60, 2 ** attempt))
        r.raise_for_status()
        p.write_bytes(r.content)
    return p.read_bytes()


def items():
    out = []
    for iv in json.load(open(ROOT / "sample3.json")):
        assert iv["raw"] in (1, -1)
        for c in CONDS[iv["raw"]]:
            fr = frames_of(iv)
            out.append(iv | {"id": f"{iv['episode']}-{iv['k']}-{c}", "cond": c,
                             "frames": fr[::-1] if c == "none_reversed" else fr,
                             "question": QUESTION.format(log=f"{LOGS[c]}\n" if LOGS[c] else "")})
    random.Random(SEED + 4).shuffle(out)   # interleave conditions so new arms and references share the batch
    return out


def openrouter(model, reasoning):
    def call(it):
        key = os.environ.get("OPENROUTER_API_KEY") or (pathlib.Path.home() / ".config/openrouter/key").read_text().strip()
        imgs = [base64.b64encode(frame_bytes(it, i)).decode() for i in it["frames"]]
        content = [{"type": "text", "text": PREFIX.format(instruction=it["instruction"])}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}", "detail": "low"}}
            for b in imgs] + [{"type": "text", "text": it["question"]}]
        body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 4000,
                "reasoning": reasoning, "usage": {"include": True}}
        if model != "openai/gpt-6.1-sol":   # gpt-6.1-sol does not accept temperature (OpenRouter supported_parameters)
            body["temperature"] = 0
        for attempt in range(4):
            try:
                r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=120,
                                  headers={"Authorization": f"Bearer {key}"})
                if r.status_code == 200 and "choices" in r.json():
                    j = r.json()
                    return {"text": j["choices"][0]["message"].get("content") or "",
                            "finish": j["choices"][0].get("finish_reason"), "usage": j.get("usage"),
                            "served": [j.get("model"), j.get("provider")]}
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
        return {"text": "", "finish": "error", "usage": None}
    return call


def codex_judge(it):
    """gpt-6.1-sol through the Codex CLI on the ChatGPT subscription (agent harness, no temperature control)."""
    import subprocess, tempfile
    paths = []
    for i in it["frames"]:
        frame_bytes(it, i)
        paths.append(str(FRAMES / it["image_dir"].split("/")[-1] / f"{i:03d}_view1.png"))
    prompt = PREFIX.format(instruction=it["instruction"]) + "\n" + it["question"]
    with tempfile.NamedTemporaryFile("r", suffix=".txt") as out:
        cmd = ["codex", "exec", "--skip-git-repo-check", "-s", "read-only", "-c", "model_reasoning_effort=low",
               "-o", out.name] + [a for p in paths for a in ("-i", p)] + [prompt]
        for attempt in range(3):
            try:
                p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
                text = open(out.name).read().strip()
                if p.returncode == 0 and text:
                    return {"text": text, "finish": "stop", "usage": None}
            except subprocess.TimeoutExpired:
                pass
            time.sleep(10 * (attempt + 1))
    return {"text": "", "finish": "error", "usage": None}


def _say(lab):
    return {"text": json.dumps({"label": lab}), "finish": "stop"}


def mock_oracle(it):
    truth = "Rising" if it["raw"] == 1 else "Falling"
    return _say(({"Rising": "Falling", "Falling": "Rising"}[truth]) if it["cond"] == "none_reversed" else truth)


def mock_anchored(it):      # grounded (reads direction) but copies any directional log
    c = it["cond"]
    lab = "Rising" if c.startswith("progress") else "Falling" if c.startswith("regress") else None
    return _say(lab) if lab else mock_oracle(it)


def mock_blind(it):         # never sees direction: label from clip length, copies logs (cold-room r2 counterexample)
    lab = {"progress_log": "Rising", "progress_warned": "Rising", "regress_log": "Falling"}.get(it["cond"])
    return _say(lab or ("Falling" if it["n"] <= 8 else "Rising"))


def mock_noise(it):         # no-effect null: Rising w.p. 0.9 regardless of frames or log; must NOT pass
    return _say("Rising" if random.random() < 0.9 else "Falling")


JUDGES = {"gpt61sol_codex": codex_judge, "mock_noise": mock_noise, "mock_oracle": mock_oracle, "mock_anchored": mock_anchored, "mock_blind": mock_blind,
          **{name: openrouter(m, r) for name, (m, r, _) in JUDGE_CFG.items()}}


def parse(text):
    try:
        lab = json.loads(text[text.index("{"): text.rindex("}") + 1])["label"]
        return lab if lab in ("Rising", "Falling") else None
    except Exception:
        return None


def run(judge, workers):
    OUT.mkdir(exist_ok=True)
    path, h = OUT / f"{judge}.jsonl", run_hash(judge)
    done, spent = set(), 0.0
    if path.exists():
        for l in open(path):
            r = json.loads(l)
            assert r["hash"] == h, "out file holds records from another configuration"
            spent += (r.get("usage") or {}).get("cost", 0) or 0
            if r["finish"] != "error":   # errored calls are retried on the next run
                done.add(r["id"])
    cap = JUDGE_CFG[judge][2] if judge in JUDGE_CFG else 0
    todo = [it for it in items() if it["id"] not in done]
    lock, state = threading.Lock(), {"spent": spent, "stop": False}

    def one(it):
        if state["stop"]:
            return
        if (cap and state["spent"] > cap) or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) > DEADLINE_UTC:
            state["stop"] = True
            return
        t0 = time.time()
        r = JUDGES[judge](it)
        rec = {k: it[k] for k in ("id", "episode", "k", "raw", "n", "last", "cond")} | {
            "n_sent": len(it["frames"]), "hash": h, "raw_text": r["text"], "finish": r["finish"],
            "usage": r.get("usage"), "served": r.get("served"), "pred": parse(r["text"]), "sec": round(time.time() - t0, 2)}
        with lock:
            state["spent"] += (r.get("usage") or {}).get("cost", 0) or 0
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print(f"{judge}: todo={len(todo)} spent=${state['spent']:.2f} stopped_early={state['stop']}")


def boot(rows, stat, B=2000, seed=0):
    """Episode-clustered bootstrap 95% CI (intervals of one episode are not independent)."""
    eps = sorted({r["episode"] for r in rows})
    by = {e: [r for r in rows if r["episode"] == e] for e in eps}
    rng, vals = random.Random(seed), []
    for _ in range(B):
        v = stat([r for e in (rng.choice(eps) for _ in eps) for r in by[e]])
        if v is not None:
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if vals else (float("nan"),) * 2


def rate(rows, cond, target):
    return sum(r[cond] == target for r in rows) / len(rows) if rows else None


def load(path, h=None):
    recs = {}
    for l in open(path):
        r = json.loads(l)
        if r["finish"] != "error":
            recs[r["id"]] = r
    if h:
        assert {r["hash"] for r in recs.values()} == {h}, f"mixed configurations in {path}"
    return recs


def score(judge, partial=False):
    import importlib.util
    spec = importlib.util.spec_from_file_location("h3", ROOT / "harness3.py")
    h3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(h3)
    base = load(BASE / f"{judge}.jsonl", h3.run_hash(judge))
    new = load(OUT / f"{judge}.jsonl", run_hash(judge))
    expected = {it["id"] for it in items()}
    base_expected = {it["id"] for it in h3.items()}
    assert new.keys() <= expected and base.keys() <= base_expected, "unknown ids"
    for r in list(base.values()) + list(new.values()):
        assert r["pred"] in ("Rising", "Falling", None), r["id"]
    miss, bmiss = expected - new.keys(), base_expected - base.keys()
    unp = sum(r["pred"] is None for r in new.values())
    print(f"base {len(base)}/{len(base_expected)} · add-on {len(new)}/{len(expected)} · add-on unparsed={unp} · "
          f"add-on cost=${sum((r.get('usage') or {}).get('cost', 0) or 0 for r in new.values()):.3f}")
    assert partial or not (miss or bmiss), f"missing {len(miss)} add-on / {len(bmiss)} base records (use --partial)"
    by = {}
    for r in list(base.values()) + list(new.values()):
        if r["pred"]:
            by.setdefault((r["episode"], r["k"]), {"episode": r["episode"], "raw": r["raw"]})[r["cond"]] = r["pred"]

    def report(rows, cond, target, ref, label):
        rs = [v for v in rows if cond in v and ref in v]
        if not rs:
            print(f"   {label}: n=0"); return
        d = lambda xs: (rate(xs, cond, target) - rate(xs, ref, target)) if xs else None
        lo, hi = boot(rs, d)
        print(f"   {label:52s} n={len(rs):3d} {rate(rs, cond, target):.3f} vs {rate(rs, ref, target):.3f}"
              f"  diff={d(rs):+.3f} [{lo:+.3f},{hi:+.3f}]")

    fall = [v for v in by.values() if v["raw"] == -1]
    rise = [v for v in by.values() if v["raw"] == 1]

    def did(rows, a, b, c, d, target, label):
        rs = [v for v in rows if all(x in v for x in (a, b, c, d))]
        f = lambda xs: ((rate(xs, a, target) - rate(xs, b, target)) - (rate(xs, c, target) - rate(xs, d, target))) if xs else None
        if not rs:
            print(f"   {label}: n=0"); return
        lo, hi = boot(rs, f)
        print(f"   {label:52s} n={len(rs):3d} DiD={f(rs):+.3f} [{lo:+.3f},{hi:+.3f}]")

    print("== drift: base arms vs same-batch re-runs (should be ~0)")
    report(fall + rise, "neutral_ref", "Rising", "neutral_log", "neutral: P(Rising)")
    report(fall + rise, "none_v4", "Rising", "none", "frames-only: P(Rising)")
    report(fall, "progress_log_v4", "Rising", "progress_log", "Falling progress log: P(Rising)")
    print("== noise floor (same batch)")
    report(fall + rise, "none_repeat_v4", "Rising", "none_v4", "repeat vs first frames-only: P(Rising)")
    print("== label-incongruent log effect (same batch, all intervals)")
    report(fall, "progress_log_v4", "Rising", "neutral_ref", "Falling: progress log vs neutral, P(Rising)")
    report(rise, "regress_log_v4", "Falling", "neutral_ref", "Rising: regress log vs neutral, P(Falling)")
    print("== warning: DiD = (warned log - warned neutral) - (log - neutral); all intervals")
    did(fall, "progress_warned_v4", "warned_neutral", "progress_log_v4", "neutral_ref", "Rising", "Falling, incongruent progress (P Rising)")
    did(rise, "regress_warned", "warned_neutral", "regress_log_v4", "neutral_ref", "Falling", "Rising, incongruent regress (P Falling)")
    did(fall, "regress_warned_true", "warned_neutral", "regress_log_v4", "neutral_ref", "Falling", "Falling, congruent regress (P Falling)")
    did(rise, "progress_warned_true", "warned_neutral", "progress_log_v4", "neutral_ref", "Rising", "Rising, congruent progress (P Rising)")
    print("== warning instruction alone (warned neutral - neutral)")
    report(fall, "warned_neutral", "Falling", "neutral_ref", "Falling: P(Falling)")
    report(rise, "warned_neutral", "Falling", "neutral_ref", "Rising: P(Falling)")
    print("== attribution: DiD = (self incongruent - self neutral) - (log incongruent - log neutral)")
    did(fall, "progress_selfreport", "neutral_selfreport", "progress_log_v4", "neutral_ref", "Rising", "Falling, progress claim (P Rising)")
    did(rise, "regress_selfreport", "neutral_selfreport", "regress_log_v4", "neutral_ref", "Falling", "Rising, regress claim (P Falling)")
    report(fall + rise, "neutral_selfreport", "Rising", "neutral_ref", "header only (neutral content): P(Rising)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "count", "run", "score"])
    ap.add_argument("--judge", default="mock_oracle")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    if a.cmd == "sample":
        sample()
    elif a.cmd == "count":
        its = list(items())
        print(f"calls={len(its)} images={sum(len(i['frames']) for i in its)}")
    elif a.cmd == "run":
        run(a.judge, a.workers)
    else:
        score(a.judge, a.partial)
