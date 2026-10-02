# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""Cross-judge comparison on the intervals that are reversal-consistent for ALL three judges."""
import json
import harness3 as h
J = ["gemini38flash", "qwen37flash", "gpt61sol"]
by = {}
for j in J:
    for l in open(h.OUT / f"{j}.jsonl"):
        r = json.loads(l)
        if r["finish"] != "error" and r["pred"]:
            by.setdefault(j, {}).setdefault((r["episode"], r["k"]), {"episode": r["episode"], "raw": r["raw"]})[r["cond"]] = r["pred"]
for raw, truth, claim, log in ((-1, "Falling", "Rising", "progress_log"), (1, "Rising", "Falling", "regress_log")):
    keys = None
    for j in J:
        ks = {k for k, v in by[j].items() if v["raw"] == raw and v.get("none") == truth and v.get("none_reversed") == claim
              and log in v and "neutral_log" in v}
        keys = ks if keys is None else keys & ks
    print(f"== true {truth}: common reversal-consistent n={len(keys)}")
    for j in J:
        rows = [by[j][k] for k in keys]
        d = lambda xs: (h.rate(xs, log, claim) - h.rate(xs, "neutral_log", claim)) if xs else None
        lo, hi = h.boot(rows, d)
        print(f"   {j:14s} {log}->{claim} {h.rate(rows, log, claim):.3f} neutral {h.rate(rows, 'neutral_log', claim):.3f} diff {d(rows):+.3f} [{lo:+.3f},{hi:+.3f}]")
