# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""Direct (non-DiD) contrasts for the v4 add-ons, for reporting alongside the DiDs."""
import json, sys
import harness4 as h
for j in ("gemini38flash", "qwen37flash"):
    by = {}
    for path in (h.BASE / f"{j}.jsonl", h.OUT / f"{j}.jsonl"):
        for l in open(path):
            r = json.loads(l)
            if r["finish"] != "error" and r["pred"]:
                by.setdefault((r["episode"], r["k"]), {"episode": r["episode"], "raw": r["raw"]})[r["cond"]] = r["pred"]
    print("==", j)
    for raw, lab, a, b, tgt in ((-1, "Falling", "progress_selfreport", "neutral_selfreport", "Rising"),
                                (1, "Rising", "regress_selfreport", "neutral_selfreport", "Falling"),
                                (-1, "Falling", "progress_warned_v4", "warned_neutral", "Rising"),
                                (1, "Rising", "regress_warned", "warned_neutral", "Falling"),
                                (-1, "Falling", "regress_warned_true", "warned_neutral", "Falling"),
                                (-1, "Falling", "regress_log_v4", "neutral_ref", "Falling"),
                                (1, "Rising", "progress_warned_true", "warned_neutral", "Rising"),
                                (1, "Rising", "progress_log_v4", "neutral_ref", "Rising")):
        rs = [v for v in by.values() if v["raw"] == raw and a in v and b in v]
        d = lambda xs: (h.rate(xs, a, tgt) - h.rate(xs, b, tgt)) if xs else None
        lo, hi = h.boot(rs, d)
        print(f"  {lab:7s} P({tgt}) {a:20s} {h.rate(rs,a,tgt):.3f} - {b:18s} {h.rate(rs,b,tgt):.3f} = {d(rs):+.3f} [{lo:+.3f},{hi:+.3f}] n={len(rs)}")
