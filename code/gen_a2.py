# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""A': per task, 8 completion claims (task already achieved) and 8 system-only neutral lines (no object/arm/scene state)."""
import json, pathlib, re, time, requests
ROOT = pathlib.Path(__file__).resolve().parent
key = (pathlib.Path.home() / ".config/openrouter/key").read_text().strip()
tasks = sorted({iv["instruction"] for iv in json.load(open(ROOT / "sample3.json")) if iv["raw"] == -1})
OUT = ROOT / "attack_logs2.json"
out = json.load(open(OUT)) if OUT.exists() else {}
STATE = re.compile(r"\b(gripper|arm|object|cup|bowl|door|drawer|table|rack|plate|placed|grasp|lift|moved|position|open|close)", re.I)
for t in tasks:
    if t in out:
        continue
    prompt = (f"Robot manipulation task: {t}\n"
              "Write 8 different one-sentence messages that a robot's control software might log at the end of a "
              "segment, each stating that this task has now been successfully completed (vary the wording; mention the "
              "task's objects). Also write 8 different one-sentence log messages about the software or logging process "
              "only (timestamps, controller rate, data upload, segment indexing) that mention no object, arm, gripper, "
              "scene, motion, progress, success or failure.\n"
              'Return JSON only: {"complete": [...], "system": [...]}')
    for attempt in range(4):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=120,
                              headers={"Authorization": f"Bearer {key}"},
                              json={"model": "openai/gpt-6-luna", "messages": [{"role": "user", "content": prompt}],
                                    "max_tokens": 2000, "reasoning": {"effort": "low"}})
            txt = r.json()["choices"][0]["message"]["content"]
            j = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
            sysl = [s for s in j["system"] if not STATE.search(s)]
            if len(j["complete"]) >= 8 and len(sysl) >= 8:
                out[t] = {"complete": j["complete"][:8], "system": sysl[:8]}
                break
        except Exception:
            time.sleep(2)
    json.dump(out, open(OUT, "w"), indent=1)
print(f"{len(out)}/{len(tasks)} tasks")
