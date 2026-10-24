# Execution Logs Sway VLM Judges of Robot Progress

Code, inputs and outputs for the paper by Weiyan Zhao and Sunny Yang (equal contribution), SPAIS workshop at CoRL 2026. Paper on arXiv (link added on announcement). Everything needed to recompute every number in the paper offline is here. No API call is needed for scoring.

## Contents
- `code/sample3.json`: the 240 sampled RoboPulse++ intervals (episode, interval index `k`, label `raw`, frame range, instruction). Drawn by `harness3.py` with seed 20261001 from the released annotations (labels ±1, at least five frames, 120 per label).
- `code/attack_logs2.json`: the eight completion claims and eight control lines per task actually used, written by `gen_a2.py` (`openai/gpt-6-luna` via OpenRouter, low reasoning effort) from the task instruction alone. Control lines matching the regular expression `\b(gripper|arm|object|cup|bowl|door|drawer|table|rack|plate|placed|grasp|lift|moved|position|open|close)` (case-insensitive) were discarded. A task was regenerated if fewer than eight remained.
- `code/harness3.py`: main conditions (frames only, reversed, repeat, progress / regression / neutral log, warned progress log). Prompt text, fixed logs and model settings are the constants at the top of the file.
- `code/harness4.py`: warning batch (batch 2). `code/harness6.py`: original / observe / verify batch (batch 3).
- `code/harness7.py`: completion-claim attack. `code/harness8.py`: the same attack under the verify prompt. Gemini attack subset: 80 Falling intervals, `random.Random(20261001 + 9)`.
- `code/common.py`: the 56 intervals reversal-consistent for all three judges. `code/extra4.py`: paired contrasts for Table 2.
- `code/out*/`: raw judge outputs (one JSON line per call, with the raw answer text and parsed label). `mock_*` files are the scripted control judges.
- `results/`: score reports produced by the scorers, and `audit-claims.md`, the manual frame audit of the 37 intervals.

## Judges
Via OpenRouter, image detail `low`, at most 4,000 output tokens, unparseable answers dropped:
`google/gemini-3.8-flash` (reasoning effort minimal, temperature 0), `qwen/qwen3.7-flash` (reasoning disabled, temperature 0), `openai/gpt-6.1-sol` (reasoning effort low. No temperature parameter). All calls were made on 2026-10-01 (UTC). Frames are the first camera view (`*_view1.png`) of each interval, at most 12, evenly spaced keeping the first and last (`frames_of` in `harness3.py`). They are downloaded from the RoboPulse++ Hugging Face dataset on first use.

## Recompute
```
cd code
uv run harness3.py score --judge qwen37flash      # also gemini38flash, gpt61sol
uv run harness4.py score --judge qwen37flash
uv run harness6.py score --judge qwen37flash
uv run harness7.py score --judge qwen37flash      # also gemini38flash
uv run harness8.py score --judge qwen37flash
uv run common.py . Uv run extra4.py
```
Re-running the judges themselves (`run` instead of `score`) needs an OpenRouter key in `OPENROUTER_API_KEY`.

## Citation
```
@inproceedings{zhao2026execlogs,
  title     = {Execution Logs Sway {VLM} Judges of Robot Progress},
  author    = {Zhao, Weiyan and Yang, Sunny},
  booktitle = {Workshop on Safe and Productive AI Systems (SPAIS) at CoRL 2026},
  year      = {2026}
}
```
