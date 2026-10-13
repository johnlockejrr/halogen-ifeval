# halogen-ifeval — does `xhigh` beat `medium` on Halogen's Qwen3.8-Flash-Next?

An A/B harness that settles one argument with data instead of vibes: for the model
served at `halogen-qwen3.8-flash-next` (Halogen), is the **`xhigh`** reasoning level
actually better at following instructions than **`medium`**? It runs the real **IFEval**
benchmark and grades it with Google's own reference verifier.

## Why the number is trustworthy

Most of the fight is about the grader, so the grader is not up for debate:

- **It is Google's code, not a rewrite.** `reference/instruction_following_eval/` is
  vendored verbatim from `google-research/google-research` at commit
  `e6890f85757dd84e27ca6df2dd30651dafad28e0` (`instructions.py`, `instructions_util.py`,
  `instructions_registry.py`). IFEval grading is deterministic code, so "coding our own
  grader" would just produce something shaped like IFEval that scores differently.
- **Self-verified.** Grading the reference **GPT-4** responses that ship in that same
  commit reproduces **77.04%** prompt-level accuracy — the published GPT-4 IFEval figure.
  If the harness can reproduce the reference number, its grading path is correct.
- **Paired design.** Every prompt is generated **twice** at `temperature 0` — once at
  `reasoning_effort:"medium"`, once at `"xhigh"` — so the *only* variable is the reasoning
  level. The two answers for the same prompt are compared head-to-head.
- **Halogen's own contract.** Effort is set through the top-level `reasoning_effort`
  field (Halogen's reasoning-parameter mapping is `{"reasoning_effort": reasoningLevel}`).
  That is exactly what the harness sends; it does not touch `chat_template_kwargs`.
- **Only the answer is graded.** `content` is scored; `reasoning_content` is stored but
  never fed to the verifier — we grade the answer, not the chain of thought.
- **A real statistical test.** The verdict is an exact two-sided **McNemar test** over
  the prompts where the two efforts disagree — not two accuracy numbers that a sample
  could shuffle around. `xhigh 79% vs medium 76%` means nothing without it.

## What it checks

IFEval scores **surface-form instruction following** — keyword presence, word/sentence/
paragraph counts, case, punctuation, section/highlight format, language — across the 541
prompts in `reference/input_data.jsonl`. It is **not** a reasoning-quality benchmark.
A win here means *better formatting adherence*, not a smarter model. That is the honest
scope of "settling" xhigh-vs-medium with IFEval.

## Layout

```
halogen-ifeval/
├── ifeval_harness.py            # the whole tool (stdlib only)
├── README.md
├── .venv/                       # venv: langdetect, immutabledict, absl-py, nltk, regex
├── reference/
│   ├── input_data.jsonl         # the 541-prompt IFEval set (from the pinned commit)
│   └── instruction_following_eval/   # Google's reference verifier (pinned commit, Apache-2.0)
└── results/
    ├── responses.jsonl          # append-only store of every (prompt, effort) reply
    ├── summary.json             # latest metrics (accuracy, paired table, McNemar p, verdict)
    └── report.html              # human-readable report of everything in the store
```

## Run it

```bash
cd ~/halogen-ifeval
```

Every reply is cached in `results/responses.jsonl`, so any run **resumes** — it only
generates the (prompt, effort) pairs it doesn't already have. Delete that file to reset.

```bash
# a quick smoke test — 2 prompts x2, echoes the raw replies + grades them
.venv/bin/python ifeval_harness.py --limit 2 --print 4

# the agreed first run — a 200-prompt random subset (seed 0), both efforts
.venv/bin/python ifeval_harness.py --limit 200 --seed 0

# the whole benchmark
.venv/bin/python ifeval_harness.py --limit 0
```

During a run a live bar prints to stderr:

```
[############····························] 142/400  3.4/s  eta 2m09s
```

Then it prints the summary and refreshes `results/report.html`. To rebuild the report
from the store without generating anything:

```bash
.venv/bin/python ifeval_harness.py --report-only
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--limit N` | `0` (all 541) | how many prompts to generate for (`0` = full set) |
| `--seed N` | `0` | which subset that `N` selects — fixed so runs are reproducible and resume cleanly |
| `--max-tokens N` | `8192` | generation budget. Raise it if `xhigh` reasoning runs out of room (`finish_reason:"length"`) and truncates the answer |
| `--concurrency N` | `4` | parallel requests to the shared endpoint |
| `--api-url` | Halogen URL | override the endpoint (or `API_URL`) |
| `--model` | `halogen-qwen3.8-flash-next` | model id (or `MODEL`) |
| `--api-key` | — | bearer token if the endpoint needs one (or `API_KEY`) |
| `--print N` | `0` | echo the first N generated replies to stdout |
| `--report-only` | off | skip generation, rebuild `summary.json` + `report.html` from the store |
| `--out-dir` | `./results` | where outputs are written |

## Reading the report

`results/report.html` shows the two accuracy tiles, a **verdict** line, the paired
agreement grid (both-pass / both-fail / medium-only / xhigh-only + the McNemar p), a
per-category breakdown (keywords vs length vs format, …), and every prompt with its
medium and xhigh answers side by side so you can audit the grading yourself.

The console summary says the same in text, e.g.:

```
VERDICT: xhigh 79.2% vs medium 76.1% -- 3.1 pt gap, but within sampling noise (p=0.456)
```

`--limit 200` gives roughly ±3 pt. A sub-5% McNemar p on 200 prompts is a genuine
difference; anything above it is a direction, not a settlement. Run `--limit 0` if you
want it airtight.

### One thing to watch: the reasoning budget

`max_tokens` covers `reasoning_content` **and** `content` together. If `xhigh` plans at
length and the budget is too low, the answer gets truncated or comes back empty and loses
on formatting for a reason that has nothing to do with reasoning quality — which would
unfairly penalise `xhigh`. The harness flags truncated replies (`finish_reason:"length"`)
and reports the McNemar p both with and without them; raise `--max-tokens` if truncation
is lopsided toward one effort.

## Setup notes

- The venv is needed only for the grader's pure-Python deps. To recreate elsewhere:
  `python3 -m venv .venv && .venv/bin/pip install langdetect immutabledict absl-py nltk regex`.
  (`nltk` corpora download automatically on first run.)
- The reference `instruction_following_eval` package is Apache License 2.0, © Google
  Research; `input_data.jsonl` and the verifier come from the commit pinned above.
- The Halogen endpoint is shared — keep `--concurrency` modest.
