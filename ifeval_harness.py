#!/usr/bin/env python3
"""IFEval A/B for Halogen's Qwen3.8-Flash-Next: does reasoning_effort "xhigh" beat "medium"?

Why this is a fair test rather than an argument:
  * The grader is Google's own reference code, imported from the pinned commit
    e6890f85757dd84e27ca6df2dd30651dafad28e0 -- not a reimplementation. Grading the
    reference GPT-4 responses that ship in that commit reproduces the published ~77%
    GPT-4 number (this dir vendored it and verified it once).
  * Every prompt is generated TWICE -- reasoning_effort "medium" and "xhigh" -- so the ONLY
    variable is reasoning effort. The two answers are paired. Generation runs at
    --temperature 0.0 (greedy) unless told otherwise; the model's own serving default is 1.0,
    so pass --temperature 1.0 with a separate --out-dir to measure what it actually serves.
  * Halogen sets effort via the top-level `reasoning_effort` field (its "Reasoning
    parameter mapping" is {"reasoning_effort": reasoningLevel}); that is what we send.
  * Only `content` is graded. `reasoning_content` is kept but never scored -- we grade
    the answer, not the thoughts.
  * The verdict is a paired McNemar exact test over the prompts the two efforts disagree
    on, not just two accuracy numbers a sample could shuffle around.

Reads   : reference/input_data.jsonl (the 541-prompt IFEval set)
Writes  : results/responses.jsonl  (append-only store of every (prompt, effort) reply)
          results/summary.json      (latest metrics)
          results/report.html       (human-readable report of everything in the store)
Run     : .venv/bin/python ifeval_harness.py --limit 200
          (--limit 0 runs the whole 541; existing replies are reused, so it resumes.
           --report-only rebuilds summary+report.html from the store without any calls.)
"""
import argparse
import collections
import concurrent.futures as cf
import html as _html
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "reference")
if REF not in sys.path:
    sys.path.insert(0, REF)

import nltk  # noqa: E402
from instruction_following_eval import instructions_registry as R  # noqa: E402

for _corpus in ("punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
    try:
        nltk.download(_corpus, quiet=True, raise_on_error=False)
    except Exception:
        pass

DEFAULT_API = "http://localhost:8731"
DEFAULT_MODEL = "halogen-qwen3.8-flash-next"
DATA = os.path.join(REF, "input_data.jsonl")
EFFORTS = ("medium", "xhigh")


# --------------------------------------------------------------------------- grading

def grade_each(instruction_id_list, kwargs_list, response):
    """Reference grade for each instruction id. Returns a list of True/False/None
    (None = this commit's registry has no verifier for that id -> unverifiable)."""
    out = []
    for iid, kw in zip(instruction_id_list, kwargs_list):
        cls = R.INSTRUCTION_DICT.get(iid)
        if cls is None:
            out.append(None)
            continue
        try:
            inst = cls(iid)
            inst.build_description(**(kw or {}))
            out.append(bool(inst.check_following(response)))
        except Exception:
            out.append(None)
    return out


def prompt_grade(per_inst):
    if not per_inst or all(v is None for v in per_inst):
        return "unknown"
    if any(v is False for v in per_inst):
        return "fail"
    if any(v is None for v in per_inst):
        return "unknown"
    return "pass"


# --------------------------------------------------------------------------- api

def make_client(api, model, key, timeout, temperature=0.0, api_seed=None):
    base = api.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = base + "/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key

    def ask(effort, prompt, max_tokens):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": effort,
        }
        if api_seed is not None:
            body["seed"] = api_seed
        req = urllib.request.Request(url, json.dumps(body).encode(), headers)
        raw = urllib.request.urlopen(req, timeout=timeout).read()
        d = json.loads(raw)
        choice = (d.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return {
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": (d.get("usage") or {}).get("completion_tokens"),
        }

    return ask


def _progress(done, total, t0):
    el = time.time() - t0
    rate = done / el if el > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0
    w = 38
    fill = min(w, int(w * done / total)) if total else w
    sys.stderr.write("\r[%s%s] %d/%d  %.1f/s  eta %dm%02ds "
                     % ("#" * fill, "." * (w - fill), done, total, rate, int(eta) // 60, int(eta) % 60))
    sys.stderr.flush()


def generate_with_retry(ask, effort, prompt, max_tokens, retries=2, backoff=2.0):
    last = None
    for attempt in range(retries + 1):
        try:
            return ask(effort, prompt, max_tokens), None
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    return None, "%s: %s" % (type(last).__name__, str(last)[:200])


# --------------------------------------------------------------------------- store

def load_store(path):
    store = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                store[(str(rec["key"]), rec["effort"])] = rec
    return store


class Store:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.data = load_store(path)

    def has(self, key, effort):
        with self.lock:
            return (str(key), effort) in self.data

    def put(self, rec):
        with self.lock:
            self.data[(str(rec["key"]), rec["effort"])] = rec
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- metrics

def exact_mcnemar_p(b, c):
    """Exact two-sided McNemar p from the two discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    obs = math.comb(n, b)
    return min(1.0, sum(math.comb(n, k) for k in range(n + 1)
                        if math.comb(n, k) <= obs + 1e-9) / (2 ** n))


def paired_ci(b, c, n_paired):
    """Wald 95% interval for the paired accuracy difference (c - b) / n_paired.

    An exact McNemar p is conservative and discrete, so it answers only "is the gap zero".
    The interval is the useful number: a 2.4 pt gap with a [-0.0, +4.8] interval can still
    mean "no effect", and it is the range a decision should be made against, not the p.
    """
    if not n_paired:
        return None
    diff = 100.0 * (c - b) / n_paired
    se = 100.0 * math.sqrt(b + c) / n_paired
    return {"diff": diff, "se": se, "lo": diff - 1.96 * se, "hi": diff + 1.96 * se}


def compute_metrics(store):
    """Everything in the store, over every prompt that has any reply."""
    keys = sorted({k for (k, _) in store})
    per = {e: {"pass": 0, "fail": 0, "unknown": 0, "truncated": 0} for e in EFFORTS}
    rlen = {e: [0, 0] for e in EFFORTS}
    clen = {e: [0, 0] for e in EFFORTS}
    alen = {e: [0, 0] for e in EFFORTS}
    cat = {}
    for key in keys:
        for e in EFFORTS:
            rec = store.get((key, e))
            if not rec:
                continue
            if rec.get("reasoning_chars") is not None:
                rlen[e][0] += rec["reasoning_chars"]; rlen[e][1] += 1
            if isinstance(rec.get("completion_tokens"), int):
                clen[e][0] += rec["completion_tokens"]; clen[e][1] += 1
            alen[e][0] += len(rec.get("content") or ""); alen[e][1] += 1
            g = rec["grade"]
            if g in ("pass", "fail"):
                per[e]["pass" if g == "pass" else "fail"] += 1
                if rec.get("finish_reason") == "length":
                    per[e]["truncated"] += 1
                for iid, ok in zip(rec.get("instruction_id_list", []), rec.get("per_inst", [])):
                    if ok is None:
                        continue
                    slot = cat.setdefault(iid.split(":")[0], {x: [0, 0] for x in EFFORTS})
                    slot[e][1] += 1
                    if ok:
                        slot[e][0] += 1
            else:
                per[e]["unknown"] += 1

    a = b = c = d = 0
    a_nt = b_nt = c_nt = d_nt = 0
    disc = {"medium_only": [], "xhigh_only": []}
    for key in keys:
        rm_rec = store.get((key, "medium"), {})
        rx_rec = store.get((key, "xhigh"), {})
        rm = rm_rec.get("grade")
        rx = rx_rec.get("grade")
        if rm not in ("pass", "fail") or rx not in ("pass", "fail"):
            continue
        if rm == "pass" and rx == "pass":
            a += 1
        elif rm == "pass" and rx == "fail":
            b += 1
        elif rm == "fail" and rx == "pass":
            c += 1
        else:
            d += 1
        if rm != rx:
            # the only rows that carry information about which effort is better: read the
            # loser's failing instruction ids to see if it has a systematic bad habit.
            loser = rx_rec if rx == "fail" else rm_rec
            disc["medium_only" if rm == "pass" else "xhigh_only"].append({
                "key": key,
                "loser": "xhigh" if rx == "fail" else "medium",
                "failed_ids": [i for i, ok in zip(loser.get("instruction_id_list", []),
                                                  loser.get("per_inst", [])) if ok is False],
                "truncated": loser.get("finish_reason") == "length",
                "prompt": (loser.get("prompt") or "").replace("\n", " ")[:200],
            })
        m_t = rm_rec.get("finish_reason") == "length"
        x_t = rx_rec.get("finish_reason") == "length"
        if not (m_t or x_t):
            if rm == "pass" and rx == "pass":
                a_nt += 1
            elif rm == "pass" and rx == "fail":
                b_nt += 1
            elif rm == "fail" and rx == "pass":
                c_nt += 1
            else:
                d_nt += 1

    def acc(e):
        g = per[e]["pass"] + per[e]["fail"]
        return 100.0 * per[e]["pass"] / g if g else 0.0

    am, ax = acc("medium"), acc("xhigh")
    p, p_nt = exact_mcnemar_p(b, c), exact_mcnemar_p(b_nt, c_nt)
    ci = paired_ci(b, c, a + b + c + d)
    # accuracy per token generated: the axis a single-box deployment actually runs out of.
    # completion_tokens is usage.completion_tokens, i.e. reasoning + answer together.
    cost = {}
    for e in EFFORTS:
        tot, n = clen[e][0], clen[e][1]
        cost[e] = {
            "total_completion_tokens": tot,
            "avg_completion_tokens": (tot // n) if n else None,
            "avg_answer_chars": (alen[e][0] // n) if n else None,
            "passes_per_1k_tokens": (1000.0 * per[e]["pass"] / tot) if tot else None,
        }
    if (a + b + c + d) == 0:
        verdict = "no paired results yet"
    elif p < 0.05:
        winner = "xhigh" if ax > am else ("medium" if am > ax else "tie")
        verdict = "%s is better by %.1f pts, beyond chance (McNemar p=%.4g)" % (winner, abs(ax - am), p)
    elif ax != am:
        mt, xt = cost["medium"]["total_completion_tokens"], cost["xhigh"]["total_completion_tokens"]
        if ci and mt and xt:
            verdict = ("xhigh %.1f%% vs medium %.1f%% -- %.1f pt [%+.1f, %+.1f] for %.2fx the tokens; "
                       "within sampling noise (p=%.3f)"
                       % (ax, am, ax - am, ci["lo"], ci["hi"], float(xt) / mt, p))
        else:
            verdict = "xhigh %.1f%% vs medium %.1f%% -- %.1f pt gap, but within sampling noise (p=%.3f)" % (
                ax, am, ax - am, p)
    else:
        verdict = "identical (%.1f%%); this sample cannot separate them" % am

    return {
        "n_prompts": len(keys),
        "per_effort": per, "accuracy": {"medium": am, "xhigh": ax},
        "avg_reasoning_chars": {e: (rlen[e][0] // rlen[e][1] if rlen[e][1] else None) for e in EFFORTS},
        "avg_completion_tokens": {e: (clen[e][0] // clen[e][1] if clen[e][1] else None) for e in EFFORTS},
        "paired": {"both_pass": a, "both_fail": d, "medium_only": b, "xhigh_only": c},
        "paired_no_trunc": {"both_pass": a_nt, "both_fail": d_nt, "medium_only": b_nt, "xhigh_only": c_nt},
        "mcnemar_p": p, "mcnemar_p_no_trunc": p_nt, "verdict": verdict,
        "paired_ci": ci, "cost": cost, "discordant": disc,
        "categories": {c: {e: cat[c][e] for e in EFFORTS} for c in cat},
    }


def failure_lines(M):
    """What the losing side of each discordant pair actually broke, per side. The aggregate
    is the failure-mode list: it says whether an effort loses on markdown habits, on word
    counts, or at random -- which the headline accuracy cannot."""
    out = []
    for side, label in (("medium_only", "medium passed, xhigh failed"),
                        ("xhigh_only", "xhigh passed, medium failed")):
        rows = M.get("discordant", {}).get(side, [])
        if not rows:
            continue
        freq = collections.Counter(i for r in rows for i in r["failed_ids"])
        top = ", ".join("%s x%d" % (i, n) for i, n in freq.most_common(6)) or "no per-instruction detail"
        trunc = sum(1 for r in rows if r["truncated"])
        out.append("%-26s %2d prompts | %s%s" % (
            label, len(rows), top, ("  [%d truncated]" % trunc) if trunc else ""))
    return out


def text_report(M):
    w, lines = [], None
    out = []
    a = out.append
    a("== IFEval paired A/B  (%d prompts in store) ==" % M["n_prompts"])
    a("%-8s %6s %5s %5s %5s %8s %10s" % ("effort", "graded", "pass", "fail", "unk", "acc", "truncated"))
    for e in EFFORTS:
        d0 = M["per_effort"][e]
        g = d0["pass"] + d0["fail"]
        a("%-8s %6d %5d %5d %5d %7.1f%% %10d" % (
            e, g + d0["unknown"], d0["pass"], d0["fail"], d0["unknown"],
            100.0 * d0["pass"] / g if g else 0.0, d0["truncated"]))
    ar, cost = M["avg_reasoning_chars"], M["cost"]
    a("          avg completion (reasoning + answer):  medium %s tok  |  xhigh %s tok" % (
        cost["medium"]["avg_completion_tokens"], cost["xhigh"]["avg_completion_tokens"]))
    a("          avg answer only:                      medium %s chars | xhigh %s chars"
      "   (reasoning: %s vs %s chars)" % (
          cost["medium"]["avg_answer_chars"], cost["xhigh"]["avg_answer_chars"],
          ar["medium"], ar["xhigh"]))
    pm, px = cost["medium"]["passes_per_1k_tokens"], cost["xhigh"]["passes_per_1k_tokens"]
    if pm and px:
        a("          passes per 1k tokens:               medium %.3f | xhigh %.3f    (medium %.2fx more efficient)"
          % (pm, px, px and pm / px))
    a("")
    pd = M["paired"]
    a("paired over %d prompts gradeable for both:" % sum(pd.values()))
    a("  both pass %d   both fail %d   medium-only %d   xhigh-only %d"
      % (pd["both_pass"], pd["both_fail"], pd["medium_only"], pd["xhigh_only"]))
    a("  McNemar exact two-sided p = %.4g" % M["mcnemar_p"])
    if M.get("paired_ci"):
        ci = M["paired_ci"]
        a("  95%% CI on the paired difference: %+.2f pt [%+.2f, %+.2f] (Wald, se %.2f)"
          % (ci["diff"], ci["lo"], ci["hi"], ci["se"]))
    if sum(M["paired_no_trunc"].values()) != sum(pd.values()):
        a("  excluding truncated replies (%d pairs): p = %.4g"
          % (sum(M["paired_no_trunc"].values()), M["mcnemar_p_no_trunc"]))
    a("")
    a("VERDICT: " + M["verdict"])
    a("")
    if M["categories"]:
        a("per instruction category (instruction-level pass rate, medium vs xhigh):")
        for c0 in sorted(M["categories"]):
            m_ok, m_n = M["categories"][c0]["medium"]
            x_ok, x_n = M["categories"][c0]["xhigh"]
            a("  %-26s %s %s" % (c0,
                ("%5.1f%% (%d)" % (100.0 * m_ok / m_n, m_n)) if m_n else "     n/a    ",
                ("%5.1f%% (%d)" % (100.0 * x_ok / x_n, x_n)) if x_n else "     n/a    "))
        a("")
    if M.get("discordant"):
        a("discordant pairs -- what the losing side broke (full list: results/discordant.md):")
        for ln in failure_lines(M):
            a("  " + ln)
        a("")
    a("note: IFEval scores surface-form instruction following (keywords, length, case,")
    a("      format), not reasoning quality -- a win here means better formatting adherence.")
    return "\n".join(out)


# --------------------------------------------------------------------------- html

_CSS = """
:root{--bg:#f7f7f8;--fg:#16181d;--mut:#6b7280;--card:#fff;--br:#e5e7eb;--blue:#2563eb;
--amber:#b45309;--green:#15803d;--green-bg:#dcfce7;--red:#b91c1c;--red-bg:#fee2e2;--unk:#a16207;}
@media(prefers-color-scheme:dark){:root{--bg:#0d0f13;--fg:#e6e8ec;--mut:#9aa1ad;--card:#16191f;--br:#262b34;
--blue:#60a5fa;--amber:#f59e0b;--green:#4ade80;--green-bg:#14351f;--red:#f87171;--red-bg:#3a1414;--unk:#ca8a04;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",
Roboto,Helvetica,Arial,sans-serif;padding:28px 20px 80px;max-width:1000px;margin-inline:auto}
h1{font-size:22px;margin:0 0 2px}h2{font-size:15px;text-transform:uppercase;letter-spacing:.06em;
color:var(--mut);margin:34px 0 10px;font-weight:600}
.sub{color:var(--mut);font-size:13px;margin-bottom:8px}
.tiles{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}
.tile{flex:1 1 220px;background:var(--card);border:1px solid var(--br);border-radius:12px;padding:16px 18px}
.tile.m{border-left:4px solid var(--blue)}.tile.x{border-left:4px solid var(--amber)}
.tile .lab{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em}
.tile .big{font-size:34px;font-weight:700;line-height:1.1;margin-top:2px;font-variant-numeric:tabular-nums}
.tile .mut{font-size:12px;color:var(--mut);margin-top:4px}
.verdict{background:var(--card);border:1px solid var(--br);border-radius:12px;padding:14px 18px;margin:6px 0 4px;
font-size:15px}.verdict b{font-weight:700}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--br);font-size:13.5px}
th{color:var(--mut);font-weight:600}
td.r{text-align:right}
.bar{height:8px;border-radius:5px;background:var(--br);position:relative;min-width:80px}
.bar>i{position:absolute;left:0;top:0;bottom:0;border-radius:5px}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600}
.p-pass{background:var(--green-bg);color:var(--green)}.p-fail{background:var(--red-bg);color:var(--red)}
.p-unk{background:#3f3f46;color:#fde68a}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;max-width:520px}
.cell{border:1px solid var(--br);border-radius:10px;padding:10px 12px;background:var(--card)}
.cell .k{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
.cell .v{font-size:22px;font-weight:700}
details{border:1px solid var(--br);border-radius:10px;background:var(--card);margin:8px 0}
summary{cursor:pointer;padding:10px 14px;font-size:13.5px;list-style:none;display:flex;gap:10px;align-items:center}
summary::-webkit-details-marker{display:none}
summary .p{flex:1;color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ans{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:4px 14px 14px}
.ans h4{margin:8px 0 4px;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.ans .m h4{color:var(--blue)}.ans .x h4{color:var(--amber)}
pre{white-space:pre-wrap;word-break:break-word;font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
margin:0;background:var(--bg);border:1px solid var(--br);border-radius:8px;padding:10px;max-height:340px;overflow:auto}
.prompt{padding:2px 14px 10px;color:var(--mut);font-size:12.5px}
.meta{font-size:11px;color:var(--mut);margin-top:3px}
a{color:var(--blue)}
"""


def _e(x):
    return _html.escape("" if x is None else str(x))


def _pill(g):
    g = g or "unknown"
    cls = {"pass": "p-pass", "fail": "p-fail"}.get(g, "p-unk")
    return '<span class="pill %s">%s</span>' % (cls, _e(g))


def _bar(pct, color):
    pct = 0 if pct is None else max(0.0, min(100.0, pct))
    return '<div class="bar"><i style="width:%.1f%%;background:%s"></i></div>' % (pct, color)


def render_html(M, store, meta):
    ar, ac = M["avg_reasoning_chars"], M["avg_completion_tokens"]
    p = M["paired"]
    v = M["verdict"]
    parts = ['<!doctype html><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             "<title>IFEval A/B — %s</title>" % _e(meta.get("model", "")),
             "<style>%s</style>" % _CSS,
             "<h1>IFEval A/B · reasoning_effort</h1>",
             '<div class="sub">%s &nbsp;·&nbsp; %s &nbsp;·&nbsp; %d prompts &nbsp;·&nbsp; generated %s</div>'
             % (_e(meta.get("model", "")), _e(meta.get("api", "")), M["n_prompts"], _e(meta.get("ts", "")))]

    # tiles
    parts.append('<div class="tiles">')
    for e, cls in (("medium", "m"), ("xhigh", "x")):
        g = M["per_effort"][e]["pass"] + M["per_effort"][e]["fail"]
        acc = (100.0 * M["per_effort"][e]["pass"] / g) if g else 0.0
        parts.append('<div class="tile %s"><div class="lab">%s</div><div class="big">%.1f%%</div>'
                     '<div class="mut">%d/%d pass · %d fail · %d unk · %d trunc · %s tok · %s char think</div></div>'
                     % (cls, _e(e), acc, M["per_effort"][e]["pass"], g, M["per_effort"][e]["fail"],
                        M["per_effort"][e]["unknown"], M["per_effort"][e]["truncated"],
                        _e(ac[e]), _e(ar[e])))
    parts.append("</div>")
    parts.append('<div class="verdict"><b>Verdict.</b> %s</div>' % _e(v))

    # paired grid
    parts.append("<h2>Paired agreement</h2><div class='grid2'>")
    for k, v2 in (("both pass", p["both_pass"]), ("xhigh only", p["xhigh_only"]),
                  ("medium only", p["medium_only"]), ("both fail", p["both_fail"])):
        parts.append('<div class="cell"><div class="k">%s</div><div class="v">%d</div></div>' % (_e(k), v2))
    parts.append("</div>")
    parts.append('<div class="meta" style="margin-top:8px">McNemar exact two-sided p = %.4g'
                 '%s</div>' % (M["mcnemar_p"],
                 ("" if sum(M["paired_no_trunc"].values()) == sum(p.values())
                  else "  (excluding truncated: p = %.4g)" % M["mcnemar_p_no_trunc"])))
    if M.get("paired_ci"):
        ci = M["paired_ci"]
        parts.append('<div class="meta">95%% CI on the paired difference: %+.2f pt [%+.2f, %+.2f]'
                     '  (Wald, se %.2f pt)</div>' % (ci["diff"], ci["lo"], ci["hi"], ci["se"]))

    if M.get("discordant"):
        parts.append("<h2>Discordant pairs</h2>")
        for ln in failure_lines(M):
            parts.append("<div class='meta'>%s</div>" % _e(ln))
        parts.append("<table><tr><th>side</th><th>prompt</th><th>what the loser broke</th></tr>")
        for side in ("medium_only", "xhigh_only"):
            for r in M["discordant"][side]:
                parts.append("<tr><td>%s</td><td>%s%s</td><td>%s</td></tr>" % (
                    _e("medium won" if side == "medium_only" else "xhigh won"), _e(r["key"]),
                    _e(" (trunc)" if r["truncated"] else ""), _e(", ".join(r["failed_ids"]))))
        parts.append("</table>")

    # categories
    if M["categories"]:
        parts.append("<h2>Per instruction category</h2><table><tr><th>category</th>"
                     "<th>medium</th><th>xhigh</th></tr>")
        for c0 in sorted(M["categories"]):
            m_ok, m_n = M["categories"][c0]["medium"]
            x_ok, x_n = M["categories"][c0]["xhigh"]
            mp = (100.0 * m_ok / m_n) if m_n else None
            xp = (100.0 * x_ok / x_n) if x_n else None
            parts.append("<tr><td>%s</td><td>%s<div class='meta'>%s</div></td><td>%s<div class='meta'>%s</div></td></tr>"
                         % (_e(c0), _bar(mp, "var(--blue)"), ("%d/%d" % (m_ok, m_n)) if m_n else "n/a",
                            _bar(xp, "var(--amber)"), ("%d/%d" % (x_ok, x_n)) if x_n else "n/a"))
        parts.append("</table>")

    # per-prompt
    keys = sorted({k for (k, _) in store})
    parts.append("<h2>Prompts (%d)</h2>" % len(keys))
    for key in keys:
        rm = store.get((key, "medium"), {})
        rx = store.get((key, "xhigh"), {})
        gm = rm.get("grade", "—"); gx = rx.get("grade", "—")
        prompt = rm.get("prompt") or rx.get("prompt") or ""
        ids = (rm or rx).get("instruction_id_list", [])
        parts.append("<details><summary><span>%s</span>%s %s <span class='p'>%s</span></summary>"
                     % (_e(key), _pill(gm), _pill(gx), _e(prompt[:90])))
        parts.append('<div class="prompt">%s</div>' % _e(prompt))
        parts.append('<div class="ans">')
        for cls, rec, eff in (("m", rm, "medium"), ("x", rx, "xhigh")):
            parts.append('<div class="%s"><h4>%s</h4><div class="meta">%d tok · %s char think · finish=%s</div>'
                         % (cls, _e(eff), rec.get("completion_tokens") or 0, rec.get("reasoning_chars") or 0,
                            _e(rec.get("finish_reason"))))
            if ids:
                parts.append('<div class="meta">%s</div>' % _e(" · ".join(ids)))
            parts.append("<pre>%s</pre></div>" % (_e(rec.get("content", "")) or "<empty>"))
        parts.append("</div></details>")

    parts.append('<div class="meta" style="margin-top:26px">IFEval scores surface-form instruction '
                 'following (keywords, length, case, format), not reasoning quality — a win here means '
                 'better formatting adherence, not a smarter model. Graded by Google\'s reference '
                 'instructions.py @ e6890f8.</div>')
    return "".join(parts)


def write_discordant(M, out_dir):
    """Every prompt the two efforts disagree on, with the loser's failing instruction ids.
    43 rows of this is the whole argument; the accuracy numbers are just its summary."""
    lines = ["# Discordant pairs (%d of %d prompts)" % (
        len(M["discordant"]["medium_only"]) + len(M["discordant"]["xhigh_only"]), M["n_prompts"]),
        "", "> " + M["verdict"], ""]
    for side, label in (("medium_only", "medium passed / xhigh failed"),
                        ("xhigh_only", "xhigh passed / medium failed")):
        rows = M["discordant"][side]
        if not rows:
            continue
        lines.append("## %s  (%d)" % (label, len(rows)))
        lines.append("")
        for r in rows:
            lines.append("### `%s`%s" % (r["key"], " — truncated" if r["truncated"] else ""))
            lines.append("- %s failed: %s" % (r["loser"], ", ".join(r["failed_ids"]) or "n/a"))
            lines.append("- prompt: %s" % (r["prompt"] or "(not cached)"))
            lines.append("")
    with open(os.path.join(out_dir, "discordant.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_all(data, out_dir):
    M = compute_metrics(data)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(M, fh, indent=2)
    write_discordant(M, out_dir)
    return M, text_report(M)


# --------------------------------------------------------------------------- main

def select(spec_rows, limit, seed):
    rows = list(spec_rows)
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-url", default=os.environ.get("API_URL", DEFAULT_API))
    ap.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--limit", type=int, default=0, help="prompts to run (0 = all 541)")
    ap.add_argument("--seed", type=int, default=0, help="subset selection seed (stable for resume)")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="generation budget; raise it if xhigh reasoning truncates (finish_reason=length)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature; 0.0 = greedy. The model's serving default is 1.0, "
                         "so measure with that if you want to know how it behaves in production")
    ap.add_argument("--api-seed", type=int, default=None,
                    help="server-side sampling seed; set one per replicate so runs are reproducible. "
                         "Replies are keyed by (prompt, effort) only -- use a separate --out-dir per replicate")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel requests to the shared endpoint")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--print", dest="print_n", type=int, default=0, help="echo first N generated replies")
    ap.add_argument("--report-only", action="store_true", help="skip generation; rebuild report from the store")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    if not os.path.exists(DATA):
        sys.exit("missing %s (vendored IFEval set)" % DATA)
    os.makedirs(args.out_dir, exist_ok=True)
    store = Store(os.path.join(args.out_dir, "responses.jsonl"))
    spec_rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]

    if args.report_only:
        M, txt = write_all(store.data, args.out_dir)
    else:
        selected = select(spec_rows, args.limit, args.seed)
        ask = make_client(args.api_url, args.model, args.api_key, args.timeout,
                          args.temperature, args.api_seed)
        todo = [(s, e) for s in selected for e in EFFORTS if not store.has(s["key"], e)]
        print("%d replies to generate (%d reused from store), %d workers"
              % (len(todo), len(selected) * len(EFFORTS) - len(todo), args.concurrency))
        t0 = time.time()
        done = [0]
        printed = [0]
        lastbar = [0.0]

        def work(item):
            spec, effort = item
            gen, err = generate_with_retry(ask, effort, spec["prompt"], args.max_tokens)
            per_inst = grade_each(spec["instruction_id_list"], spec["kwargs"], gen["content"]) if gen else []
            if gen is None:
                grade = "unknown"
            elif gen["content"] == "":
                grade = "fail"
            else:
                grade = prompt_grade(per_inst)
            rec = {
                "key": spec["key"], "effort": effort,
                "instruction_id_list": spec["instruction_id_list"], "kwargs": spec["kwargs"],
                "prompt": spec["prompt"], "grade": grade, "error": err, "per_inst": per_inst,
                "content": gen["content"] if gen else "",
                "reasoning_chars": len(gen["reasoning"]) if gen else 0,
                "finish_reason": gen["finish_reason"] if gen else None,
                "completion_tokens": gen["completion_tokens"] if gen else None,
                "ts": int(time.time()),
            }
            store.put(rec)
            done[0] += 1
            if args.print_n and printed[0] < args.print_n:
                printed[0] += 1
                print("\n--- [%s] key=%s %s ---\ncontent: %s\n(reasoning %d chars, finish=%s, %s tok)" % (
                    rec["grade"], rec["key"], effort, (rec["content"][:300] or "<empty>"),
                    rec["reasoning_chars"], rec["finish_reason"], rec["completion_tokens"]))
            else:
                now = time.time()
                if done[0] == len(todo) or now - lastbar[0] >= 0.25:
                    lastbar[0] = now
                    _progress(done[0], len(todo), t0)

        if todo:
            if not args.print_n:
                _progress(0, len(todo), t0)          # paint 0/N immediately so it never looks stalled
            with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
                list(pool.map(work, todo))
            if not args.print_n:
                sys.stderr.write("\n")
        M, txt = write_all(store.data, args.out_dir)

    # refresh HTML always, from the full store
    meta = {"model": args.model, "api": args.api_url,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(args.out_dir, "report.html"), "w", encoding="utf-8") as fh:
        fh.write(render_html(M, store.data, meta))
    print(txt)
    print("\nreport: %s" % os.path.join(args.out_dir, "report.html"))


if __name__ == "__main__":
    main()
