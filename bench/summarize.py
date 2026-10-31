#!/usr/bin/env python3
"""
Usage: summarize.py [RESULTS_DIR]

Print the tables in README.md from the files in RESULTS_DIR (default:
"results" next to this script).

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote this script so the README's tables can be regenerated from the
  raw benchmark results for the user's request for optimized Mistral Small 4
  images.
"""
import glob
import json
import os
import re
import sys

here = os.path.dirname(os.path.abspath(__file__))
results = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results")


def load(pattern):
    out = {}
    for path in sorted(glob.glob(os.path.join(results, pattern))):
        with open(path) as f:
            out[os.path.basename(path)[:-5]] = json.load(f)
    return out


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" if i == 0 else "--:"
                         for i in range(len(header))) + "|")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")
    print()


def ppl(quant):
    path = os.path.join(results, f"ppl-{quant}.txt")
    if not os.path.exists(path):
        return None
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)",
                  open(path).read())
    return f"{m.group(1)} ± {m.group(2)}" if m else None


print("### Speculative decoding settings (MXFP4_MOE, 1 GPU)\n")
rows = []
for name, r in load("spec-mxfp4-*.json").items():
    rows.append((name[len("spec-mxfp4-"):], r["tok_s"], r["accept_rate"] or "-"))
rows.append(("baseline (no speculation)", load("mxfp4-baseline.json")
             ["mxfp4-baseline"]["tok_s"], "-"))
rows.sort(key=lambda r: -r[1])
table(["Setting", "tok/s", "Accepted"], rows)

print("### 4-bit quants (1 GPU, 12 prompts, 3 counted passes)\n")
quants = [("UD-Q4_K_XL", "ud-q4-k-xl"), ("UD-Q4_K_S", "ud-q4-k-s"),
          ("UD-IQ4_XS", "ud-iq4-xs"), ("UD-IQ4_NL", "ud-iq4-nl"),
          ("MXFP4_MOE", "mxfp4-moe"), ("UD-Q4_K_M", "ud-q4-k-m")]
data = load("c4-*.json")
rows = []
for label, key in quants:
    base = data[f"c4-{key}-baseline"]["tok_s"]
    spec = {n: data[f"c4-{key}-dspark-n{n}"] for n in (2, 3)}
    best_n = max(spec, key=lambda n: spec[n]["tok_s"])
    best = spec[best_n]
    rows.append((label, base, best["tok_s"], f"n={best_n}",
                 best["accept_rate"], ppl(label) or "-"))
fastest = max(r[2] for r in rows)
rows = [r + ("yes" if r[2] >= 0.9 * fastest else "no",) for r in rows]
table(["Quant", "Plain tok/s", "Best speculative tok/s", "Draft n",
       "Accepted", "WikiText-2 PPL", "Within 10% of fastest"], rows)

q8 = load("q8-*.json")
print("### 8-bit UD-Q8_K_XL (2 GPUs)\n")
rows = [(n[3:], r["tok_s"], r["accept_rate"] or "-") for n, r in q8.items()]
rows.sort(key=lambda r: -r[1])
table(["Setting", "tok/s", "Accepted"], rows)
print("PPL:", ppl("UD-Q8_K_XL"))

print("\n### Shipped images\n")
rows = []
for name, r in load("image-*.json").items():
    if "users" in r:
        rows.append((name[len("image-"):], r["users"], r["total_tok_s"],
                     r["per_user_tok_s"]))
table(["Run", "Users", "Total tok/s", "Per user tok/s"], rows)
