#!/usr/bin/env python3
"""
Usage: restart-bias-check.py URL [BIAS [TRIALS [SEED [BUDGET]]]]

Raise the "[THINK]" token's logit by BIAS (default 10) so the model writes it
again inside the thinking block or after it, far more often than it does by
itself, and count the thinking-mode replies that contain a generated "[THINK]"
(token id 34) once the prompt has opened the block, split by whether it came
before or after the "[/THINK]" (id 35). The request field
"reasoning_single_block": false switches the protection off for the control
arm. With BUDGET, the server ends the thinking after BUDGET tokens
(reasoning_budget_tokens), so that a "[THINK]" after the block can be seen
even with a strong bias. Every request has its own seed, drawn from SEED (default 20261116) and
printed in the failure lines.

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote this diagnostic to prove that --reasoning-single-block stops a
  second "[THINK]" inside and after the thinking block, which the user saw in
  Open WebUI, without relying on the rare natural slips.
"""
import json
import random
import sys
import urllib.request

base = sys.argv[1].rstrip("/")
bias = float(sys.argv[2]) if len(sys.argv) > 2 else 10
trials = int(sys.argv[3]) if len(sys.argv) > 3 else 8
seeds = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 20261116)
budget = int(sys.argv[5]) if len(sys.argv) > 5 else None
QUESTION = "How many trailing zeros does 100! have? Explain briefly."


def post(path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


rendered = post("/apply-template", {"messages": [{"role": "user", "content": QUESTION}],
                                     "reasoning_effort": "high"})["prompt"]
print("prompt ends with [THINK]:", rendered.endswith("[THINK]"))
for guard in (False, True):
    inside = after = closed = 0
    for _ in range(trials):
        seed = seeds.getrandbits(31)
        body = {"messages": [{"role": "user", "content": QUESTION}], "reasoning_effort": "high",
                "max_tokens": 500, "logprobs": True, "top_logprobs": 1, "seed": seed,
                "logit_bias": [[34, bias]]}
        if budget is not None:
            body["reasoning_budget_tokens"] = budget
        if not guard:
            body["reasoning_single_block"] = False
        ids = [t["id"] for t in post("/v1/chat/completions", body)["choices"][0]["logprobs"]["content"]]
        end = ids.index(35) if 35 in ids else None
        closed += end is not None
        before = [i for i, t in enumerate(ids) if t == 34 and (end is None or i < end)]
        later = [i for i, t in enumerate(ids) if t == 34 and end is not None and i > end]
        inside += bool(before)
        after += bool(later)
        if guard and (before or later):
            print("   seed", seed, "still has [THINK] at positions", before + later)
    print(f"guard {'on ' if guard else 'off'}: [THINK] inside the block in {inside}/{trials} replies, "
          f"after [/THINK] in {after}/{trials}; {closed}/{trials} replies reached [/THINK]", flush=True)
