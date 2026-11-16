#!/usr/bin/env python3
"""
Usage: eos-bias-check.py URL [BIAS [TRIALS]]

Raise the end-of-sequence token's logit by BIAS (default 12) so the model
ends its reply early, far more often than it does by itself, and count the
thinking-mode replies that end inside the thinking block, judged from the
token ids ([THINK] is 34 and [/THINK] is 35; a bias that also ends the reply
right after [/THINK] is not counted). When the chat template ends the prompt
with "[THINK]" (PREFILL_THINK), the block is open from the first token. Run against a server with
"--reasoning-block-eog": the request field "reasoning_block_eog": false
switches the protection off for the control arm. Arms: plain chat, chat with a
tool, and the Responses API (which has no token ids, so only the average
length and the number of replies shorter than 30 tokens are shown).

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote this diagnostic to prove that the guard really stops the
  end-of-sequence token inside thinking in every request path, rather than
  relying on the rare natural failures.
"""
import importlib.util
import json
import sys
import urllib.request

spec = importlib.util.spec_from_file_location("probe", "grammar-probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

base = sys.argv[1].rstrip("/")
bias = float(sys.argv[2]) if len(sys.argv) > 2 else 12
trials = int(sys.argv[3]) if len(sys.argv) > 3 else 8
QUESTION = "What is 17 * 23? Explain briefly."


def post(path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


rendered = post("/apply-template", {"messages": [{"role": "user", "content": QUESTION}],
                                     "reasoning_effort": "high"})["prompt"]
prefilled = rendered.endswith("[THINK]")
print("prompt ends with [THINK]:", prefilled)
for arm in ("chat", "chat with tools", "responses"):
    for guard in (False, True):
        stuck = total = 0
        for _ in range(trials):
            extra = {"logit_bias": [[2, bias]]}
            if not guard:
                extra["reasoning_block_eog"] = False
            if arm == "responses":
                r = post("/v1/responses", {"input": QUESTION, "reasoning": {"effort": "high"},
                                           "max_output_tokens": 2500, **extra})
                tokens = r["usage"]["output_tokens"]
                bad = tokens < 30
            else:
                body = {"messages": [{"role": "user", "content": QUESTION}],
                        "reasoning_effort": "high", "max_tokens": 2500,
                        "logprobs": True, "top_logprobs": 1, **extra}
                if arm == "chat with tools":
                    body["tools"] = probe.TOOLS
                r = post("/v1/chat/completions", body)
                tokens = r["usage"]["completion_tokens"]
                ids = [t["id"] for t in r["choices"][0]["logprobs"]["content"]]
                start = len(ids) - 1 - ids[::-1].index(34) if 34 in ids else 0
                bad = (prefilled or 34 in ids) and 35 not in ids[start:]
            stuck += bad
            total += tokens
        what = "shorter than 30 tokens" if arm == "responses" else "ended inside the thinking block"
        print(f"{arm:16} guard {'on ' if guard else 'off'}: {stuck}/{trials} {what}, "
              f"{total // trials} tokens per reply on average", flush=True)
