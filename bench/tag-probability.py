#!/usr/bin/env python3
"""
Usage: tag-probability.py URL RESULTS.jsonl

For every reply saved by "think-probe.py --conversation --save" that has a
"[THINK]" or "[/THINK]" tag written out as text, rebuild the prompt (the
recorded conversation, ending in "[THINK]") plus everything the model wrote up
to the first such tag, and print the model's top next-token probabilities
there. Shows whether the stray "[THINK]" is a near-certain restart or a slip
(for example for "[/THINK]", the neighbouring token) and what the model would
write instead.

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote this probe to look at the second "[THINK]" tag the user saw in
  Open WebUI after a thinking block.
"""
import importlib.util
import json
import sys
import urllib.request

spec = importlib.util.spec_from_file_location("probe", "think-probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


base = sys.argv[1].rstrip("/")
messages = [{"role": "system", "content": probe.SYSTEM},
            {"role": "user", "content": probe.TURN1},
            {"role": "assistant", "content": probe.ANSWER1},
            {"role": "user", "content": probe.TURN2}]
prompt = post(base, "/apply-template", {"messages": messages, "reasoning_effort": "high"})["prompt"]
print("prompt ends with:", repr(prompt[-30:]))
for line in open(sys.argv[2]):
    row = json.loads(line)
    text = row["reasoning"] + ("[/THINK]" + row["content"] if row["content"] else "")
    hits = [i for tag in ("[THINK]", "[/THINK]") for i in [text.find(tag)] if i >= 0]
    inside = [(t, row["reasoning"].find(t)) for t in ("[THINK]", "[/THINK]") if row["reasoning"].find(t) >= 0]
    if not (inside or "[THINK]" in row["content"]):
        continue
    cut = min(i for i in hits)
    tokens = post(base, "/tokenize", {"content": prompt + text[:cut], "parse_special": True,
                                      "add_special": False})["tokens"]
    r = post(base, "/completion", {"prompt": tokens, "n_predict": 1, "n_probs": 8,
                                   "temperature": 0, "cache_prompt": True})
    top = r["completion_probabilities"][0].get("top_logprobs") or []
    shown = " | ".join(f"id {e['id']} {2.718281828 ** e['logprob']:.4f}" for e in top[:6])
    print(f"{row['api']} trial {row['trial']} seed {row['seed']}: tag at char {cut} of "
          f"{len(text)}; before it: {text[max(0, cut - 40):cut]!r}\n    top tokens: {shown}")
