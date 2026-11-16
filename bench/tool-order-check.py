#!/usr/bin/env python3
"""
Usage: tool-order-check.py URL [TRIALS [MASTER_SEED]]

Ask a llama-server running Mistral Small 4 questions that need a tool, with
reasoning_effort "high", and read the generated token ids (from logprobs) to see
where "[TOOL_CALLS]" comes relative to "[/THINK]": after the block is closed,
or inside it. Two arms: the server's defaults (the image's guards on) and
reasoning_block_eog / reasoning_single_block switched off per request. Every
request carries its own seed, drawn from MASTER_SEED (default 20261116) and
printed with the examples. The prompt already ends with "[THINK]" (the image's
chat template), so the block is open from the first generated token.

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote this to answer the user's question whether Mistral writes a tool
  call inside its [THINK]...[/THINK] block or after closing it.
"""
import json
import random
import sys
import urllib.request

base = sys.argv[1].rstrip("/")
trials = int(sys.argv[2]) if len(sys.argv) > 2 else 12
seeds = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 20261116)


def post(path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def token_id(text):
    ids = post("/tokenize", {"content": text, "parse_special": True, "add_special": False})["tokens"]
    assert len(ids) == 1, (text, ids)
    return ids[0]


THINK_END = token_id("[/THINK]")
TOOL_CALLS = token_id("[TOOL_CALLS]")
ARGS = token_id("[ARGS]")
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_time", "description": "Get the current local time for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
]
QUESTIONS = [
    "What is the weather in Paris right now?",
    "What time is it in Tokyo, and what is the weather there?",
    "Is it raining in London at the moment?",
    "I'm flying to Sydney tonight. What is the weather and local time there?",
    "Tell me the current time in New York.",
    "Check the weather in Madrid and in Rome, please.",
]
SYSTEM = {"role": "system", "content": "Keep answers simple and to the point."}


def run(guards):
    rows = []
    for i in range(trials):
        body = {"messages": [SYSTEM, {"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}],
                "tools": TOOLS, "reasoning_effort": "high", "max_tokens": 3000,
                "logprobs": True, "top_logprobs": 1, "seed": seeds.getrandbits(31)}
        if not guards:
            body["reasoning_block_eog"] = False
            body["reasoning_single_block"] = False
        reply = post("/v1/chat/completions", body)
        choice = reply["choices"][0]
        toks = choice["logprobs"]["content"]
        ids = [t["id"] for t in toks]
        end = ids.index(THINK_END) if THINK_END in ids else None
        call = ids.index(TOOL_CALLS) if TOOL_CALLS in ids else None
        msg = choice["message"]
        rows.append({
            "seed": body["seed"], "question": i % len(QUESTIONS), "finish": choice["finish_reason"],
            "tokens": len(ids), "think_end_at": end, "tool_calls_at": call,
            "n_tool_call_tokens": ids.count(TOOL_CALLS), "n_args": ids.count(ARGS),
            "parsed_calls": len(msg.get("tool_calls") or []),
            "content_chars": len((msg.get("content") or "")),
            "reasoning_chars": len(msg.get("reasoning_content") or ""),
            "between": (call - end - 1) if (end is not None and call is not None and call > end) else None,
            "tail": "".join(t["token"] for t in toks[max(0, (end if end is not None else 0)):
                                                   (call + 6 if call is not None else len(toks))][:40]),
        })
    return rows


def summarize(label, rows):
    called = [r for r in rows if r["tool_calls_at"] is not None]
    after = [r for r in called if r["think_end_at"] is not None and r["tool_calls_at"] > r["think_end_at"]]
    inside = [r for r in called if r["think_end_at"] is None or r["tool_calls_at"] < r["think_end_at"]]
    print(f"\n== {label}: {len(rows)} requests, {len(called)} with a tool call; "
          f"[TOOL_CALLS] after [/THINK] in {len(after)}, inside the block in {len(inside)}")
    print("   finish reasons:", sorted({r['finish'] for r in rows}),
          "| parsed calls match [TOOL_CALLS] count:",
          all(r["parsed_calls"] == r["n_tool_call_tokens"] for r in called),
          "| no answer text before a call:", all(r["between"] in (0, None) for r in called))
    for r in rows[:3]:
        print(f"   seed={r['seed']} q{r['question']} finish={r['finish']} tokens={r['tokens']} "
              f"[/THINK] at {r['think_end_at']} [TOOL_CALLS] at {r['tool_calls_at']} "
              f"(x{r['n_tool_call_tokens']}) between={r['between']} content_chars={r['content_chars']}")
        print("     from [/THINK]:", repr(r["tail"][:150]))
    for r in inside:
        print("   INSIDE:", json.dumps(r)[:300])
    return rows


results = {"guards on": summarize("server defaults (guards on)", run(True)),
           "guards off": summarize("reasoning_block_eog and reasoning_single_block off", run(False))}
json.dump(results, open("tool-order-results.json", "w"), indent=1)
