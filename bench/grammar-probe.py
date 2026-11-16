#!/usr/bin/env python3
"""
Usage: grammar-probe.py URL TRIALS LABEL OUTPUT.jsonl [--grammar FILE]
                        [--tools] [--temperature T] [--max-tokens N]
                        [--prompt N] [--field KEY=JSON]...

Send TRIALS thinking-mode chat requests (reasoning_effort "high") to a
llama-server, rotating over a few prompts, and count the replies that end
with reasoning but no answer, which is what happens when the model emits its
end-of-sequence token before [/THINK]. With --grammar, the file's contents
are sent as the request's "grammar". With --prompt, every trial uses prompt
N. Each --field adds KEY with the JSON value to the request body (e.g.
--field reasoning_block_eog=false). Each trial, including the full reasoning and answer text, is appended to
OUTPUT.jsonl.

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote this probe to measure whether a token grammar that requires
  [/THINK] removes the early-stop failure the user saw with Mistral Small 4.
"""
import argparse
import json
import time
import urllib.request

HTML = ("```html\n<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "    <title>Sticky Header Example</title>\n    <style>\n"
        "        .header { background-color: #333; color: white;\n"
        "                  padding: 1rem; position: sticky; top: 0; }\n"
        "    </style>\n</head>\n<body>\n    <header class=\"header\">\n"
        "        <h1>Website Header</h1>\n    </header>\n</body>\n</html>\n```")
SYSTEM = {"role": "system", "content": "Keep answers simple and to the point."}
CHATS = [
    [SYSTEM,
     {"role": "user", "content": "Show me a code snippet of a website's sticky header in CSS and JavaScript."},
     {"role": "assistant", "content": HTML},
     {"role": "user", "content": "Make the background red."}],
    [SYSTEM, {"role": "user", "content": "Write a Python function that merges overlapping intervals."}],
    [SYSTEM, {"role": "user", "content": "How many trailing zeros does 100! have, and why?"}],
    [SYSTEM, {"role": "user", "content": "Explain the difference between a mutex and a semaphore."}],
    [SYSTEM, {"role": "user", "content": "Write a shell one-liner that prints the ten largest files under the current directory."}],
    [SYSTEM, {"role": "user", "content": "A train leaves at 9:15 and travels 210 km at 84 km/h. When does it arrive?"}],
]
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("trials", type=int)
    ap.add_argument("label")
    ap.add_argument("output")
    ap.add_argument("--grammar")
    ap.add_argument("--tools", action="store_true")
    ap.add_argument("--prompt", type=int)
    ap.add_argument("--field", action="append", default=[])
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-tokens", type=int, default=6000)
    args = ap.parse_args()
    grammar = open(args.grammar).read() if args.grammar else None
    bad = errors = 0
    with open(args.output, "a") as out:
        for i in range(args.trials):
            index = args.prompt if args.prompt is not None else i % len(CHATS)
            body = {"messages": CHATS[index], "reasoning_effort": "high",
                    "max_tokens": args.max_tokens}
            if args.temperature is not None:
                body["temperature"] = args.temperature
            if grammar:
                body["grammar"] = grammar
            for field in args.field:
                key, _, value = field.partition("=")
                body[key] = json.loads(value)
            if args.tools:
                body["tools"] = TOOLS
            req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions",
                                         json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            start = time.time()
            try:
                reply = json.load(urllib.request.urlopen(req, timeout=600))
            except Exception as exc:
                rec = {"label": args.label, "trial": i, "error": str(exc)[:300]}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                errors += 1
                print(i, "ERROR", rec["error"], flush=True)
                continue
            choice = reply["choices"][0]
            msg = choice["message"]
            content = (msg.get("content") or "").strip()
            reasoning = msg.get("reasoning_content") or ""
            calls = msg.get("tool_calls") or []
            unclosed = not content and not calls and bool(reasoning)
            bad += unclosed
            t = reply.get("timings", {})
            rec = {"label": args.label, "trial": i, "prompt": index,
                   "finish": choice["finish_reason"], "unclosed": unclosed,
                   "content_chars": len(content), "reasoning_chars": len(reasoning),
                   "tool_calls": len(calls),
                   "completion_tokens": reply["usage"]["completion_tokens"],
                   "predicted_per_second": t.get("predicted_per_second"),
                   "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
                   "seconds": round(time.time() - start, 2),
                   "reasoning": reasoning, "content": content}
            if unclosed or not content and not calls:
                rec["tail"] = (reasoning or content)[-300:]
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if unclosed or choice["finish_reason"] == "length":
                print(i, "unclosed" if unclosed else "length", rec["completion_tokens"], flush=True)
    print(f"{args.label}: {bad}/{args.trials} replies ended inside the thinking block, {errors} request errors")


if __name__ == "__main__":
    main()
