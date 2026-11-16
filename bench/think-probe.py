#!/usr/bin/env python3
"""
Usage: think-probe.py URL [TRIALS]
       think-probe.py --history URL [TRIALS]
       think-probe.py --conversation URL [TRIALS] [--tools] [--seed N] [--save FILE]
       think-probe.py --sampling URL [TRIALS]

Send streaming chat requests with reasoning_effort unset, "none" and "high"
(TRIALS of each, default 2) at temperature 0.7 and report, for each, whether
the server returned the thinking as reasoning_content or left "[THINK]" in
the visible content. With --history, the second turn of a conversation is
sent instead, with the first answer stored in the ways a chat front end might
send it back: the plain answer, the answer prefixed by the thinking in a
"[THINK]" block and the answer prefixed by the thinking in an HTML
"<details>" block (Open WebUI's format for hidden thinking). With
--conversation, a two-turn conversation recorded from Open WebUI (system
prompt "Keep answers simple and to the point.", reasoning_effort high, the
first answer with its thinking, then "Make the background red.") is replayed
through the chat completions and the Responses APIs (the first answer is sent
back without its thinking, like Open WebUI does; --tools adds about 1.7k
tokens of tool definitions, like Open WebUI's built-in tools), and each reply is
classified as normal, thinking only (the answer is inside the reasoning and
the content is empty) or leaking "[THINK]". Every request in this mode carries
its own "seed", drawn from a generator seeded with --seed (default 20261116)
and printed, so a bad reply can be replayed with the same request; --save
appends every request, reply and output item type to FILE as JSON lines. With --sampling, the same
conversation is sent through chat completions at several temperatures (and
with the server's default) and the number of thinking-only replies is counted,
along with how the bad replies ended.

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote this probe to find out why Open WebUI sometimes shows a raw
  "[THINK]" tag in the model's output, at the user's request.
"""
import json
import random
import sys
import urllib.request

PROMPTS = ["If a train covers 150 km in 2.5 hours, what is its average speed "
           "in km/h?",
           "Is 221 a prime number? Explain briefly."]


def run(base, prompt, effort, history=None):
    messages = (history or []) + [{"role": "user", "content": prompt}]
    body = {"messages": messages,
            "max_tokens": 1200, "temperature": 0.7, "stream": True}
    if effort is not None:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    content = reasoning = ""
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[6:])
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0]["delta"]
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
    return content, reasoning


FIRST = "If a train covers 150 km in 2.5 hours, what is its average speed?"
THOUGHT = "Speed is distance over time, so 150 divided by 2.5 gives 60."
ANSWER = "The average speed is 60 km/h."
HISTORIES = {
    "plain answer": ANSWER,
    "[THINK] block + answer": f"[THINK]{THOUGHT}[/THINK]{ANSWER}",
    "<details> block + answer": (
        '<details type="reasoning" done="true" duration="2">\n'
        f"<summary>Thought for 2 seconds</summary>\n> {THOUGHT}\n"
        f"</details>\n{ANSWER}"),
}


def main_history(base, trials):
    for label, assistant in HISTORIES.items():
        history = [{"role": "user", "content": FIRST},
                   {"role": "assistant", "content": assistant}]
        for effort in (None, "high"):
            for i in range(trials):
                content, reasoning = run(
                    base, "And how far in 4 hours at that speed?", effort,
                    history)
                print(f"{label:24} effort={effort!s:5} trial={i} reasoning="
                      f"{len(reasoning):4}  content={len(content):4}  "
                      f"[THINK] in content={'[THINK]' in content}  "
                      f"<details> in content={'<details' in content}")


SYSTEM = "Keep answers simple and to the point."
TURN1 = "Show me a code snippet of a website's sticky header in CSS and JavaScript."
ANSWER1 = (
    "```html\n<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
    "    <meta charset=\"UTF-8\">\n    <title>Sticky Header Example</title>\n"
    "    <style>\n        * { margin: 0; padding: 0; box-sizing: border-box; }\n"
    "        body { font-family: Arial, sans-serif; }\n"
    "        .header {\n            background-color: #333;\n"
    "            color: white;\n            padding: 1rem;\n"
    "            text-align: center;\n            position: sticky;\n"
    "            top: 0;\n            z-index: 1000;\n        }\n"
    "        .header h1 { font-size: 2rem; }\n"
    "        .content { padding: 1rem; height: 2000px; }\n    </style>\n"
    "</head>\n<body>\n    <header class=\"header\">\n"
    "        <h1>Website Header</h1>\n    </header>\n"
    "    <div class=\"content\">\n"
    "        <p>Scroll down to see the header stick to the top.</p>\n"
    "    </div>\n</body>\n</html>\n```")
TURN2 = "Make the background red."


def post_json(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def classify(reasoning, content):
    if "[THINK]" in content or "[/THINK]" in content:
        kind = "LEAKS [THINK]"
    elif not content.strip():
        kind = "THINKING ONLY"
    else:
        kind = "normal"
    return (f"{kind:14} reasoning={len(reasoning):4} content={len(content):4}"
            f" answer-code-in-reasoning={'```' in reasoning}")


def fake_tools():
    """Eight tools with long descriptions, about 1.7k tokens in total."""
    names = ["search_web", "fetch_url", "run_python", "read_file",
             "write_file", "search_memory", "add_memory", "generate_image"]
    tools = []
    for name in names:
        text = (f"Use {name.replace('_', ' ')} when the user asks for something "
                "that needs it. Do not guess; call the tool, wait for its "
                "result and then answer using only that result. Arguments "
                "must be valid JSON. Never call this tool more than three "
                "times for a single question, and say so if it fails. ") * 2
        tools.append({"name": name, "description": text, "parameters": {
            "type": "object", "required": ["query"], "properties": {
                "query": {"type": "string", "description": text[:200]},
                "count": {"type": "integer", "description": text[:120]}}}})
    return tools


def main_conversation(base, trials, use_tools, master_seed, save):
    seeds = random.Random(master_seed)
    out = open(save, "a") if save else None
    chat_history = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TURN1},
        {"role": "assistant", "content": ANSWER1},
        {"role": "user", "content": TURN2}]
    tools = fake_tools() if use_tools else []
    for i in range(trials):
        body = {"messages": chat_history, "reasoning_effort": "high",
                "max_tokens": 2000, "seed": seeds.getrandbits(31)}
        if tools:
            body["tools"] = [{"type": "function", "function": t}
                             for t in tools]
        choice = post_json(base, "/v1/chat/completions", body)["choices"][0]
        reply = choice["message"]
        calls = len(reply.get("tool_calls") or [])
        print(f"chat completions  trial={i} seed={body['seed']} " + classify(
            reply.get("reasoning_content") or "", reply.get("content") or "")
            + f" tool_calls={calls}")
        if out:
            out.write(json.dumps({"api": "chat", "trial": i, "seed": body["seed"],
                                  "tools": bool(tools), "finish": choice["finish_reason"],
                                  "reasoning": reply.get("reasoning_content") or "",
                                  "content": reply.get("content") or "",
                                  "tool_calls": calls}) + "\n")
            out.flush()
    for i in range(trials):
        body = {
            "instructions": SYSTEM, "reasoning": {"effort": "high"},
            "max_output_tokens": 2000, "seed": seeds.getrandbits(31), "input": [
                {"role": "user", "content": TURN1},
                {"type": "message", "role": "assistant", "id": "msg_1",
                 "content": [{"type": "output_text", "text": ANSWER1}]},
                {"role": "user", "content": TURN2}]}
        if tools:
            body["tools"] = [{"type": "function", **t} for t in tools]
        reply = post_json(base, "/v1/responses", body)
        reasoning = content = ""
        kinds = [item["type"] for item in reply.get("output", [])]
        for item in reply.get("output", []):
            if item["type"] == "reasoning":
                reasoning += "".join(c["text"] for c in item.get("content", []))
            elif item["type"] == "message":
                content += "".join(c["text"] for c in item["content"])
        print(f"responses         trial={i} seed={body['seed']} " + classify(reasoning, content)
              + f" items={','.join(kinds)}")
        if out:
            out.write(json.dumps({"api": "responses", "trial": i, "seed": body["seed"],
                                  "tools": bool(tools), "items": kinds,
                                  "reasoning": reasoning, "content": content}) + "\n")
            out.flush()


def main_sampling(base, trials):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": TURN1},
                {"role": "assistant", "content": ANSWER1},
                {"role": "user", "content": TURN2}]
    for temperature in (0, 0.3, 0.7, None):
        bad = []
        for _ in range(trials):
            body = {"messages": messages, "reasoning_effort": "high",
                    "max_tokens": 2500}
            if temperature is not None:
                body["temperature"] = temperature
            reply = post_json(base, "/v1/chat/completions", body)
            choice = reply["choices"][0]
            if not (choice["message"].get("content") or "").strip():
                bad.append((choice["finish_reason"],
                            reply["usage"]["completion_tokens"]))
        label = "server default" if temperature is None else temperature
        print(f"temperature={label!s:14} thinking-only {len(bad)}/{trials}"
              f"  (finish_reason, tokens): {bad}")


def main():
    if sys.argv[1] == "--sampling":
        main_sampling(sys.argv[2].rstrip("/"),
                      int(sys.argv[3]) if len(sys.argv) > 3 else 20)
        return
    if sys.argv[1] == "--conversation":
        args = sys.argv[3:]
        master = 20261116
        save = None
        if "--seed" in args:
            master = int(args[args.index("--seed") + 1])
        if "--save" in args:
            save = args[args.index("--save") + 1]
        trials = [a for a in args if a.isdigit() and args[args.index(a) - 1] != "--seed"]
        print(f"master seed {master}")
        main_conversation(sys.argv[2].rstrip("/"),
                          int(trials[0]) if trials else 5,
                          "--tools" in args, master, save)
        return
    if sys.argv[1] == "--history":
        main_history(sys.argv[2].rstrip("/"),
                     int(sys.argv[3]) if len(sys.argv) > 3 else 2)
        return
    base = sys.argv[1].rstrip("/")
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    for effort in (None, "none", "high"):
        for i in range(trials):
            content, reasoning = run(base, PROMPTS[i % len(PROMPTS)], effort)
            print(f"effort={effort!s:5} trial={i} reasoning_content="
                  f"{len(reasoning):4} chars  content={len(content):4} chars  "
                  f"'[THINK]' in content={'[THINK]' in content}  "
                  f"'[/THINK]' in content={'[/THINK]' in content}")
            if "[THINK]" in content or "[/THINK]" in content:
                print("   content starts:", repr(content[:90]))


if __name__ == "__main__":
    main()
