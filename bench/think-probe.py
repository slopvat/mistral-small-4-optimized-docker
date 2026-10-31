#!/usr/bin/env python3
"""
Usage: think-probe.py URL [TRIALS]
       think-probe.py --history URL [TRIALS]

Send streaming chat requests with reasoning_effort unset, "none" and "high"
(TRIALS of each, default 2) at temperature 0.7 and report, for each, whether
the server returned the thinking as reasoning_content or left "[THINK]" in
the visible content. With --history, the second turn of a conversation is
sent instead, with the first answer stored in the ways a chat front end might
send it back: the plain answer, the answer prefixed by the thinking in a
"[THINK]" block and the answer prefixed by the thinking in an HTML
"<details>" block (Open WebUI's format for hidden thinking).

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote this probe to find out why Open WebUI sometimes shows a raw
  "[THINK]" tag in the model's output, at the user's request.
"""
import json
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


def main():
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
