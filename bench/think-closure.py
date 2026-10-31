#!/usr/bin/env python3
"""
Usage: think-closure.py URL [TRIALS [TEMPERATURE]]

Replay the second turn of a short coding chat (system prompt "Keep answers
simple and to the point.", reasoning_effort "high", the server's default
sampling unless TEMPERATURE is given) TRIALS times (default 20) and count the replies whose thinking never
closes, i.e. that come back with reasoning_content and no visible answer.

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote this probe to measure how often the model fails to close its
  thinking block, which the user saw in Open WebUI, and whether speculative
  decoding changes it.
"""
import json
import sys
import urllib.request

HTML = ("```html\n<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "    <meta charset=\"UTF-8\">\n    <title>Sticky Header Example</title>\n"
        "    <style>\n        .header {\n            background-color: #333;\n"
        "            color: white;\n            padding: 1rem;\n"
        "            position: sticky;\n            top: 0;\n        }\n"
        "        .content {\n            height: 2000px;\n        }\n"
        "    </style>\n</head>\n<body>\n    <header class=\"header\">\n"
        "        <h1>Website Header</h1>\n    </header>\n"
        "    <div class=\"content\"><p>Scroll down.</p></div>\n</body>\n"
        "</html>\n```")
MESSAGES = [
    {"role": "system", "content": "Keep answers simple and to the point."},
    {"role": "user", "content": "Show me a code snippet of a website's "
                                "sticky header in CSS and JavaScript."},
    {"role": "assistant", "content": HTML},
    {"role": "user", "content": "Make the background red."},
]


def main():
    base = sys.argv[1].rstrip("/")
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    temperature = float(sys.argv[3]) if len(sys.argv) > 3 else None
    unclosed = drafted = accepted = 0
    for i in range(trials):
        body = {"messages": MESSAGES, "reasoning_effort": "high",
                "max_tokens": 2500}
        if temperature is not None:
            body["temperature"] = temperature
        req = urllib.request.Request(base + "/v1/chat/completions",
                                     json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        reply = json.load(urllib.request.urlopen(req, timeout=300))
        choice = reply["choices"][0]
        msg = choice["message"]
        bad = not (msg.get("content") or "").strip()
        unclosed += bad
        t = reply.get("timings", {})
        drafted += t.get("draft_n", 0)
        accepted += t.get("draft_n_accepted", 0)
        if bad:
            print(f"  trial {i}: UNCLOSED, reasoning="
                  f"{len(msg.get('reasoning_content') or '')} chars, "
                  f"finish_reason={choice['finish_reason']}")
    rate = f", draft acceptance {accepted / drafted:.2f}" if drafted else ""
    print(f"{unclosed}/{trials} replies had no visible answer{rate}")


if __name__ == "__main__":
    main()
