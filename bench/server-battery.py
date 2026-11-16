#!/usr/bin/env python3
"""
Usage: server-battery.py URL OUTPUT.json

Send the message layouts of template-battery.py to a llama-server's
/apply-template and save, for each, the rendered prompt or the error. Run it
against a server with the old template and one with the new template, then
compare the two files with server-battery-diff.py.

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote this so the chat template change is compared on llama.cpp's own
  template engine, not only on jinja2.
"""
import json
import sys
import urllib.error
import urllib.request

src = open("template-battery.py").read().split("bad = 0")[0]
ns = {}
exec(compile(src, "template-battery.py", "exec"), ns)
base = sys.argv[1].rstrip("/")
out = {}
for name, case in ns["CASES"].items():
    body = {"messages": case["messages"]}
    for key in ("tools", "reasoning_effort"):
        if key in case:
            body[key] = case[key]
    req = urllib.request.Request(base + "/apply-template", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        out[name] = {"prompt": json.load(urllib.request.urlopen(req, timeout=60))["prompt"]}
    except urllib.error.HTTPError as exc:
        text = exc.read().decode()
        out[name] = {"error": text.strip().splitlines()[-1][:120] if text.strip() else str(exc.code)}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(len(out), "layouts saved,", sum("error" in v for v in out.values()), "errors")
