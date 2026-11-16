#!/usr/bin/env python3
"""
Usage: server-battery-diff.py OLD.json NEW.json

Compare two files written by server-battery.py. A layout the old template
rendered must render the same text with the new one; a layout the old one
refused is listed with what the new one did. Exits with status 1 if a layout
that worked before changed.

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote this to compare the chat template's renders on llama.cpp's own
  template engine before and after joining several leading system messages.
"""
import json
import sys

old = json.load(open(sys.argv[1]))
new = json.load(open(sys.argv[2]))
changed = 0
for name in old:
    o, n = old[name], new[name]
    if "prompt" in o:
        same = "prompt" in n and n["prompt"] == o["prompt"]
        changed += not same
        print(f"{name:34} {'identical' if same else 'DIFFERENT'}")
    elif "prompt" in n:
        start = n["prompt"].find("[SYSTEM_PROMPT]")
        end = n["prompt"].find("[/SYSTEM_PROMPT]") + 16
        print(f"{name:34} old refused ({o['error'][-60:]}); new renders {json.dumps(n['prompt'][start:end])}")
    else:
        print(f"{name:34} old refused; new refuses too ({n['error'][-60:]})")
sys.exit(1 if changed else 0)
