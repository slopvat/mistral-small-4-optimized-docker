#!/usr/bin/env python3
"""
Usage: role-check.py URL [MASTER_SEED]

Send the same question to a llama-server with the Mistral Small 4 template in
different system-message layouts and print the HTTP status and the error's
last line: Pi's "developer" first, an Open WebUI-style "system" in front of
Pi's "developer" or "system" message, a single "system", and a later "system".
Every request carries its own seed from MASTER_SEED (default 20261116).

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote this check to reproduce the "Only user, assistant and tool
  roles are supported, got system" error the user got from Pi through Open
  WebUI after adding "reasoning": true to a Pi model entry.
"""
import json
import random
import sys
import urllib.error
import urllib.request

base = sys.argv[1].rstrip("/")
seeds = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 20261116)
U = {"role": "user", "content": "Say hi."}
CASES = [
    ("[developer, user]  (Pi, reasoning true, direct)", [{"role": "developer", "content": "Be brief."}, U]),
    ("[system, user]  (Pi, reasoning false)", [{"role": "system", "content": "Be brief."}, U]),
    ("[system, developer, user]  (Open WebUI in front of Pi)", [{"role": "system", "content": "OWUI prompt."}, {"role": "developer", "content": "Be brief."}, U]),
    ("[system, system, user]", [{"role": "system", "content": "OWUI prompt."}, {"role": "system", "content": "Be brief."}, U]),
    ("[system+merged, user]  (Open WebUI merging into Pi's system)", [{"role": "system", "content": "OWUI prompt.\n\nBe brief."}, U]),
    ("[developer, user, system]  (system later)", [{"role": "developer", "content": "Be brief."}, U, {"role": "system", "content": "late"}]),
]
for label, messages in CASES:
    body = {"messages": messages, "max_tokens": 30, "seed": seeds.getrandbits(31), "reasoning_effort": "none"}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=120))
        res = "200 " + repr(r["choices"][0]["message"]["content"][:40])
    except urllib.error.HTTPError as e:
        msg = json.loads(e.read()).get("error", {}).get("message", "")
        res = f"{e.code} " + msg.strip().splitlines()[-1][:110]
    print(f"{label:62} seed={body['seed']:<11} {res}")
