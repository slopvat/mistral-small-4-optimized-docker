#!/usr/bin/env python3
"""
Usage: template-battery.py OLD_TEMPLATE NEW_TEMPLATE

Render two copies of the Mistral Small 4 chat template over the same message
layouts with Python's jinja2 (a stand-in for llama.cpp's engine, used only to
compare the two files) and report, for each layout, whether the old template
errored, whether the new one did and whether the two renders are identical.
Exits with status 1 if a layout the old template rendered differs in the new
one.

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote this comparison to show that joining several leading system
  messages in the chat template changes no render that already worked.
"""
import datetime
import json
import sys

import jinja2


def render(path, case):
    env = jinja2.Environment()
    env.globals["strftime_now"] = lambda fmt: datetime.datetime(2026, 11, 16).strftime(fmt)

    def raise_exception(message):
        raise RuntimeError(message)

    env.globals["raise_exception"] = raise_exception
    template = env.from_string(open(path).read())
    return template.render(**case)


S = lambda t: {"role": "system", "content": t}
U = lambda t: {"role": "user", "content": t}
A = lambda t: {"role": "assistant", "content": t}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "w",
                                          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
CALL = {"role": "assistant", "content": "", "tool_calls": [
    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}
TOOL = {"role": "tool", "tool_call_id": "c1", "content": "18 C"}
CASES = {
    "user only": {"messages": [U("hi")]},
    "one system": {"messages": [S("Be brief."), U("hi")]},
    "one system with newlines": {"messages": [S("Line 1\n\nLine 2\n"), U("hi")]},
    "empty system": {"messages": [S(""), U("hi")]},
    "system as two text blocks": {"messages": [{"role": "system", "content": [
        {"type": "text", "text": "Be "}, {"type": "text", "text": "brief."}]}, U("hi")]},
    "system with an image block": {"messages": [{"role": "system", "content": [
        {"type": "image_url", "image_url": {"url": "x"}}]}, U("hi")]},
    "two systems": {"messages": [S("OWUI prompt."), S("Be brief."), U("hi")]},
    "three systems": {"messages": [S("a"), S("b"), S("c"), U("hi")]},
    "empty then system": {"messages": [S(""), S("Be brief."), U("hi")]},
    "system then empty": {"messages": [S("Be brief."), S(""), U("hi")]},
    "system blocks then system": {"messages": [{"role": "system", "content": [
        {"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}, S("C"), U("hi")]},
    "system after user": {"messages": [S("a"), U("hi"), S("late")]},
    "system, user, assistant, user": {"messages": [S("Be brief."), U("hi"), A("hello"), U("again")]},
    "two systems, long chat": {"messages": [S("a"), S("b"), U("hi"), A("hello"), U("again")]},
    "tools + system": {"messages": [S("Be brief."), U("weather?")], "tools": TOOLS},
    "tools + two systems": {"messages": [S("a"), S("b"), U("weather?")], "tools": TOOLS},
    "tool round trip": {"messages": [S("Be brief."), U("weather?"), CALL, TOOL]},
    "high effort": {"messages": [S("Be brief."), U("hi")], "reasoning_effort": "high", "add_generation_prompt": True},
    "none effort": {"messages": [S("Be brief."), U("hi")], "reasoning_effort": "none", "add_generation_prompt": True},
    "high effort, two systems": {"messages": [S("a"), S("b"), U("hi")], "reasoning_effort": "high", "add_generation_prompt": True},
    "high, assistant last": {"messages": [U("hi"), A("Hel")], "reasoning_effort": "high", "add_generation_prompt": True},
}
bad = 0
for name, case in CASES.items():
    out = []
    for path in sys.argv[1:3]:
        try:
            out.append(("ok", render(path, case)))
        except Exception as exc:
            out.append(("error", str(exc)[:70]))
    (o_kind, o), (n_kind, n) = out
    if o_kind == "ok":
        verdict = "identical" if (n_kind == "ok" and n == o) else "DIFFERENT"
        bad += verdict != "identical"
    else:
        verdict = f"old errors ({o}); new {n_kind}"
    print(f"{name:34} {verdict}")
    if o_kind == "error" and n_kind == "ok":
        start = n.find("[SYSTEM_PROMPT]")
        print("    ->", json.dumps(n[start:n.find("[/SYSTEM_PROMPT]") + 16]))
sys.exit(1 if bad else 0)
