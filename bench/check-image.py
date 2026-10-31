#!/usr/bin/env python3
"""
Usage: check-image.py URL

Check that the server at URL answers a plain chat request, makes a tool call,
names the colors in an image and reasons when asked to. Exits with status 1 if any check
fails.

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote these checks to verify that the shipped images work end to end
  for the user's request for optimized Mistral Small 4 images.
"""
import base64
import json
import struct
import sys
import urllib.request
import zlib


def two_color_png(size=512):
    """Left half red, right half green. Tiny images confuse the model."""
    def chunk(kind, data):
        body = kind + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body)))
    half = size // 2
    row = b"\x00" + b"\xff\x00\x00" * half + b"\x00\xc0\x00" * (size - half)
    rows = row * size
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def chat(base, **body):
    body.setdefault("temperature", 0)
    req = urllib.request.Request(
        base + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)["choices"][0]["message"]


def main():
    base = sys.argv[1].rstrip("/")
    failed = 0

    def check(name, ok, detail):
        nonlocal failed
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
        failed += not ok

    msg = chat(base, messages=[{"role": "user", "content":
                                "What is 17 * 23? Reply with just the number."}],
               max_tokens=30, reasoning_effort="none")
    check("chat", "391" in (msg.get("content") or ""), repr(msg.get("content")))

    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the weather in a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}, "required": ["city"]}}}]
    msg = chat(base, messages=[{"role": "user", "content":
                                "What is the weather in Paris right now?"}],
               tools=tools, max_tokens=100, reasoning_effort="none")
    calls = msg.get("tool_calls") or []
    args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    check("tool call", bool(calls) and "paris" in str(args).lower(),
          json.dumps(calls)[:120])

    url = "data:image/png;base64," + base64.b64encode(two_color_png()).decode()
    msg = chat(base, messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "Which two colors are in this image? "
                                 "Answer with just the two color names."}]}],
        max_tokens=20, reasoning_effort="none")
    colors = (msg.get("content") or "").lower()
    check("image", "red" in colors and "green" in colors, repr(colors))

    msg = chat(base, messages=[{"role": "user", "content":
                                "If a train covers 150 km in 2.5 hours, what "
                                "is its average speed in km/h?"}],
               max_tokens=1500, reasoning_effort="high")
    text = (msg.get("content") or "")
    check("reasoning", "60" in text and bool(msg.get("reasoning_content")),
          f"reasoning_content={len(msg.get('reasoning_content') or '')} chars, "
          f"answer={text[:60]!r}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
