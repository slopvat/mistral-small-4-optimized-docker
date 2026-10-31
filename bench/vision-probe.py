#!/usr/bin/env python3
"""
Usage: vision-probe.py URL [SIZE]

Ask the server at URL which colors are on the left and right halves of a PNG
of SIZE x SIZE pixels (default: 64) whose left half is red and right half is
green, and print the prompt token count and the answer.

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote this helper for bench/vision-matrix.sh while finding out why
  the images answered image questions wrongly.
"""
import base64
import json
import struct
import sys
import urllib.request
import zlib


def png(w, h):
    def chunk(kind, data):
        body = kind + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body)))
    row = b"\x00" + b"\xff\x00\x00" * (w // 2) + b"\x00\xc0\x00" * (w - w // 2)
    rows = row * h
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


size = int(sys.argv[2]) if len(sys.argv) > 2 else 64
url = "data:image/png;base64," + base64.b64encode(png(size, size)).decode()
body = {"messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": url}},
    {"type": "text", "text": "This image has a left half and a right half. What color is each half? Answer in the form: left=COLOR, right=COLOR."}]}],
    "max_tokens": 30, "temperature": 0, "reasoning_effort": "none"}
req = urllib.request.Request(sys.argv[1] + "/v1/chat/completions",
                             json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
reply = json.load(urllib.request.urlopen(req, timeout=300))
print(size, reply["usage"]["prompt_tokens"],
      repr(reply["choices"][0]["message"]["content"]))
