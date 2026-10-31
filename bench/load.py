#!/usr/bin/env python3
"""
Usage: load.py --url URL [--users N] [--rounds N] NAME

Send the prompts of bench.py from N concurrent users (each user sends the
prompts in turn, starting at a different one) and write the aggregate output
speed to results/NAME.json. The server has to be started with at least N
slots (PARALLEL=N for the images). "1 user" is the token-weighted speed of
one stream; the other figures are total output tokens per second across all
users.

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote the load tester to measure how the images behave with several
  concurrent users, for the user's request for optimized Mistral Small 4
  images.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import MAX_TOKENS, PROMPTS, post  # noqa: E402


def main():
    ap = argparse.ArgumentParser(usage=__doc__.split("\n\n")[0].strip()[7:])
    ap.add_argument("--url", required=True)
    ap.add_argument("--users", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("name")
    args = ap.parse_args()

    tokens = []
    lock = threading.Lock()

    def user(i):
        for r in range(args.rounds * len(PROMPTS)):
            _, prompt = PROMPTS[(i + r) % len(PROMPTS)]
            data = post(args.url.rstrip("/"), {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_TOKENS, "temperature": 0,
                "reasoning_effort": "none", "cache_prompt": False})
            with lock:
                tokens.append(data["usage"]["completion_tokens"])

    # Warm-up request so the first measurement does not include lazy setup.
    post(args.url.rstrip("/"), {
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 8, "temperature": 0, "reasoning_effort": "none"})
    start = time.time()
    threads = [threading.Thread(target=user, args=(i,))
               for i in range(args.users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - start
    result = {"name": args.name, "users": args.users,
              "requests": len(tokens), "tokens": sum(tokens),
              "seconds": round(elapsed, 1),
              "total_tok_s": round(sum(tokens) / elapsed, 1),
              "per_user_tok_s": round(sum(tokens) / elapsed / args.users, 1)}
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    with open(os.path.join(here, "results", args.name + ".json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"{args.name}: {result['users']} users, {result['total_tok_s']} "
          f"tok/s total, {result['per_user_tok_s']} per user")


if __name__ == "__main__":
    main()
