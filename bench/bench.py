#!/usr/bin/env python3
"""
Usage: bench.py [--gpus IDS] [--draft GGUF] [--runs N] NAME MODEL [-- SERVER_ARG...]
       bench.py [--runs N] --url URL NAME

Start a llama-server container ("msl4-llama-tools" image) on the given GPUs,
send a fixed set of greedy chat prompts and write the token-weighted decode
speed to results/NAME.json. Run it as root on the GPU host, from a checkout
of this project, with the models in ../models relative to this folder or in
the folder given by the MODELS environment variable. MODEL is the path of the
first GGUF shard relative to the models folder's "gguf" directory.

Options:
  --gpus IDS    Comma-separated GPU indexes for the container (default: 0).
  --draft GGUF  Drafter path relative to the models folder; adds it with
                "-md". Use SERVER_ARG for the "--spec-type" and other flags.
  --url URL     Benchmark the already running server at URL (e.g. one started
                from an image) instead of starting a container; MODEL and
                SERVER_ARG are not used.
  --runs N      Passes over the 12 prompts (default: 2); the first pass of
                each server is a warm-up that is not counted.

AI Disclosure:
- Authorship: Autonomous
- Models: claude-sonnet-5-5 (Anthropic)
- Scope: Wrote the benchmark harness to measure and compare the speed of the
  Mistral Small 4 quants and speculative decoding settings for the user's
  request for optimized images.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

PROMPTS = [
    ("code", "Write a Python module implementing an LRU cache with a TTL, "
             "type hints, docstrings and a pytest test suite."),
    ("code-edit", "Here is a function:\n\ndef mean(xs):\n    total = 0\n"
                  "    for x in xs:\n        total += x\n    return total / len(xs)\n\n"
                  "Rewrite it to handle empty input, accept any iterable and "
                  "add type hints. Show the full function again, then explain."),
    ("sql", "Design a PostgreSQL schema for a library lending system and "
            "write queries for overdue books, most borrowed titles and "
            "per-member fines."),
    ("prose", "Write a detailed essay on how the printing press changed "
              "European society."),
    ("story", "Write a short story about a lighthouse keeper who finds a "
              "message in a bottle."),
    ("explain", "Explain how TCP congestion control works, including slow "
                "start, congestion avoidance and fast recovery."),
    ("summary", "Summarize the causes, course and consequences of the French "
                "Revolution in about 400 words."),
    ("translation", "Translate into French, German and Spanish: 'The meeting "
                    "has been moved to Thursday afternoon because the "
                    "conference room is being repainted.' Then explain the "
                    "grammar of each translation."),
    ("qa", "What are the main differences between a mutex, a semaphore and a "
           "condition variable? Give an example of each in C++."),
    ("math", "A train leaves at 9:15 travelling at 80 km/h. A second train "
             "leaves the same station at 10:00 at 100 km/h on the same track. "
             "When does the second catch up? Show every step."),
    ("json", "Produce a JSON array of 12 fictional products, each with id, "
             "name, price, tags (array) and a nested dimensions object."),
    ("list", "List 30 programming language features that were introduced in "
             "the last decade, with one sentence on each."),
]
MAX_TOKENS = 512


def post(base, body):
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def main():
    ap = argparse.ArgumentParser(usage=__doc__.split("\n\n")[0].strip()[7:])
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--draft")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--url")
    ap.add_argument("name")
    ap.add_argument("model", nargs="?")
    ap.add_argument("server_args", nargs="*")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    models = os.environ.get("MODELS", os.path.join(here, "..", "models"))
    models = os.path.abspath(models)
    results = os.path.join(here, "results")
    os.makedirs(results, exist_ok=True)

    base = args.url.rstrip("/") if args.url else \
        f"http://127.0.0.1:{args.port}"
    cname = None
    if not args.url:
        cname = "msl4-bench-" + re.sub(r"[^A-Za-z0-9_.-]", "-", args.name)
        subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
        cmd = ["docker", "run", "-d", "--name", cname,
               "-p", f"{args.port}:8080", "-v", f"{models}:/models:ro",
               "--entrypoint", "llama-server"]
        for gpu in args.gpus.split(","):
            cmd += ["--device", f"nvidia.com/gpu={gpu}"]
        cmd += ["msl4-llama-tools", "-m", f"/models/gguf/{args.model}",
                "--host", "0.0.0.0", "--port", "8080", "-ngl", "99",
                "-fa", "on", "--jinja", "--parallel", "1"]
        if args.draft:
            cmd += ["-md", f"/models/{args.draft}", "-ngld", "99"]
        cmd += args.server_args
        subprocess.run(cmd, check=True, capture_output=True)

    try:
        start = time.time()
        while True:
            try:
                urllib.request.urlopen(f"{base}/health", timeout=2)
                break
            except Exception:
                state = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", cname],
                    capture_output=True, text=True).stdout.strip() \
                    if cname else "true"
                if state != "true":
                    log = subprocess.run(["docker", "logs", "--tail", "30", cname],
                                         capture_output=True, text=True)
                    sys.exit("server exited:\n" + log.stdout + log.stderr)
                if time.time() - start > 1800:
                    sys.exit("server did not become healthy")
                time.sleep(2)
        load_s = time.time() - start

        rows = []
        for run in range(args.runs):
            for pname, prompt in PROMPTS:
                data = post(base, {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": MAX_TOKENS, "temperature": 0,
                    "reasoning_effort": "none", "cache_prompt": False})
                t = data["timings"]
                rows.append({
                    "run": run, "prompt": pname,
                    "predicted_n": t["predicted_n"],
                    "predicted_ms": t["predicted_ms"],
                    "tok_s": t["predicted_per_second"],
                    "draft_n": t.get("draft_n", 0),
                    "draft_accepted_n": t.get("draft_n_accepted", 0)})
        counted = [r for r in rows if r["run"] > 0] or rows
        n = sum(r["predicted_n"] for r in counted)
        ms = sum(r["predicted_ms"] for r in counted)
        drafted = sum(r["draft_n"] for r in counted)
        accepted = sum(r["draft_accepted_n"] for r in counted)
        summary = {
            "name": args.name, "model": args.model, "gpus": args.gpus,
            "draft": args.draft, "server_args": args.server_args,
            "load_s": round(load_s, 1), "max_tokens": MAX_TOKENS,
            "n_prompts": len(PROMPTS),
            "tok_s": round(n / ms * 1000, 1),
            "accept_rate": round(accepted / drafted, 3) if drafted else None,
            "by_prompt": {r["prompt"]: round(r["tok_s"], 1)
                          for r in counted},
            "rows": rows}
        with open(os.path.join(results, args.name + ".json"), "w") as f:
            json.dump(summary, f, indent=1)
        print(f"{args.name}: {summary['tok_s']} tok/s, accept "
              f"{summary['accept_rate']}, load {summary['load_s']} s")
        print("  " + "  ".join(f"{k}={v}" for k, v in summary["by_prompt"].items()))
    finally:
        if cname:
            log = subprocess.run(["docker", "logs", cname],
                                 capture_output=True, text=True)
            with open(os.path.join(results, args.name + ".log"), "w") as f:
                f.write(log.stdout + log.stderr)
            subprocess.run(["docker", "rm", "-f", cname], capture_output=True)


if __name__ == "__main__":
    main()
