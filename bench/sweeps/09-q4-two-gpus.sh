#!/bin/sh
# Usage: 09-q4-two-gpus.sh
#
# Does the 4-bit model benefit from two GPUs? Compares it on one GPU with a
# layer split over both and with two replicas (one container per GPU) under
# the same concurrent load. Stops starting new tests 30 minutes after it
# started. Run as root on the GPU host from the project folder, with the
# "mistral-small-4-optimized:q4" image built.
#
# AI Disclosure:
# - Authorship: Autonomous
# - Models: claude-sonnet-5-5 (Anthropic)
# - Scope: Wrote this sweep to test, at the user's request, the README's
#   untested claims about using two GPUs with the 4-bit model.

set -eu

START=$(date +%s)
BUDGET=1800
M=UD-Q4_K_XL/Mistral-Small-4-119B-2603-UD-Q4_K_XL-00001-of-00003.gguf
IMG=mistral-small-4-optimized:q4
SPEC=draft-dspark,ngram-mod

within_budget()
{
    elapsed=$(( $(date +%s) - START ))
    if [ "$elapsed" -ge "$BUDGET" ]; then
        echo "budget exhausted after ${elapsed}s; skipping: $*"
        return 1
    fi
    echo "--- $* (at ${elapsed}s)"
}

start_server()
{
    name=$1 port=$2 gpus=$3 parallel=$4 spec=$5
    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run -d --name "$name" -p "$port:8000" --device nvidia.com/gpu=all \
        -e CUDA_VISIBLE_DEVICES="$gpus" -e PARALLEL="$parallel" \
        -e SPEC_TYPE="$spec" "$IMG" >/dev/null
}

wait_healthy()
{
    for port in "$@"; do
        until curl -sf "http://127.0.0.1:$port/health" >/dev/null; do
            sleep 2
        done
    done
}

# Aggregate of load.py results: all tokens over the longest elapsed time.
aggregate()
{
    label=$1
    shift
    python3 -I - "$label" "$@" <<'PY'
import json
import sys

label, names = sys.argv[1], sys.argv[2:]
runs = [json.load(open(f"bench/results/{n}.json")) for n in names]
total = sum(r["tokens"] for r in runs) / max(r["seconds"] for r in runs)
print(f"{label}: {total:.1f} tok/s total")
PY
}

B="python3 -I bench/bench.py --gpus 0,1 --runs 4"
if within_budget "split over both GPUs, speculation"; then
    $B tg-q4-split-spec "$M" -- -c 32768 --tensor-split 1,1 \
        --spec-type "$SPEC" --spec-draft-n-max 2 \
        -md /models/drafters-gguf/dspark-q8_0.gguf -ngld 99 | head -1
fi
if within_budget "split over both GPUs, plain"; then
    $B tg-q4-split-plain "$M" -- -c 32768 --tensor-split 1,1 | head -1
fi

L="python3 -I bench/load.py --rounds 1"
for spec in $SPEC none; do
    label=$([ "$spec" = none ] && echo plain || echo spec)
    users=$([ "$spec" = none ] && echo 4 || echo 1)
    if within_budget "one GPU, $label, $((users * 2)) users"; then
        start_server tg-one 8091 0 $((users * 2)) "$spec"
        wait_healthy 8091
        $L --url http://127.0.0.1:8091 --users $((users * 2)) \
            "tg-one-gpu-$label-$((users * 2))-users"
    fi
    if within_budget "two replicas, $label, $users users each"; then
        docker rm -f tg-one >/dev/null 2>&1 || true
        start_server tg-a 8091 0 "$users" "$spec"
        start_server tg-b 8092 1 "$users" "$spec"
        wait_healthy 8091 8092
        $L --url http://127.0.0.1:8091 --users "$users" \
            "tg-replica0-$label" &
        $L --url http://127.0.0.1:8092 --users "$users" \
            "tg-replica1-$label" &
        wait
        aggregate "two replicas, $label" "tg-replica0-$label" \
            "tg-replica1-$label"
        docker rm -f tg-a tg-b >/dev/null
    fi
done
docker rm -f tg-one tg-a tg-b >/dev/null 2>&1 || true
echo "finished after $(( $(date +%s) - START ))s"
