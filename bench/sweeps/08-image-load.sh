#!/bin/sh
# Usage: 08-image-load.sh IMAGE_TAG [GPU_FLAGS]...
#
# Measure a built image ("q4" or "q8") with and without speculative decoding
# for 1, 4 and 8 concurrent users, the way it is shipped (full context, image
# input enabled). GPU_FLAGS are passed to "docker run" (default: all GPUs).
# Run as root on the GPU host from the project folder.
#
# AI Disclosure:
# - Authorship: Autonomous
# - Models: claude-sonnet-5-5 (Anthropic)
# - Scope: Wrote this sweep to measure the shipped images for the user's
#   request for optimized Mistral Small 4 images.

set -eu

tag=$1
shift
[ $# -gt 0 ] || set -- --device nvidia.com/gpu=all

for spec in draft-dspark,ngram-mod none; do
    for users in 1 4 8; do
        name="image-$tag-$users-users-spec-$spec"
        docker rm -f "msl4-image-$tag" >/dev/null 2>&1 || true
        docker run -d --name "msl4-image-$tag" -p 8091:8000 "$@" \
            -e PARALLEL="$users" -e SPEC_TYPE="$spec" \
            "mistral-small-4-optimized:$tag" >/dev/null
        until curl -sf http://127.0.0.1:8091/health >/dev/null; do
            sleep 3
        done
        python3 -I bench/load.py --url http://127.0.0.1:8091 \
            --users "$users" --rounds 1 "$name"
    done
done
docker rm -f "msl4-image-$tag" >/dev/null
