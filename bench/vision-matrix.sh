#!/bin/sh
# Usage: vision-matrix.sh
#
# Check image understanding across projector precisions and projector
# placement (GPU or CPU), on GPU 0 with the 4-bit quant. Prints the answer to
# the colors of the halves of a two-color image at several sizes (SIZES
# in the environment overrides them). Run as root
# on the GPU host from the project folder.
#
# AI Disclosure:
# - Authorship: Autonomous
# - Models: claude-sonnet-5-5 (Anthropic)
# - Scope: Wrote this script to find out why the images answered image
#   questions wrongly, for the user's request for optimized Mistral Small 4
#   images.

set -eu

M=/models/gguf/UD-Q4_K_XL/Mistral-Small-4-119B-2603-UD-Q4_K_XL-00001-of-00003.gguf

try() {
    label=$1 proj=$2
    shift 2
    docker rm -f msl4-vision >/dev/null 2>&1 || true
    docker run -d --name msl4-vision --device nvidia.com/gpu=0 -p 8092:8080 \
        -v "$PWD/models:/models:ro" --entrypoint llama-server \
        msl4-llama-tools -m "$M" --mmproj "/models/gguf/$proj" --host 0.0.0.0 \
        --port 8080 -ngl 99 -fa on --jinja -c 16384 --parallel 1 "$@" >/dev/null
    until curl -sf http://127.0.0.1:8092/health >/dev/null; do sleep 2; done
    echo "== $label"
    for s in ${SIZES:-224 336 384 448 480 512 576 640 768 896}; do
        python3 -I bench/vision-probe.py http://127.0.0.1:8092 "$s" | cut -c1-200
    done
}

try "bf16 projector on the GPU" mmproj-BF16.gguf
try "bf16 projector on the CPU" mmproj-BF16.gguf --no-mmproj-offload
docker rm -f msl4-vision >/dev/null
