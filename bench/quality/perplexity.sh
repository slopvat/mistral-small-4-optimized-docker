#!/bin/sh
# Usage: perplexity.sh
#
# Measure WikiText-2 test perplexity of every 4-bit quant (and the 8-bit one)
# with llama-perplexity, two quants at a time, one per GPU. Run as root on the
# GPU host from the project folder, after bench/quality/prepare-wikitext.
# Results go to bench/results/ppl-QUANT.txt. Lower is better; it is only
# meaningful for comparing the quants of this one model.
#
# AI Disclosure:
# - Authorship: Autonomous
# - Models: claude-sonnet-5-5 (Anthropic)
# - Scope: Wrote this script to rank the quants by quality for the user's
#   request to pick the best 4-bit quant from the fastest cluster.

set -eu

mkdir -p bench/results

run() {
    gpu=$1 q=$2
    m=$(cd models/gguf && ls "$q"/*-00001-of-*.gguf)
    docker run --rm --device "nvidia.com/gpu=$gpu" \
        -v "$PWD/models:/models:ro" --entrypoint llama-perplexity \
        msl4-llama-tools -m "/models/gguf/$m" -f /models/data/wiki.test.raw \
        -ngl 99 -fa on -c 512 > "bench/results/ppl-$q.txt" 2>&1
    echo "$q: $(grep -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' \
        "bench/results/ppl-$q.txt")"
}

set -- UD-Q4_K_XL UD-Q4_K_S UD-IQ4_XS UD-IQ4_NL MXFP4_MOE UD-Q4_K_M UD-Q8_K_XL
while [ $# -gt 0 ]; do
    run 0 "$1" &
    if [ $# -gt 1 ]; then
        run 1 "$2" &
    fi
    wait
    shift
    [ $# -gt 0 ] && shift
done
