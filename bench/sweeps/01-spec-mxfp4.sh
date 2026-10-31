#!/bin/sh
# Usage: 01-spec-mxfp4.sh
#
# Compare speculative decoding modes on the MXFP4_MOE quant on GPU 0. Run as
# root on the GPU host from the project folder.
#
# AI Disclosure:
# - Authorship: Autonomous
# - Models: claude-sonnet-5-5 (Anthropic)
# - Scope: Wrote this sweep to pick the speculative decoding settings for the
#   user's request for optimized Mistral Small 4 images.

set -eu

M=MXFP4_MOE/Mistral-Small-4-119B-2603-MXFP4_MOE-00001-of-00003.gguf
B="python3 -I bench/bench.py"
C="-c 32768"

for n in 3 5 7; do
    $B --draft drafters-gguf/dspark-bf16.gguf "spec-mxfp4-dspark-n$n" $M -- \
        $C --spec-type draft-dspark --spec-draft-n-max "$n"
done
$B --draft drafters-gguf/dflash-bf16.gguf spec-mxfp4-dflash-n7 $M -- \
    $C --spec-type draft-dflash --spec-draft-n-max 7
$B spec-mxfp4-ngram-mod $M -- $C --spec-type ngram-mod
$B spec-mxfp4-ngram-map-k $M -- $C --spec-type ngram-map-k --spec-draft-n-max 64
