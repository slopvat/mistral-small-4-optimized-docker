#!/bin/sh
# Usage: 02-spec-mxfp4-refine.sh
#
# Refine the speculative decoding settings found by 01-spec-mxfp4.sh: shorter
# drafts, a draft probability cutoff and combining a drafter with ngram-mod.
# Run as root on the GPU host from the project folder.
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
DS="--draft drafters-gguf/dspark-bf16.gguf"

for n in 2 4; do
    $B $DS "spec-mxfp4-dspark-n$n" $M -- \
        $C --spec-type draft-dspark --spec-draft-n-max "$n"
done
for p in 0.5 0.7; do
    $B $DS "spec-mxfp4-dspark-n7-pmin$p" $M -- \
        $C --spec-type draft-dspark --spec-draft-n-max 7 --spec-draft-p-min "$p"
done
$B $DS spec-mxfp4-dspark-n3-ngram-mod $M -- \
    $C --spec-type draft-dspark,ngram-mod --spec-draft-n-max 3
$B --draft drafters-gguf/dflash-bf16.gguf spec-mxfp4-dflash-n3 $M -- \
    $C --spec-type draft-dflash --spec-draft-n-max 3
