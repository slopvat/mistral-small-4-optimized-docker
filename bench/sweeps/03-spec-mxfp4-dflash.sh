#!/bin/sh
# Usage: 03-spec-mxfp4-dflash.sh
#
# Third round of speculative decoding tuning on MXFP4_MOE: DFlash lengths,
# DFlash combined with ngram-mod and Q8_0 drafters. Run as root on the GPU
# host from the project folder, after "convert-drafters".
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
DF=drafters-gguf/dflash
DS=drafters-gguf/dspark

$B --draft $DF-bf16.gguf spec-mxfp4-dflash-n2 $M -- \
    $C --spec-type draft-dflash --spec-draft-n-max 2
$B --draft $DF-bf16.gguf spec-mxfp4-dflash-n3-ngram-mod $M -- \
    $C --spec-type draft-dflash,ngram-mod --spec-draft-n-max 3
$B --draft $DF-bf16.gguf spec-mxfp4-dflash-n4-ngram-mod $M -- \
    $C --spec-type draft-dflash,ngram-mod --spec-draft-n-max 4
$B --draft $DF-q8_0.gguf spec-mxfp4-dflash-q8-n3-ngram-mod $M -- \
    $C --spec-type draft-dflash,ngram-mod --spec-draft-n-max 3
$B --draft $DS-q8_0.gguf spec-mxfp4-dspark-q8-n3-ngram-mod $M -- \
    $C --spec-type draft-dspark,ngram-mod --spec-draft-n-max 3
