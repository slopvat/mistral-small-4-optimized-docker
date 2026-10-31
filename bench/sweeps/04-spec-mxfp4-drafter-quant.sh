#!/bin/sh
# Usage: 04-spec-mxfp4-drafter-quant.sh
#
# Fourth round of speculative decoding tuning on MXFP4_MOE: a 4-bit drafter
# and the draft length around the best Q8_0 DSpark setting. Run as root on the
# GPU host from the project folder.
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
DS=drafters-gguf/dspark

docker run --rm --device nvidia.com/gpu=0 --entrypoint llama-quantize \
    -v "$PWD/models:/models" msl4-llama-tools \
    /models/$DS-bf16.gguf /models/$DS-q4_0.gguf Q4_0 2>&1 | tail -1

for n in 2 4; do
    $B --draft $DS-q8_0.gguf "spec-mxfp4-dspark-q8-n$n-ngram-mod" $M -- \
        $C --spec-type draft-dspark,ngram-mod --spec-draft-n-max "$n"
done
$B --draft $DS-q4_0.gguf spec-mxfp4-dspark-q4-n3-ngram-mod $M -- \
    $C --spec-type draft-dspark,ngram-mod --spec-draft-n-max 3
