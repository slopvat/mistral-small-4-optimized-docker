#!/bin/sh
# Usage: 07-q8-split.sh
#
# Tune UD-Q8_K_XL, which needs both GPUs: the split between them, then
# speculative decoding settings. Layer split is the only multi-GPU mode that
# works for this architecture (llama.cpp denies "-sm tensor" for it). GPU 0
# is the faster of the two cards. Run as root on the GPU host from the
# project folder.
#
# AI Disclosure:
# - Authorship: Autonomous
# - Models: claude-sonnet-5-5 (Anthropic)
# - Scope: Wrote this sweep to tune the 8-bit image for the user's request
#   for optimized Mistral Small 4 images.

set -eu

M=UD-Q8_K_XL/Mistral-Small-4-119B-2603-UD-Q8_K_XL-00001-of-00004.gguf
B="python3 -I bench/bench.py --gpus 0,1 --runs 3"
C="-c 32768"
DS=drafters-gguf/dspark-q8_0.gguf

for ts in 1,1 55,45 60,40; do
    $B "q8-ts$ts-baseline" $M -- $C --tensor-split "$ts"
done
for n in 2 3 4 5; do
    $B --draft $DS "q8-ts1,1-dspark-n$n" $M -- $C --tensor-split 1,1 \
        --spec-type draft-dspark,ngram-mod --spec-draft-n-max "$n"
done
