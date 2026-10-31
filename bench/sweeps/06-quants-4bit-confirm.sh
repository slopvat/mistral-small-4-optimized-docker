#!/bin/sh
# Usage: 06-quants-4bit-confirm.sh
#
# Repeat the speculative decoding runs of 05-quants-4bit.sh with a bigger
# sample (12 prompts of up to 512 tokens, three counted passes), because the
# quants generate different text under greedy decoding and the draft
# acceptance rate, and with it the speed, depends on the text. Run as root on
# the GPU host from the project folder.
#
# AI Disclosure:
# - Authorship: Autonomous
# - Models: claude-sonnet-5-5 (Anthropic)
# - Scope: Wrote this sweep so the user's request to pick the best 4-bit quant
#   by speed rests on a sample large enough to be meaningful.

set -eu

B="python3 -I bench/bench.py --runs 4"
C="-c 32768"

for q in MXFP4_MOE UD-IQ4_XS UD-IQ4_NL UD-Q4_K_S UD-Q4_K_M UD-Q4_K_XL; do
    m=$(cd models/gguf && ls "$q"/*-00001-of-*.gguf)
    n=$(echo "$q" | tr 'A-Z_' 'a-z-')
    $B "c4-$n-baseline" "$m" -- $C
    $B --draft drafters-gguf/dspark-q8_0.gguf "c4-$n-dspark-n3" "$m" -- \
        $C --spec-type draft-dspark,ngram-mod --spec-draft-n-max 3
    $B --draft drafters-gguf/dspark-q8_0.gguf "c4-$n-dspark-n2" "$m" -- \
        $C --spec-type draft-dspark,ngram-mod --spec-draft-n-max 2
done
