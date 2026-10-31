#!/bin/sh
# Usage: 05-quants-4bit.sh
#
# Benchmark every 4-bit quant on GPU 0 without speculative decoding and with
# the three best settings found on MXFP4_MOE (sweeps 01 to 04): DSpark and
# DFlash Q8_0 drafters, each combined with ngram-mod. Run as root on the GPU
# host from the project folder, after "convert-drafters".
#
# AI Disclosure:
# - Authorship: Autonomous
# - Models: claude-sonnet-5-5 (Anthropic)
# - Scope: Wrote this sweep so the user's request to pick the best 4-bit quant
#   by speed can be answered with measurements.

set -eu

B="python3 -I bench/bench.py"
C="-c 32768"

for q in MXFP4_MOE UD-IQ4_XS UD-IQ4_NL UD-Q4_K_S UD-Q4_K_M UD-Q4_K_XL; do
    m=$(cd models/gguf && ls "$q"/*-00001-of-*.gguf)
    n=$(echo "$q" | tr 'A-Z_' 'a-z-')
    $B "q4-$n-baseline" "$m" -- $C
    $B --draft drafters-gguf/dspark-q8_0.gguf "q4-$n-dspark-n3" "$m" -- \
        $C --spec-type draft-dspark,ngram-mod --spec-draft-n-max 3
    $B --draft drafters-gguf/dspark-q8_0.gguf "q4-$n-dspark-n2" "$m" -- \
        $C --spec-type draft-dspark,ngram-mod --spec-draft-n-max 2
    $B --draft drafters-gguf/dflash-q8_0.gguf "q4-$n-dflash-n3" "$m" -- \
        $C --spec-type draft-dflash,ngram-mod --spec-draft-n-max 3
done
