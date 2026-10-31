#!/usr/bin/env python3
# Usage: fix-mistral3-drafter-vocab.py QWEN_PY
#
# Patch llama.cpp v0.8.0's conversion/qwen.py in place so that DFlash and
# DSpark drafters can borrow the vocabulary of a Mistral Small 4 target.
# Mistral3Model.set_vocab() forwards to an "impl" object that a drafter
# instance does not have, so the conversion dies with an AttributeError.
# Borrowing Mistral4Model.set_vocab() instead is what the forwarding would do.
#
# AI Disclosure:
# - Authorship: Autonomous
# - Models: claude-sonnet-5-5 (Anthropic)
# - Scope: Wrote this patch so the RedHat AI Mistral Small 4 drafters can be
#   converted to GGUF for the user's request for optimized Mistral Small 4
#   images.
# - References: https://github.com/ggml-org/llama.cpp (v0.8.0,
#   conversion/qwen.py and conversion/mistral3.py)

import sys

path = sys.argv[1]
text = open(path).read()
old = "            target_cls.set_vocab(self)  # ty: ignore[unresolved-attribute]\n"
new = (
    "            if target_arch == \"Mistral3ForConditionalGeneration\":\n"
    "                from .mistral3 import Mistral3Model\n"
    "                target_cls = Mistral3Model.Mistral4Model\n"
    + old
)
if text.count(old) != 1:
    sys.exit("fix-mistral3-drafter-vocab: unexpected conversion/qwen.py")
open(path, "w").write(text.replace(old, new))
