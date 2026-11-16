# syntax=docker/dockerfile:1

# Mistral Small 4 119B served with llama.cpp on NVIDIA RTX PRO 6000 Blackwell
# GPUs. Targets:
#   q4     UD-Q4_K_XL (4-bit) on one GPU: the fastest quant that loses the
#          least quality; see README.md for the benchmarks.
#   q8     UD-Q8_K_XL (8-bit) split across two GPUs.
#   tools  llama.cpp with the model converter, for the benchmarks.
# Both images use the DSpark speculative decoding drafter, converted to GGUF
# and quantized to 8 bits while the image is built.
#
# AI Disclosure:
# - Authorship: Autonomous
# - Models: claude-sonnet-5-5 (Anthropic)
# - Scope: Wrote the image definition so the user's request for optimized
#   Mistral Small 4 images has a reproducible build with pinned inputs.
# - References:
#   - https://github.com/ggml-org/llama.cpp (v0.8.0, .devops/cuda.Dockerfile)
#   - https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF
#   - https://huggingface.co/RedHatAI/Mistral-Small-4-119B-2603-speculator.dspark
#
# The models are downloaded during the build and baked into the images, so a
# built image needs no network access. The revisions are the ones that were
# benchmarked. A Hugging Face token is optional; pass it as the "hf_token"
# build secret (e.g. "docker build --secret id=hf_token,env=HF_TOKEN ...") so
# it is not stored in the image.

ARG CUDA_VERSION=13.4.3
ARG UBUNTU_VERSION=24.04

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libssl-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG LLAMA_CPP_TAG=v0.8.0
RUN git clone --depth 1 --branch "$LLAMA_CPP_TAG" \
        https://github.com/ggml-org/llama.cpp /src \
    && git -C /src rev-parse HEAD > /src/COMMIT

# Mistral Small 4 does not record layer inputs, which speculative decoding with
# the DFlash and DSpark drafters needs.
COPY patches/deepseek2-layer-inputs.patch /tmp/
WORKDIR /src
RUN patch -p1 < /tmp/deepseek2-layer-inputs.patch

# Only the Blackwell workstation architecture is built to keep the build
# short; the images are not meant for other GPUs.
WORKDIR /src
RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=120a-real -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release -j"$(nproc)" \
    && mkdir -p /out/bin /out/lib \
    && cp build/bin/llama-* /out/bin/ \
    && find build -name '*.so*' -exec cp -P {} /out/lib/ \;

# Replies that end inside the thinking block: the model sometimes emits its
# end-of-sequence token before "[/THINK]", and the server then files the whole
# answer under reasoning. This patch lets the server refuse that token while a
# thinking block is open ("--reasoning-block-eog"; the entrypoint turns it on).
# It is applied on top of the finished build, so only the files it touches are
# rebuilt and the stages below that use the converter and download the models
# keep their cache.
FROM build AS build-patched
COPY patches/reasoning-block-eog.patch /tmp/
WORKDIR /src
RUN patch -p1 < /tmp/reasoning-block-eog.patch \
    && cmake --build build --config Release -j"$(nproc)" \
    && mkdir -p /out-patched/bin /out-patched/lib \
    && cp build/bin/llama-server build/bin/llama-cli /out-patched/bin/ \
    && find build -name '*.so*' -exec cp -P {} /out-patched/lib/ \;

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime-base

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The "tools" target uses the build without the patch above.
FROM runtime-base AS runtime
COPY --from=build /out/lib/ /usr/local/lib/
COPY --from=build /out/bin/llama-server /out/bin/llama-cli /usr/local/bin/
COPY --from=build /src/COMMIT /usr/local/share/llama.cpp-commit
RUN ldconfig

# Converter, benchmark tools and Python for the drafter conversion.
FROM runtime AS tools

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/bin/ /usr/local/bin/
COPY --from=build /src/ /opt/llama.cpp-src/
# Mistral Small 4's tokenizer config names a transformers v5 class, so v5 is
# installed over the converter's pinned version. The patch makes the drafters'
# vocabulary loading work with a Mistral Small 4 target.
COPY patches/fix-mistral3-drafter-vocab.py /tmp/
RUN pip install --break-system-packages --no-cache-dir \
        -r /opt/llama.cpp-src/requirements/requirements-convert_hf_to_gguf.txt \
    && pip install --break-system-packages --no-cache-dir -U 'transformers>=5' \
    && python3 /tmp/fix-mistral3-drafter-vocab.py \
        /opt/llama.cpp-src/conversion/qwen.py \
    && rm /tmp/fix-mistral3-drafter-vocab.py

# The DSpark drafter, converted to an 8-bit GGUF. 4-bit drafters accept almost
# nothing (3% in the benchmarks) and BF16 ones are slower than 8-bit.
FROM tools AS drafter

ARG DRAFTER_REVISION=1dcfc8a0c2cef296881de0b0752f383ae6d2a07f
ARG BASE_REVISION=a11f36bebf709121056b1dbcc943d1c6afbe494d
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    HF_HOME=/tmp/hf-home HF_XET_HIGH_PERFORMANCE=1 sh -ec ' \
        hf download RedHatAI/Mistral-Small-4-119B-2603-speculator.dspark \
            --revision "$DRAFTER_REVISION" --local-dir /work/dspark \
            --include "*.json" --include "model.safetensors"; \
        hf download mistralai/Mistral-Small-4-119B-2603 \
            --revision "$BASE_REVISION" --local-dir /work/base \
            --include "*.json" --include "*.jinja"; \
        mkdir -p /out; \
        python3 /opt/llama.cpp-src/convert_hf_to_gguf.py /work/dspark \
            --target-model-dir /work/base --outtype q8_0 \
            --outfile /out/dspark-q8_0.gguf'

# Each model is downloaded in its own stage so the two images share nothing
# but the drafter.
FROM tools AS gguf-q4
ARG GGUF_REVISION=bd93c721735aa32c035c0f19e738cb3371fd56ff
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    HF_HOME=/tmp/hf-home HF_XET_HIGH_PERFORMANCE=1 \
        hf download unsloth/Mistral-Small-4-119B-2603-GGUF \
        --revision "$GGUF_REVISION" --local-dir /models \
        --include "UD-Q4_K_XL/*" --include "mmproj-BF16.gguf" \
    && rm -rf /models/.cache

FROM tools AS gguf-q8
ARG GGUF_REVISION=bd93c721735aa32c035c0f19e738cb3371fd56ff
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    HF_HOME=/tmp/hf-home HF_XET_HIGH_PERFORMANCE=1 \
        hf download unsloth/Mistral-Small-4-119B-2603-GGUF \
        --revision "$GGUF_REVISION" --local-dir /models \
        --include "UD-Q8_K_XL/*" --include "mmproj-BF16.gguf" \
    && rm -rf /models/.cache

# The models come before llama.cpp, the entrypoint script and the settings so
# that changing those only rebuilds the small last layers instead of copying
# the model again.
FROM runtime-base AS serve
COPY --from=drafter /out/dspark-q8_0.gguf /models/dspark-q8_0.gguf

FROM serve AS q4-models
COPY --from=gguf-q4 /models/ /models/

FROM serve AS q8-models
COPY --from=gguf-q8 /models/ /models/

FROM q4-models AS q4
COPY --from=build-patched /out-patched/lib/ /usr/local/lib/
COPY --from=build-patched /out-patched/bin/ /usr/local/bin/
COPY --from=build-patched /src/COMMIT /usr/local/share/llama.cpp-commit
RUN ldconfig
COPY mistral-serve /usr/local/bin/mistral-serve
COPY mistral-chat-template.jinja /usr/local/share/mistral-chat-template.jinja
# One GPU holds the whole model; layer-splitting it over two only adds hops.
ENV HF_HUB_OFFLINE=1 \
    DEFAULT_MODEL=/models/UD-Q4_K_XL/Mistral-Small-4-119B-2603-UD-Q4_K_XL-00001-of-00003.gguf \
    DEFAULT_SERVED_MODEL_NAME="mistral-small-4 unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q4_K_XL" \
    DEFAULT_CUDA_VISIBLE_DEVICES=0 \
    DEFAULT_SPEC_DRAFT_N_MAX=2
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30m \
    CMD curl -fs "http://127.0.0.1:${PORT:-8000}/health" || exit 1
ENTRYPOINT ["/usr/local/bin/mistral-serve"]

FROM q8-models AS q8
COPY --from=build-patched /out-patched/lib/ /usr/local/lib/
COPY --from=build-patched /out-patched/bin/ /usr/local/bin/
COPY --from=build-patched /src/COMMIT /usr/local/share/llama.cpp-commit
RUN ldconfig
COPY mistral-serve /usr/local/bin/mistral-serve
COPY mistral-chat-template.jinja /usr/local/share/mistral-chat-template.jinja
ENV HF_HUB_OFFLINE=1 \
    DEFAULT_MODEL=/models/UD-Q8_K_XL/Mistral-Small-4-119B-2603-UD-Q8_K_XL-00001-of-00004.gguf \
    DEFAULT_SERVED_MODEL_NAME="mistral-small-4 unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q8_K_XL" \
    DEFAULT_SPEC_DRAFT_N_MAX=3
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30m \
    CMD curl -fs "http://127.0.0.1:${PORT:-8000}/health" || exit 1
ENTRYPOINT ["/usr/local/bin/mistral-serve"]
