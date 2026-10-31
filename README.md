Mistral Small 4 Optimized Docker
================================

Docker images for serving [Mistral Small 4 119B][model] (a mixture of
experts model with 6.5B active parameters) as fast as possible on NVIDIA RTX
PRO 6000 Blackwell GPUs with llama.cpp, plus the benchmark harness and raw
results used to choose every setting.

There are two images:

- **q4:** The 4-bit `UD-Q4_K_XL` GGUF on **one** GPU. It is the fastest
  configuration found, and out of the 4-bit quants that were as fast as the
  fastest one, it has the best perplexity.
- **q8:** The 8-bit `UD-Q8_K_XL` GGUF split across **two** GPUs, for the
  best quality.

Both use llama.cpp v0.8.0 with speculative decoding: RedHat AI's DSpark
drafter (converted to GGUF and quantized to 8 bits) combined with llama.cpp's
`ngram-mod` drafting. Speculative decoding is lossless (the target model
verifies every drafted token), so it speeds the images up without changing
their output distribution. Image input is enabled and the context is the
model's full 262,144 tokens.

The models are downloaded while the images are built and baked into them at
pinned revisions (the ones benchmarked here). A built image has no remote
dependencies: both were started with networking disabled (`docker run
--network none`) and answered chat requests, and they pass the checks in
"bench/check-image.py" (chat, tool calls, image input, reasoning).

[model]: https://huggingface.co/mistralai/Mistral-Small-4-119B-2603

Results at a Glance
-------------------

Measured on 2x RTX PRO 6000 Blackwell Workstation Edition (96 GB each, PCIe
5.0 x16, no NVLink) using the images exactly as shipped. The 4-bit numbers
were measured on the faster of the two cards, which holds about 7% higher
clocks than the other under sustained load. "Decode" is the speed at which a
single stream generates tokens (the token-weighted average of 12 greedy
prompts of up to 512 tokens, three passes). "Total" is the output tokens per
second across all concurrent users, including prompt processing and
queueing, so a single user's total is lower than their decode speed.

| Image | Speculation | Decode, 1 user | Total, 1 user | Total, 4 users | Total, 8 users |
|---|---|--:|--:|--:|--:|
| q4 | on (default) | **258** | **191** | 194 | 190 |
| q4 | off | 163 | 153 | 295 | 365 |
| q8 | on (default) | 183 | 144 | 157 | 165 |
| q8 | off | 136 | 129 | 229 | 266 |

Speculative decoding makes a single stream 1.3x to 1.6x faster but **does
not scale across concurrent users**: with it on, total throughput stays flat
at 4 and 8 users, and with it off the same server serves them 1.5x to 1.9x
faster in total. Serve several users with `SPEC_TYPE=none PARALLEL=8`.

| Image | WikiText-2 perplexity (lower is better) | Model files | GPU memory |
|---|--:|--:|---|
| q4 | 5.7247 ± 0.037 | 70 GiB | 78 GiB on one GPU |
| q8 | 5.6555 ± 0.036 | 124 GiB | 71 GiB + 64 GiB on two GPUs |

The speed of plain (non-speculative) decoding is almost the same for every
quant (163 to 174 tokens/s on one GPU): the model does not run at the speed
of the GPU's memory, so a smaller quant does not decode faster. A 4-bit
model's advantage is that it fits on one GPU.

Quick Start
-----------

Build the images (this is the only step that needs network access):

    cp .env.example .env    # Optionally add a Hugging Face token.
    docker compose --profile q4 --profile q8 build

The q4 build downloads 70 GiB and the q8 build 124 GiB. The token in ".env" is only used during the
build and is passed as a build secret, so it never ends up in an image.

Then run one image at a time; both listen on port 8000:

    docker compose --profile q4 up -d
    # ...or...
    docker compose --profile q8 up -d

The server speaks the OpenAI API. Use the per-request `reasoning_effort`
parameter to choose between fast answers (`none`) and reasoning (`high`;
Mistral recommends temperature 0.7 for it, and 0 to 0.7 with `none`):

    curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Hello"}],"reasoning_effort":"none"}'

Without Compose (the GPU flag is required; without it the container exits
with "no GPU is available to the container"):

    docker build --target q4 --secret id=hf_token,env=HF_TOKEN \
        -t mistral-small-4-optimized:q4 .
    docker run -d --device nvidia.com/gpu=all -p 8000:8000 \
        mistral-small-4-optimized:q4

Use `--gpus all` instead of `--device nvidia.com/gpu=all` if Docker is set up
with the older NVIDIA runtime hook instead of CDI. The model name for API
requests is `mistral-small-4`.

Configuration
-------------

"mistral-serve" (the image's entrypoint) builds the `llama-server` command
line from environment variables. Arguments given to `docker run` are
appended to it, so any `llama-server` option works. Its header lists every
variable; the main ones are:

| Variable | Default | |
|---|---|---|
| `SPEC_TYPE` | `draft-dspark,ngram-mod` | `none` turns speculative decoding off. |
| `SPEC_DRAFT_N_MAX` | 2 (q4), 3 (q8) | Tokens drafted per step. |
| `PARALLEL` | 1 | Concurrent requests. Use `SPEC_TYPE=none` with more than 1. |
| `CONTEXT_SIZE` | 262144 | Total context, shared by all slots. |
| `MMPROJ` | the baked-in projector | `none` for text-only; saves about 1 GB. |
| `CUDA_VISIBLE_DEVICES` | GPU 0 (q4), all (q8) | |
| `TENSOR_SPLIT` | even | Proportions per GPU (q8). It made no difference to speed. |

To use both GPUs with the q4 image, run two containers, one per GPU
(`CUDA_VISIBLE_DEVICES=0` and `1`, different `PORT`s) and spread the
requests between them. This doubles the throughput; splitting one model
over two GPUs does not make it faster (see below).

How the Settings Were Chosen
----------------------------

Everything below can be regenerated from "bench/results" with
"bench/summarize.py"; "bench/sweeps" holds the scripts that produced it. The
first five sweeps (the `spec-` and `q4-` results) used 6 prompts of up to 400
tokens; everything after them (`c4-`, `q8-`, `image-` and `final-`) uses 12
prompts of up to 512 tokens.

### Speculative Decoding

Mistral publishes an EAGLE head for the model, but llama.cpp cannot use it:
it is an EAGLE-1/2 head in Mistral's own format and the converter stops with
`Can not map tensor 'eagle_linear.weight'` (llama.cpp only supports
EAGLE-3). RedHat AI trained [DFlash][dflash] and [DSpark][dspark] drafters
for this model, which llama.cpp v0.8.0 supports. On `MXFP4_MOE` on one GPU
(165 tokens/s without speculation):

| Setting | tok/s |
|---|--:|
| DSpark Q8_0, 3 tokens + `ngram-mod` | **227.1** |
| DFlash Q8_0, 3 tokens + `ngram-mod` | 225.9 |
| DSpark Q8_0, 2 tokens + `ngram-mod` | 220.5 |
| DSpark BF16, 3 tokens + `ngram-mod` | 212.3 |
| DFlash BF16, 3 tokens | 209.2 |
| DSpark BF16, 2 tokens | 204.7 |
| DSpark BF16, 3 tokens | 201.8 |
| `ngram-mod` alone | 190.0 |
| DSpark BF16, 7 tokens (the drafters' default) | 167.1 |
| DSpark Q4_0, 3 tokens + `ngram-mod` | 89.2 |

- **Short drafts win.** The drafters' model cards recommend 7 tokens, but
  every token past the second or third is accepted so rarely that verifying
  it costs more than it saves. A probability cutoff does not rescue longer
  drafts.
- **Combining a drafter with `ngram-mod`** (`--spec-type
  draft-dspark,ngram-mod`) beats either alone, because the n-gram drafter
  covers repetitive text (code, JSON, math steps) at no cost.
- **An 8-bit drafter is faster than BF16** (half the bytes read) with the
  same acceptance, but a 4-bit drafter is useless: only 3% of its tokens
  are accepted.
- DSpark and DFlash are within noise of each other, and DSpark was better
  with the 8-bit target, so DSpark is used.

[dflash]: https://huggingface.co/RedHatAI/Mistral-Small-4-119B-2603.dflash
[dspark]: https://huggingface.co/RedHatAI/Mistral-Small-4-119B-2603-speculator.dspark

### The 4-Bit Quant

Each Unsloth 4-bit quant was run on one GPU without speculation and with
the best settings above (12 prompts, three counted passes). The quants
write different text under greedy decoding, and how much of it the drafter
predicts changes the speed, so the speculative numbers carry noise of up to
about 10%; the plain numbers are clean. Perplexity is measured on the
WikiText-2 test split (context 512).

| Quant | Plain tok/s | Best speculative tok/s | Draft tokens | Accepted | Perplexity | Within 10% of the fastest |
|---|--:|--:|--:|--:|--:|--|
| UD-Q4_K_XL | 163.0 | 260.9 | 2 | 0.585 | 5.7247 ± 0.037 | yes |
| UD-Q4_K_S | 167.8 | 260.8 | 2 | 0.585 | 5.7285 ± 0.037 | yes |
| UD-IQ4_XS | 173.6 | 258.4 | 2 | 0.565 | 5.8767 ± 0.038 | yes |
| UD-IQ4_NL | 173.5 | 242.1 | 3 | 0.519 | 5.8772 ± 0.038 | yes |
| MXFP4_MOE | 164.5 | 226.0 | 2 | 0.524 | 5.9429 ± 0.039 | no |
| UD-Q4_K_M | 165.3 | 219.2 | 2 | 0.588 | 5.7283 ± 0.037 | no |

The fastest cluster (within 10% of the fastest) is UD-Q4_K_XL, UD-Q4_K_S,
UD-IQ4_XS and UD-IQ4_NL. UD-Q4_K_XL has the lowest perplexity in it, but
its difference from UD-Q4_K_S is far inside the error bars (so is
UD-Q4_K_M's, which is slower and so not in the cluster). UD-Q4_K_XL has the most bits, so it is used.
The UD-IQ4 quants lose about 0.15 perplexity, and MXFP4_MOE loses 0.22 and is
slower. Judged by plain decoding, all six are within 7% of each other, and
UD-Q4_K_XL is still the best of them.

The shipped q4 image, measured as a running server (full context, image
input enabled), decodes at 257.7 tokens/s.

### Two GPUs With the 4-Bit Model

Measured with "bench/sweeps/09-q4-two-gpus.sh" (11 minutes). Layer-splitting
the model over both GPUs (llama.cpp does not allow tensor parallelism for
this architecture) gives no speedup for a single stream, and two replicas
serve 1.6x (plain) to 1.9x (speculation) the throughput of one GPU for the
same number of users:

| Setup | Load | Total tok/s |
|---|---|--:|
| One GPU, speculation | 2 users | 199.4 |
| **Two replicas, speculation** | 1 user each | **380.4** |
| One GPU, plain | 8 users | 375.3 |
| **Two replicas, plain** | 4 users each | **592.3** |

For a single stream the split makes no difference (decode speed on the
benchmark prompts):

| Setup | Speculation | Plain |
|---|--:|--:|
| One GPU | 260.9 | 163.0 |
| Two GPUs, layer split | 256.9 | 163.5 |

The replica on the slower card (GPU 1) served within 2% of the other one
under these loads. A layer split runs the layers one after another, so
only one GPU works at a time (not measured); it is only worth it when the model does not fit on one GPU, as with
the 8-bit image.

### The 8-Bit Quant

UD-Q8_K_XL needs both GPUs. llama.cpp's tensor-parallel mode is disabled for
this architecture, so the layers are split between the GPUs. The split ratio
made no difference (136.1, 136.1 and 135.9 tokens/s for 1:1, 55:45 and
60:40), and the best speculative setting is the same DSpark Q8_0 drafter
with 3 tokens plus `ngram-mod` (176.9 tokens/s; 4 tokens is the same, 5
tokens is slower). DFlash was slower here (162.7). The shipped image
decodes at 182.5 tokens/s.

Changes to llama.cpp
--------------------

llama.cpp v0.8.0 does not run these drafters with this model out of the
box, so the build applies two small patches ("patches/"):

- **deepseek2-layer-inputs.patch:** Mistral Small 4 uses the DeepSeek-2
  graph, which did not record the layer inputs that DFlash and DSpark
  drafters read from the target model. Without it, the server aborts on the
  first request with `GGML_ASSERT(t_layer_inp[il] != nullptr)`. It is the
  same one-line change other architectures received in llama.cpp pull
  request [#30927](https://github.com/ggml-org/llama.cpp/pull/30927).
- **fix-mistral3-drafter-vocab.py:** The converter's code for borrowing the
  target's vocabulary calls a method that does not exist on drafters when
  the target is a Mistral 3 model. The patch uses the Mistral 4 method the
  call was meant to reach. Converting also needs transformers 5 for the
  model's tokenizer.

Known Problems
--------------

- **Image answers are not always stable.** Solid or tiny images confuse the
  model (a 64x64 image is answered wrongly), and in a test of ten image
  sizes with the vision projector on the GPU one 448x448 image had its two
  halves swapped, which did not happen with the projector on the CPU. Pass
  `--no-mmproj-offload` to run the projector on the CPU. See
  "bench/vision-matrix.sh".
- **llama.cpp issue [#30224][30224] does not apply to these images:** during
  generation one CPU thread is busy and the GPU is at 97% utilization, so
  the expert matrix multiplications are not running on the CPU.
- No end-to-end accuracy benchmark was run; perplexity is the only quality
  measure.
- The images target Blackwell workstation GPUs (compute capability 12.0)
  only, to keep the build short.

[30224]: https://github.com/ggml-org/llama.cpp/issues/30224

Files
-----

| Path | |
|---|---|
| "Dockerfile", "compose.yaml", "mistral-serve" | The images. |
| "patches/" | The llama.cpp fixes. |
| "download-models", "convert-drafters" | Fetch the models and make the BF16 and Q8_0 drafter files the benchmarks used. |
| "bench/bench.py", "bench/load.py" | Single-stream and concurrent-user benchmarks. They can start a container or measure a running server (`--url`). |
| "bench/sweeps/" | The sweeps, in the order they were run. |
| "bench/quality/" | The perplexity measurement. |
| "bench/check-image.py", "bench/vision-matrix.sh" | Functional checks. |
| "bench/summarize.py" | Prints the tables above from "bench/results/". |
| "bench/results/" | Raw results, including each server's log. |

**AI Disclosure:**
- **Authorship:** Autonomous
- **Models:** claude-sonnet-5-5 (Anthropic)
- **Scope:** Wrote everything in this folder (the images, the patches, the
  benchmarks and this documentation) and ran the benchmarks so the user
  could have optimized 8-bit and 4-bit Mistral Small 4 images, choosing the
  4-bit quant by the user's rule: measure each, group the ones within 10%
  of one another by speed and take the highest quality one in the fastest
  group.
- **References:**
  - <https://huggingface.co/mistralai/Mistral-Small-4-119B-2603>
  - <https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF>
  - <https://huggingface.co/RedHatAI/Mistral-Small-4-119B-2603-speculator.dspark>
  - <https://huggingface.co/RedHatAI/Mistral-Small-4-119B-2603.dflash>
  - <https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-eagle>
  - <https://github.com/ggml-org/llama.cpp> (v0.8.0, 8642541)
