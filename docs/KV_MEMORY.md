# Experimental KV memory for the Flash-Next CUDA engine

This branch preserves the local native KV experiment and the v6 API adapter as
an **opt-in, single-GPU CUDA feature**. It is a development branch, not a tested
release for every Xolotl model or backend. Existing deployments are not changed
by checking out or building this branch.

With `STRATA_ENABLE_KV_MEMORY=OFF` (the default), the original engine sources
and compile definitions remain selected. Without an enabled `memory_gateway`
configuration, the Python server does not import or install the adapter.
Upstream's request logger and the current GGML pool-hook fix are retained.

## Sources

- `experimental/kv_memory/native/`: the independently validated QSA/GDN/PLE
  prototype, page pool, persistent session/block controls, geometry replay,
  MTP active-region diagnostics, and native transaction fence.
- `cmake/kv_memory.cmake`: selects those sources only for the explicit CUDA build.
- `serve/kv_memory/`: global transaction isolation, signed text snapshot catalog,
  exact text/image resident prefix matching, validation, fallback, and logs.
- `tools/test_kv_memory*.py`: model-free HTTP, identity, cancellation, invalidation
  and publication tests. They use temporary folders and synthetic credentials.

The overlay intentionally keeps upstream source files in place, making it easy
to compare/rebase the experiment and to build an unmodified default backend.
It currently duplicates several large source files; this is a preservation
branch, not a claim that the experiment has been reduced to a minimal upstream PR.

## Build and model-free checks

Use the repository's normal CUDA toolchain and pinned, patched GGML dependency.
The dependency is llama.cpp `6050b467c07a95c7b87debd4ddac2f8ba8d9b807` plus
`tools/xolotl/ggml/xolotl-ggml.patch`. The top-level build fetches that dependency;
an existing patched checkout may instead be passed as `STRATA_GGML_DIR`.

```sh
cmake -S . -B build-kv-memory -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_ENABLE_KV_MEMORY=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-kv-memory --parallel 2 --target strata \
  kv_memory_fence_test kv_geometry_test kv_mtp_validity_test \
  kv_compact_layout_test kv_hot_reset_contract_test
ctest --test-dir build-kv-memory -L kv-memory --output-on-failure
python tools/test_kv_memory_default.py
python tools/test_kv_memory.py
python tools/test_kv_memory_images.py
```

Use `-G Ninja -DCMAKE_BUILD_TYPE=Release` in an initialized MSVC developer shell
on Windows. Select the actual CUDA architecture for another NVIDIA GPU.
`kv_pool_parity_test` is a separate synthetic **GPU** executable; it is deliberately
not registered with CTest. Running it consumes GPU resources and needs a free GPU.
HIP/SYCL and multi-GPU activation are rejected by the overlay. Their default
build paths are not replaced; this branch does not claim their validation.

## Enable on a separate test server

Start from a working setup-generated Flash-Next configuration; preserve its model,
pack, tokenizer, MTP assets and API key. Merge the `memory_gateway` and `env` sections
from `configs/kv-memory.example.json` instead of replacing the whole model config.
Point its engine executable at the new experimental `strata` binary.

Set an **absolute**, private data path and a namespace digest that binds the engine
binary, template, model and relevant arguments. Change the namespace/data directory
when changing these inputs. Never mount an untrusted snapshot library. The catalog
HMAC detects edits; it does not encrypt data or protect against an administrator
who can rewrite the key and payloads together.

The measured configuration used an INT8 main KV stream, a real MTP ring, at least
three native prompt-cache checkpoints, `--kv-resident 32768`, a fixed MTP window
of 32768 and fixed expert placement. Keep the existing model's MTP asset and
argument syntax. This adapter requires its advertised native geometry/MTP/fence
capabilities and refuses a stock engine that does not provide them.

The local experiment used `--max-context 524288`, YaRN factor 2, `--spec 4`, a
512K reported context and an 8192-token output allowance. **The example's
`normal`/`summary` strings document the client policy; they do not override a
request's reasoning settings or enforce an output reservation.** Keep the client's
Low/Off routes and token budgeting configured separately.

The ordinary API retains its existing listener and authentication. The additional
control listener binds **only `127.0.0.1`** on the configured port. Its mode-changing
POSTs use the same server API key; do not expose it through a public proxy.
The data directory contains sensitive token IDs and KV state: give it private
filesystem permissions. No machine-specific firewall, scheduled task, public
domain or credential is included in this branch.

To disable, stop the test service, use the previous binary/config, and restart.
For a compiled experimental binary, disable both the gateway and its native fence
environment flag together; an enabled fence rejects an unfenced stock frontend.
Keep previous binaries/configs and immutable evidence outside the source tree.

## Resident reuse and failures

The adapter retains a proof for the most recent compatible native session:

1. The native process object and the complete submitted history prefix must match.
2. `QBCHECK` must still match the verified GDN/PLE/index guard and live-token hash.
3. Images additionally bind ordered positions, dimensions, encoder bytes (SHA256),
   and frontend identity keys. Equal URLs or image-pad tokens are insufficient.
4. The native engine chooses the actual matching checkpoint using raw IDs,
   image identities and recorded geometry. Logged `reused` is its measured count.
5. Reused answers are buffered until `QBDIAG`/`QBCHECK` and the MTP active-region
   validity check pass. Invalid state is discarded and a complete-text retry is
   attempted before publication. Already emitted cold streams are not replayed.

Changes, cancellation, incompatible text/image switching, unknown identity or a
backend restart invalidate resident proofs. Ordinary model output has no generic
semantic-correctness oracle; a valid KV state does not guarantee a correct answer.

Resident matching can work above 64K. **Disk snapshots are still admitted only
for text requests up to 65536 tokens**, with a minimum 4096-token prefix and one
active version. Images are not persisted by this adapter. The configured 8 GiB
budget is an admission check with a 4 GiB save reservation and post-write rejection,
not a filesystem quota or a guarantee on every diagnostic file's growth.

For strict comparisons at/above 32K, replay the checkpoint's recorded prefix
execution geometry using `QBGEOMPREFIX` and its `.g<token-count>` file. A different
cold batching shape changes the MTP active region and may change floating-point
results. Do not silently compare it as a bitwise-equivalent baseline.

## Logs and validation scope

Every API POST gets `X-KV-Request-ID`; JSON errors include `kv_request_id`.
`logs/events.jsonl` records native transaction epochs, input counts/hashes,
prefix selection/invalidation, actual reused/recomputed tokens, guard validity,
timings, HTTP status, and exception type/source stack. It excludes prompt/answer
text, raw tool output, Authorization and API keys. Raw native traces can still
contain sensitive token/state diagnostics and should stay private.

Look for `text_reuse_selected`, `image_reuse_selected`, and
`generation_done.text_warm_selected` / `.image_warm_selected`. A skipped disk
checkpoint or `health.blocks == 0` does not mean resident reuse is disabled.
`resident_common_tokens` and `resident_required_prefix_tokens` help locate a
client's history rewrite without logging the text itself.

Historical local measurements: Windows, RTX 4090 48 GB, 64 GB RAM,
SC117 Flash-Next IQ3_XXS, 512K configured context, fixed expert mapping, MTP=4,
32768 MTP window, Low, and 8192 output limit. Four interleaved growing histories
contained 123075–132415 tokens; each pair used byte-identical API payloads and
sampling (temperature 0, seed 42). Two rounds included fixed tool evidence.

| Four-query group | Cold recompute | Resident reuse |
|---|---:|---:|
| Recomputed tokens, total | 510970 | 12439 |
| Reused tokens, total | 0 | 498531 |
| Prefill, median seconds | 28.284 | 2.070 |
| First answer delta, median seconds | 29.262 | 4.955 |
| Request duration, median seconds | 30.886 | 5.008 |

Those four pairs plus seeding and old-source invalidation/repeat checks returned
11/11 expected synthetic values. Same-image tests used 120149-token histories:
three cold queries recomputed 360447 tokens, three resident queries recomputed
15; median request durations were 29.268 and 3.103 seconds. Image changes,
positions, text/image switches and fixed tool transcripts were also checked.

These are **historical deployed-candidate** results (native interface v55, adapter
v6), not a new GPU run of this rebased fork. CPU/CUDA build results for the fork
are recorded separately in `KV_MEMORY_VALIDATION.md`. No HIP/SYCL, arbitrary
model, 512K full-window accuracy, all-branch caching, unlimited memory or
long-duration correctness guarantee is claimed. Desktop DSH AgentLoop integration
remains separate; this feature hooks the API path used by existing clients.

## Native EOF, cancellation and streaming failures

The opt-in adapter records each native process UUID/PID, executable hash, stdout
EOF, exit code (decimal and unsigned hexadecimal), server-requested termination,
and classified verifier failures. Verifier stdout is hashed, never copied into
the public journal. `native_exit` can arrive after `native_stdout_eof`; correlate
both by `native_run`, rather than assuming EOF means the OS already reaped it.
An unexplained exit remains `native_exit_or_external_termination`; it is not
labelled an out-of-memory event without evidence.

`kv_control_failed` includes the operation and distinguishes write/flush EOF
from a read EOF. `request_cancelled` records the request ID and whether the
client disconnected or a caller cancelled it. Cancelled generations discard
resident proofs, skip the post-generation audit, and never launch an automatic
retry. Warm buffered generations forward progress heartbeats while withholding
all model output until the state audit passes.

A failed text reuse/restore may retry **once**, with a fresh native process and
the complete text history, only if the client is still connected and no model
output has been published. The retry is buffered and audited too. A failed
retry emits an explicit API error, not a successful empty answer. Image errors
fail closed to a fresh client request: the original generation may have consumed
its GENI embedding file, so an automatic replay using that file is unsafe.

`sse_failed`, `sse_write_failed` and `http_end.outcome` distinguish application
failure/cancellation from successful delivery. HTTP 200 alone is not a success
criterion once SSE headers have been sent. OpenAI chat, Responses and Anthropic
streams receive their own error framing. Transaction cleanup errors retain the
primary failure instead of replacing it with a second EOF.

Run `python tools/test_kv_memory_failures.py` for model-free real-process and
HTTP/SSE regressions. It needs no GPU, account credentials or runtime service.

## Redacted runtime ERR records

After READY, the adapter classifies native `ERR` frames before forwarding them
to the server. Only fixed templates and bounded numeric fields are retained.
For example, `GEOMETRY_LINEAGE_MISS resume=221082 identity` becomes code
`GEOMETRY_LINEAGE_IDENTITY` with `parameters.resume=221082`. This means a saved
geometry tape was found at that position but its token/image identity did not
match the requested prefix. It is a refusal to reuse, not proof of a crash or
permission to bypass geometry/MTP validation.

`native_error` includes a unique `error_id`, request ID, native run UUID/PID,
operation, family, code and safe summary. Its frame is sanitized before ordinary
`ValueError` handling, so that handler's stdout/API error message cannot re-emit
arbitrary native text. `generation_failed.native_error` and
`cold_retry_failed.native_error` only link to a matching error from the same
request; a stale native error is not attached to an unrelated exception.

Unknown messages are replaced with `NATIVE_ERR_UNCLASSIFIED [native message
omitted]`. There is no raw-message or partially scrubbed fallback field. Paths,
URLs, credentials, prompt fragments, extra suffixes and multiline payloads do not
pass the template allowlist. A private 32-byte `native-log-key.bin` in the data
directory produces a keyed fingerprint for repeated-message correlation; never
publish or include this key in a diagnostic export. It is separate from API keys
and checkpoint authentication. Fingerprints differ across independent data roots.

This scope covers runtime ERR frames and their downstream error handling. It
does not claim that startup logs, previously written logs, raw native stderr or
KV/state traces have been scrubbed; keep those private. Existing one-shot,
audited full-text fallback, cancellation and image lifetime restrictions remain.
Run `python tools/test_kv_memory_native_errors.py` for classifier/privacy and
real-child-pipe regression tests.
