# KV memory branch validation

Base: Xolotl `90435e83bcbd557cef004957ab71aba5f7ad6326`.
This source export uses the v55 native interface and v6 resident API adapter.
The source branch does not deploy or replace an existing model service.

## Checks on the exported tree

- Windows x64, MSVC 19.44, CUDA 12.8.93, `sm_89`, C++20, Release.
- Independently configured and built `strata` and all five KV contract targets
  from this tree, using the repository's pinned patched GGML checkout. The build
  used two jobs and did not run a model or the GPU pool-parity executable.
- `ctest -L kv-memory`: **5/5 passed** — native transaction ownership/epoch/fence,
  recorded geometry, MTP active validity, compact layout and hot-reset contract.
- `python tools/test_kv_memory_default.py`: passed for absent, null, empty and
  disabled configuration; no adapter import, engine replacement or service install.
- `python tools/test_kv_memory.py`: model-free HTTP/request-ID/transaction tests
  passed, including text/image matching, history/backend/state invalidation,
  cancellation, log credential exclusion, and withholding an invalid warm answer
  while publishing only the simulated complete-text retry.
- `python tools/test_kv_memory_images.py`: 17 identity scenarios passed, including
  multiple image ordering, with five malformed media-layout/key cases refused.
- The disabled hook can be removed from `serve/server.py` to recover the upstream
  source; upstream request logging and current GGML pool hooks were preserved.
- Changed-file scan: deployment keys, private endpoint/domain, machine-specific
  absolute paths, model files, binaries, logs and KV payloads are excluded.

The initial build exposed a missing CUDA header dependency in standalone test
targets. Linking those targets to `CUDA::cudart` fixed the include/runtime dependency;
the final incremental build and five CTests passed. This is a build integration
repair, not an additional model numerical result.

## Existing Windows test limitation

`python -m unittest serve.test_request_log` ran 20 tests with no assertion failures,
but 12 teardown errors (`PermissionError: WinError 32`) because temporary request
log files remained open during directory cleanup. The same 12 cleanup cases
reproduced on clean, unmodified upstream `90435e83`. They are recorded as existing
Windows test failures, **not counted as passes** and not hidden by modifying the
upstream tests. The feature-specific tests above close their own listeners/files.

## What this does not establish

No new model/GPU inference was run from this rebased export. The measured long
history and image results in [KV_MEMORY.md](KV_MEMORY.md) belong to the earlier
deployed candidate. The original service and its binary/configuration stayed in
place while this fork was compiled.

HIP, SYCL, multi-GPU, other models/quantizations, Linux hardware, strict cold/warm
bitwise model parity, a 512K full-window accuracy sweep and long-duration numerical
stability were not validated for this source export. The overlay is CUDA-only and
off by default; it is not an upstream-ready cross-backend certification.

## EOF-handling follow-up (2026-10-30)

Only Python adapter code changed; the server hook and native executable stayed
unchanged. The following feature tests passed on Windows/Python 3.11:

- `tools/test_kv_memory.py`: ordinary HTTP, transaction fencing, text/image
  invalidation, cancellation, discarded bad warm output and an audited retry.
- `tools/test_kv_memory_failures.py`: a real child exiting with code 37,
  classified fatal stdout, cancellation reason/deduplication, bounded audited
  retry, double failure withholding output, no replay after publication, and
  actual HTTP/SSE error responses in three protocol dialects.
- `tools/test_kv_memory_images.py`: 17 image identity scenarios and five malformed
  layouts. `tools/test_kv_memory_default.py`: no adapter import/engine replacement
  for absent, null, empty or disabled configuration.

An isolated GPU fixture used the **already deployed native candidate**, not the
rebased fork executable: SC117 IQ3_XXS, 524288 context, 8192 output limit, Low,
fixed mapping/MTP window and vision. Six requests covered priming, a deliberately
killed native process with verified complete-text retry, re-priming, repeated
failure with explicit SSE error/no model output, next-request recovery and actual
socket cancellation. Write/flush failure now records `QBDIAG`, native run/PID and
exit code 1; this exit code belongs to injected termination, not the historical
unexplained failure.

Cancellation skipped audit and automatic retry, but the fixture's `MEM_END`
also encountered EOF and discarded the backend. This is recorded as a failed
transaction, not evidence that the native engine survives every cancellation.
The underlying historical native-exit cause cannot be reconstructed from the
old logs; it is not attributed to OOM or client cancellation without evidence.

The first local trial failed a diagnostic-coverage check for write/flush EOF
and automatically restored the prior version. The second trial included that
classifier and passed the failure semantics above. Subsequent live/public
models, text/image generation, resident text/image SSE reuse and Summary Off
probes passed, with native state/MTP audits and protected-file hashes checked.
No fault-injection module or test-only control endpoint is installed in the
live adapter. These checks do not replace long-duration stability testing.

## Runtime ERR classification follow-up (2026-11-01)

The deployed native executable and server hook were unchanged. Five feature
scripts passed: the existing default-off, image, ordinary adapter and EOF/SSE
failure suites, plus `tools/test_kv_memory_native_errors.py`. The new script
checks ten recognized message forms, bounded numeric parameters, omitted unknown
credentials/paths/URLs/prompt fragments, suffix/multiline/oversize inputs, keyed
fingerprints, real-child pipe sanitization and same-request exception association.

An isolated instance of the deployed GPU candidate handled five fixture queries:
prime, an actual native ERR followed by verified cold retry, re-prime, two ERRs
ending in explicit SSE failure without model output, and next-request recovery.
The fixture deliberately sent `GEN max_new=0` for selected native calls; ordinary
API requests still used an 8192 output limit. All three native errors were
classified as `INVALID_MAX_NEW`, assigned different error IDs and linked to their
generation/retry failures. The server's stdout received the sanitized summary.
The test-only adapter is not installed in the live runtime. It validates error
handling, **not** bitwise replay of a missing historical 221K-token request.

The historical ValueError was independently located in retained frontend stdout:
`GEOMETRY_LINEAGE_MISS resume=221082 identity`. Source inspection shows an
identity rejection after a length-indexed geometry entry was found, before new
prefill. The preceding prompt geometry covered 219278 tokens while the main
state reached 221082. Without the old geometry-map snapshot and complete failed
history, the exact writer/origin of the mismatched entry is not established.
The historical guard correctly refused it; a verified complete-text retry
recomputed 221370 tokens and completed in 157.862 seconds. Neither the guard nor
the 32K+ geometry replay constraint was relaxed by this logging change.

Local and public models, normal text/image generation, resident text/image SSE
reuse and Summary Off probes passed after activation. Protected-file and native
state/MTP audits passed. Startup/old/raw stderr/state logs remain private and are
outside the runtime ERR sanitization guarantee. The prior cancellation/engine
survival limitation and other platform/long-duration limitations still apply.

## Cancelled prefill / queued ownership follow-up (2026-11-02)

The old deployed candidate was reproduced independently: a 264063-token prompt
was disconnected after progress reached 8192 tokens. Its nominal final-window
MTP check reported missing cells, followed by `MEM_END` EOF and process exit.
The old adapter's actual two-request HTTP test also reproduced queued B replacing
active A's native request ID. The repaired adapter passes the same ownership test.

The revised candidate passed 16 isolated GPU/API cases covering ordinary answers,
cold-prefill cancellation/retry, a persistent checkpoint followed by restored
prefill cancellation/retry, decode cancellation/retry, queued/disconnected B while
A owns the native lease, controlled reset failure/discard/recovery, reasoning
budget continuation, forced tool choice, explicit re-priming and one bounded
audited retry after a real injected native ERR. In normal cancellation cases the
original native PID was retained; no candidate MTP miss or control EOF occurred.
State/GDN/PLE/index/MTP audits and expected synthetic record values passed.

The native reset-required boundary is limited to cancelled **prefill**. Ordinary
decoder STOP also serves reasoning-budget/tool continuations and must remain
usable. The first candidate's broader decoder-STOP boundary was identified during
coverage review; an activation hold arrived after CAS, so that candidate was
briefly enabled and then restored to the previous version. That interval's logs
contain only six fixed acceptance probes and no recorded failures. It is not
counted as final acceptance. A later isolated acceptance failed because the ERR
fixture lost its warm-history precondition after tool testing; cold ERR was
correctly not replayed. That failed round is retained, not counted as a pass.
Final acceptance explicitly re-primes before testing the existing reuse retry.

Both the pinned candidate and fork CUDA builds passed. The deployed executable
uses the same eleven frozen compute libraries as the previous deployment; its
native host/control source changed. The newly compiled fork executable itself was
not installed. Six model-free CTest contracts (including cancel policy) and the
Python default, image, adapter, ERR, EOF/SSE and cancellation/ownership suites
passed. Local/public text, vision, resident SSE reuse and Summary Off probes
passed after activation, with configuration/key/protected-file checks and
rollback evidence retained. No HIP/SYCL, long-duration soak, full-window accuracy
or Desktop AgentLoop certification is implied. Genuine non-cancelled validity
failures still fail closed; only unconsumable cancelled prefill avoids the false
full-prompt completion check.
