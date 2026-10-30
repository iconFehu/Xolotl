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
