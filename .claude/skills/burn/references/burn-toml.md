# `burn.toml`

Burn 0.21 introduced a project-level `burn.toml` file that centralizes runtime knobs which used to live across env vars, build features, and Rust APIs. Drop it at the project root next to `Cargo.toml`. Burn picks it up automatically.

The shape and vocabulary is **per-subsystem** — each section uses words that fit that subsystem. So autotune uses `minimal` / `balanced` / `extensive` / `full`, beam search uses `disabled` / `basic` / `verbose`, etc. Don't try to memorize the whole alphabet of values; read the section you care about.

## Full example

```toml
# burn.toml — drop at project root, beside Cargo.toml

[fusion.beam_search]
max_blocks = 5

[cubecl.autotune]
level = "balanced"   # "minimal" | "balanced" | "extensive" | "full"
cache = "target"     # "local" | "target" | "global" | { file = "<path>" }

[cubecl.compilation]
check_mode = "auto"  # "enforce" | "validate" | "auto"
cache = "target"

[cubecl.streaming]
max_streams = 8

[cubecl.memory]
persistent_memory = "enabled"  # "enabled" | "disabled" | "enforced"

# Per-component logging
[cubecl.autotune.logger]
level = "full"
targets = ["stdout", "log", { file = "autotune.log" }]

[fusion.beam_search.logger]
level = "basic"
targets = ["stderr"]
```

## Section by section

### `[fusion.beam_search]`

Tunes Burn's kernel-fusion beam search.

- `max_blocks` (int) — how many adjacent operation blocks to consider for fusion. Higher = more aggressive fusion, more compile time at startup, potentially fewer kernels launched. Default is reasonable for most workloads; raise it if profiling shows a lot of unfused element-wise sequences.

### `[cubecl.autotune]`

Controls how aggressively CubeCL benchmarks kernel variants when it first sees a new (op, dtype, shape) tuple.

- `level` — one of:
  - `"minimal"` — only obvious fast paths, fastest cold start, may leave perf on the table
  - `"balanced"` — default, decent trade-off
  - `"extensive"` — wider search, slower cold start, better steady-state perf
  - `"full"` — exhaustive search, longest cold start
- `cache` — where to store the autotune cache:
  - `"local"` — project-local cache directory
  - `"target"` — inside the `target/` build dir (default)
  - `"global"` — `~/.cache/cubecl/` or platform equivalent
  - `{ file = "path/to/cache.json" }` — explicit file

For deployment, set `cache = { file = "..." }` and bundle that file with the binary, so the first run on a new machine doesn't pay autotune cost.

### `[cubecl.compilation]`

Controls the CubeCL kernel-compilation pipeline.

- `check_mode`:
  - `"enforce"` — reject any kernel that fails validation (strict mode for dev)
  - `"validate"` — run validation, log failures, still execute
  - `"auto"` — heuristic; default
- `cache` — same options as `[cubecl.autotune].cache`.

The new kernel validation layer is the thing that "caught kernels generating out-of-bounds memory accesses" in 0.21 development. Useful to set `"enforce"` when developing custom kernels.

### `[cubecl.streaming]`

Controls how many parallel CubeCL streams can run on a device.

- `max_streams` (int) — default depends on backend. Raising it can improve throughput when many small kernels are queued; lowering it reduces memory pressure on GPUs with limited resources.

### `[cubecl.memory]`

- `persistent_memory`:
  - `"enabled"` — keep an allocator pool warm between kernels (default for GPU backends)
  - `"disabled"` — return buffers to the system after each kernel
  - `"enforced"` — same as enabled, but errors if the backend can't honor it

Disable this if you're memory-constrained (small GPU + large model) and would rather pay re-allocation cost than over-pool.

### Per-component loggers

Each subsystem with a `[<name>.logger]` section gets its own routing:

```toml
[cubecl.autotune.logger]
level = "full"
targets = ["stdout", "log", { file = "autotune.log" }]
```

- `level` — vocabulary is **per-subsystem**:
  - `cubecl.autotune.logger.level`: `"disabled"` | `"minimal"` | `"full"`
  - `fusion.beam_search.logger.level`: `"disabled"` | `"basic"` | `"verbose"`
- `targets` — array of routes. Each entry is one of:
  - `"stdout"` — print to stdout
  - `"stderr"` — print to stderr
  - `"log"` — go through Rust's `log` crate (so respects `RUST_LOG` and any subscriber)
  - `{ file = "path.log" }` — write to a file
  Combine freely.

## When to actually edit `burn.toml`

Most apps don't need a `burn.toml`. Defaults are tuned. Reach for it when:

- **Cold-start matters.** Set `cache = { file = "..." }` and bundle the file so production deploys skip autotune.
- **You're seeing memory pressure on GPU.** Try `[cubecl.memory] persistent_memory = "disabled"`.
- **You're developing a custom kernel.** Set `[cubecl.compilation] check_mode = "enforce"` and `[cubecl.autotune.logger] level = "full"`. The validation layer catches OOB accesses; the autotune logs tell you which kernel won and why.
- **You're investigating a perf regression after upgrading.** Enable both autotune and beam-search verbose logs:
  ```toml
  [cubecl.autotune.logger]
  level = "full"
  targets = ["stderr"]

  [fusion.beam_search.logger]
  level = "verbose"
  targets = ["stderr"]
  ```
  Then run a single iteration and compare to the previous version's output.

## Things to know

- **Defaults still work.** If `burn.toml` is absent, Burn behaves as if every section were at its default value. Don't create one until you have a reason.
- **Section vocabulary may grow.** New 0.21 mechanism, more subsystems will gain config sections over time. If a knob you want isn't here, check the next release.
- **Per-component logging respects `log` crate setup.** Targeting `"log"` means whatever subscriber the user's app installed (`env_logger`, `tracing-log` bridge, etc.) gets the messages.
- **The autotune cache from `[cubecl.autotune] cache = "target"`** is invalidated on `cargo clean`. That's usually fine in dev. In CI, use `"global"` or `{ file = "..." }` to survive cleans.
- **No env-var precedence rules documented yet.** When in doubt, prefer `burn.toml` over env vars; the project file is what 0.21 is centralizing on.
