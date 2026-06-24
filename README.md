# distance-kernels

A small Rust workspace for distance and similarity kernels used in ML experiments.
The project starts with safe Rust baseline implementations and keeps low-level
optimization work behind tests and benchmarks.

## Current status

- `kernels` provides baseline `f32` implementations for dot product, squared
  L2 distance, squared L2 norm, and cosine similarity.
- `cli` provides a small command-line wrapper for running the kernels manually.
- CI runs formatting, Clippy, and tests on Linux and Windows.
- Benchmarks, Python validation scripts, and SIMD/ASM backends are planned but
  are not part of the current public API yet.

## Workspace

- `crates/kernels` - reusable library code.
- `crates/cli` - command-line entry point for manual checks and experiments.

## Library example

```rust
use kernels::{cosine_similarity_f32, dot_f32, l2_sq_f32};

let a = [1.0, 2.0, 3.0];
let b = [4.0, 5.0, 6.0];

assert_eq!(dot_f32(&a, &b), 32.0);
assert_eq!(l2_sq_f32(&a, &b), 27.0);
assert!(cosine_similarity_f32(&a, &b).is_some());
```

## CLI examples

```powershell
cargo run -p cli -- dot 1,2,3 4,5,6
cargo run -p cli -- l2-sq 1,2,3 4,5,6
cargo run -p cli -- cosine 1,0 0,1
```

## Development checks

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
```

## Roadmap

1. Keep safe Rust kernels as the correctness reference.
2. Add benchmark coverage for throughput comparisons.
3. Add Python reference scripts for research validation.
4. Add optimized backends only after benchmarks justify them.
