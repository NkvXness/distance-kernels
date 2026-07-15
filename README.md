# distance-kernels

A small Rust workspace for distance and similarity kernels used in ML experiments.
The project starts with safe Rust baseline implementations and keeps low-level
optimization work behind tests and benchmarks.

## Current status

- `kernels` provides baseline `f32` implementations for dot product, squared
  L2 distance, L2 norm, L2 normalization, cosine similarity, and nearest
  squared-L2 lookup.
- `kernels` exposes panic-on-contract APIs and checked `Result`-based variants
  for callers that want explicit error handling.
- `kernels` includes batch scoring, top-k squared-L2 lookup, cosine search, and
  a small `u32` kNN prediction helper.
- `cli` provides a command-line wrapper for running the kernels manually.
- CI runs formatting, Clippy, and tests on Linux and Windows.
- Benchmarks, Python validation scripts, and SIMD/ASM backends are planned but
  are not part of the current public API yet.

## Workspace

- `crates/kernels` - reusable library code.
- `crates/cli` - command-line entry point for manual checks and experiments.

## Library example

```rust
use kernels::{
    cosine_similarity_f32, dot_f32, knn_predict_l2_sq_u32, l2_sq_f32,
    nearest_l2_sq_f32, normalized_l2_f32,
};

let a = [1.0, 2.0, 3.0];
let b = [4.0, 5.0, 6.0];
let candidates: [&[f32]; 2] = [&[9.0, 9.0, 9.0], &[2.0, 3.0, 4.0]];
let labels = [0, 1];

assert_eq!(dot_f32(&a, &b), 32.0);
assert_eq!(l2_sq_f32(&a, &b), 27.0);
assert!(cosine_similarity_f32(&a, &b).is_some());
assert!(normalized_l2_f32(&a).is_ok());
assert_eq!(nearest_l2_sq_f32(&a, &candidates), Ok((1, 3.0)));
assert_eq!(knn_predict_l2_sq_u32(&a, &candidates, &labels, 1), Ok(1));
```

## CLI examples

```powershell
cargo run -p cli -- dot 1,2,3 4,5,6
cargo run -p cli -- l2-sq 1,2,3 4,5,6
cargo run -p cli -- cosine 1,0 0,1
cargo run -p cli -- norm 3,4
cargo run -p cli -- normalize 3,4
cargo run -p cli -- l2-sq-all 1,1 "1,1;2,1;3,3"
cargo run -p cli -- nearest-l2-sq 1,1 "5,5;2,1;0,0"
cargo run -p cli -- nearest-k-l2-sq 1,1 "5,5;2,1;1,1;0,0" 3
cargo run -p cli -- nearest-cosine 1,0 "0,1;1,0;-1,0"
cargo run -p cli -- knn-l2-sq 1,1 "1,1;1.2,1.1;8,8;7.5,8" "10,10,20,20" 3
```

## Examples

```powershell
cargo run -p kernels --example knn_demo
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
3. Add small ML examples on top of the kernels, starting with kNN.
4. Add Python reference scripts for research validation.
5. Add optimized backends only after benchmarks justify them.
