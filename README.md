# pcloud-cat

Point cloud processing built on [comp-cat-rs](https://github.com/MavenRain/comp-cat-rs): ICP registration, surface normal estimation, and anomaly detection as composable, lazy effects.

## Crates

| Crate | Description |
|---|---|
| `pcloud-cat-core` | Core library: algebra, point cloud types, ICP, normals, anomaly detection, file I/O |
| `pcloud-cat-viewer` | WASM-based 3D point cloud viewer using WebGL2 |

## Quick Start

```rust
use pcloud_cat_core::{
    algebra::{NonNegF64, PosUsize, RigidTransform},
    cloud::PointCloud,
    registration::{icp_align, IcpConfig, IcpVariant},
};

// Build an ICP pipeline (nothing executes yet)
let config = IcpConfig::new(
    IcpVariant::PointToPoint,
    PosUsize::new(50).unwrap_or_else(|| unreachable!()),
    NonNegF64::new(1e-6).unwrap_or_else(|| unreachable!()),
);

let align_io = icp_align(source_cloud, target_cloud, config);

// Execute at the boundary
let transform: Result<RigidTransform, _> = align_io.run();
```

## Architecture

All algorithms are expressed as comp-cat-rs effects:

- **ICP iteration** is a `Stream::unfold`, where each pull computes one iteration step.
- **ICP alignment** folds the stream into a single `Io<Error, RigidTransform>`.
- **Normal estimation** and **anomaly detection** are `Io::suspend` computations.
- **File I/O** uses `Resource` for bracket-based handle management.

Nothing executes until `.run()` is called at the outermost boundary.

## Core Features

- **Point-to-point and point-to-plane ICP** with configurable convergence thresholds
- **PCA-based surface normal estimation** with k-nearest neighbor support
- **Statistical outlier removal** based on mean neighbor distances
- **Surface deviation detection** against a reference cloud
- **PLY file parsing** via `Resource`-managed I/O
- **Zero external dependencies** beyond comp-cat-rs (hand-rolled 3x3 SVD, eigendecomposition)

## Building

```sh
# Build the core library
cargo build -p pcloud-cat-core

# Build the WASM viewer
wasm-pack build pcloud-cat-viewer --target web

# Run tests
cargo test --workspace

# Run clippy
RUSTFLAGS="-D warnings" cargo clippy --workspace
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
