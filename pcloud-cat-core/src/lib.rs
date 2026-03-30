//! Point cloud processing built on
//! [`comp-cat-rs`](https://docs.rs/comp-cat-rs): ICP registration,
//! surface normal estimation, and anomaly detection as composable,
//! lazy effects.
//!
//! # Architecture
//!
//! All algorithms are expressed as `comp-cat-rs` effects:
//!
//! - **ICP iteration** is a [`Stream::unfold`](comp_cat_rs::effect::stream::Stream::unfold),
//!   where each pull computes one iteration step.
//! - **ICP alignment** folds the stream into a single
//!   [`Io<Error, RigidTransform>`](comp_cat_rs::effect::io::Io).
//! - **Normal estimation** and **anomaly detection** are
//!   [`Io::suspend`](comp_cat_rs::effect::io::Io::suspend) computations.
//! - **File I/O** uses [`Resource`](comp_cat_rs::effect::resource::Resource)
//!   for bracket-based handle management.
//!
//! Nothing executes until `.run()` is called at the outermost boundary.

pub mod algebra;
pub mod anomaly;
pub mod cloud;
pub mod error;
pub mod io_formats;
pub mod normal;
pub mod registration;
