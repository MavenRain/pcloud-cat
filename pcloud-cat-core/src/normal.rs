//! Surface normal estimation via PCA.
//!
//! Estimates a surface normal at each point by fitting a local plane
//! to its k nearest neighbors.  The normal is the eigenvector
//! corresponding to the smallest eigenvalue of the local covariance
//! matrix.
//!
//! The result is an [`Io`] that defers the (expensive) computation
//! until `.run()` is called.

use comp_cat_rs::effect::io::Io;

use crate::algebra::{symmetric_eigen_3x3, Mat3, PosUsize, Vec3};
use crate::cloud::{PointCloud, PointNormal};
use crate::error::Error;

/// Configuration for surface normal estimation.
#[derive(Debug, Clone, Copy)]
pub struct NormalEstimationConfig {
    k_neighbors: PosUsize,
}

impl NormalEstimationConfig {
    /// Construct with the given neighbor count.
    #[must_use]
    pub fn new(k_neighbors: PosUsize) -> Self {
        Self { k_neighbors }
    }

    /// The number of nearest neighbors used for each local PCA.
    #[must_use]
    pub fn k_neighbors(self) -> PosUsize {
        self.k_neighbors
    }
}

/// Estimate surface normals for every point in a cloud.
///
/// Returns an [`Io`] that, when run, produces a new [`PointCloud`]
/// with normals attached.
///
/// # Errors
///
/// Returns [`Error::InsufficientPoints`] if the cloud has fewer
/// points than `k_neighbors`, or [`Error::Algebra`] if
/// eigendecomposition fails.
///
/// # Examples
///
/// ```rust,ignore
/// let config = NormalEstimationConfig::new(PosUsize::new(10).unwrap());
/// let cloud_with_normals: Result<PointCloud, Error> =
///     estimate_normals(&cloud, config).run();
/// ```
#[must_use]
pub fn estimate_normals(cloud: &PointCloud, config: NormalEstimationConfig) -> Io<Error, PointCloud> {
    let cloud = cloud.clone();
    Io::suspend(move || compute_normals(&cloud, config))
}

/// Pure implementation of normal estimation.
fn compute_normals(cloud: &PointCloud, config: NormalEstimationConfig) -> Result<PointCloud, Error> {
    let k = config.k_neighbors.value();

    if cloud.len() < k {
        Err(Error::InsufficientPoints {
            required: k,
            found: cloud.len(),
        })?;
    }

    let normals: Result<Vec<Vec3>, Error> = cloud
        .points()
        .iter()
        .map(|point| estimate_single_normal(point.position(), cloud, k))
        .collect();

    normals.and_then(|ns| {
        cloud.with_normals(&ns).ok_or(Error::InsufficientPoints {
            required: ns.len(),
            found: cloud.len(),
        })
    })
}

/// Estimate the normal at a single point using its k nearest neighbors.
fn estimate_single_normal(
    query: Vec3,
    cloud: &PointCloud,
    k: usize,
) -> Result<Vec3, Error> {
    // Find k nearest neighbors (brute force, sorted by distance).
    let neighbors = k_nearest(query, cloud, k);

    // Compute centroid of neighbors.
    #[allow(clippy::cast_precision_loss)]
    let centroid = neighbors
        .iter()
        .fold(Vec3::zero(), |acc, p| acc + *p)
        .scale(1.0 / neighbors.len() as f64);

    // Build 3x3 covariance matrix.
    let covariance = neighbors.iter().fold(Mat3::zero(), |acc, p| {
        let d = *p - centroid;
        acc + Mat3::outer(d, d)
    });

    // Eigenvector for smallest eigenvalue is the surface normal.
    symmetric_eigen_3x3(covariance)
        .map_err(Error::from)
        .and_then(|eig| {
            // Eigenvalues are sorted ascending; first eigenvector is the normal.
            eig.eigenvectors()
                .col(0)
                .ok_or(Error::Algebra(crate::error::AlgebraErrorKind::SingularMatrix))
        })
        .and_then(|n| {
            n.normalized().ok_or(Error::Algebra(
                crate::error::AlgebraErrorKind::SingularMatrix,
            ))
        })
}

/// Find the k nearest neighbors of `query` in `cloud`.
///
/// Returns their positions (not including `query` itself if it appears
/// in the cloud, though we do not filter it out since the distance
/// would be zero and the covariance unaffected).
fn k_nearest(query: Vec3, cloud: &PointCloud, k: usize) -> Vec<Vec3> {
    // Collect all distances, partially sort to get k smallest.
    let distances: Vec<(usize, f64)> = cloud
        .points()
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p.position().distance_squared(query)))
        .collect();

    // Simple selection: sort and take k.  For large clouds a k-d tree
    // would be better, but this is correct and sufficient for now.
    // For correctness, we do a simple O(n*k) selection.
    select_k_smallest(distances, k)
        .iter()
        .filter_map(|&(i, _dist)| cloud.points().get(i).map(PointNormal::position))
        .collect()
}

/// Select the k entries with smallest second component.
///
/// O(n*k) selection without mutation: builds the result by folding.
fn select_k_smallest(items: Vec<(usize, f64)>, k: usize) -> Vec<(usize, f64)> {
    items.into_iter().fold(Vec::new(), |acc, item| {
        if acc.len() < k {
            insert_sorted(&acc, item)
        } else {
            let should_insert = acc.last().is_none_or(|&last| item.1 < last.1);
            if should_insert {
                insert_sorted(&acc, item).into_iter().take(k).collect()
            } else {
                acc
            }
        }
    })
}

/// Insert an item into a vec sorted by the second component (ascending).
fn insert_sorted(existing: &[(usize, f64)], item: (usize, f64)) -> Vec<(usize, f64)> {
    let pos = existing
        .iter()
        .position(|&(_, d)| d > item.1)
        .unwrap_or(existing.len());

    existing
        .iter()
        .copied()
        .take(pos)
        .chain(std::iter::once(item))
        .chain(existing.iter().copied().skip(pos))
        .collect()
}
