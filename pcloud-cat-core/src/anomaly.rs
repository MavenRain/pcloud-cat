//! Defect and anomaly detection for point clouds.
//!
//! Two detection strategies are provided:
//!
//! - **Statistical outlier removal**: classifies points whose mean
//!   distance to k nearest neighbors exceeds a threshold derived
//!   from the global distribution.
//!
//! - **Surface deviation**: classifies points based on their distance
//!   to a reference cloud (or signed distance along the reference normal).

use comp_cat_rs::effect::io::Io;

use crate::algebra::{NonNegF64, PosUsize, Vec3};
use crate::cloud::PointCloud;
use crate::error::Error;

/// Classification of a single point after anomaly detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PointClassification {
    /// The point is within normal statistical bounds.
    Inlier,
    /// The point is a statistical outlier.
    StatisticalOutlier {
        /// How many standard deviations the point lies from the mean.
        sigma_distance: f64,
    },
    /// The point deviates from the reference surface beyond the threshold.
    SurfaceDeviation {
        /// The signed or unsigned deviation distance.
        deviation: f64,
    },
}

/// A point cloud with per-point anomaly classifications.
#[derive(Debug, Clone)]
pub struct ClassifiedCloud {
    cloud: PointCloud,
    classifications: Vec<PointClassification>,
}

impl ClassifiedCloud {
    /// The underlying point cloud.
    #[must_use]
    pub fn cloud(&self) -> &PointCloud {
        &self.cloud
    }

    /// The per-point classifications (same order as the cloud's points).
    #[must_use]
    pub fn classifications(&self) -> &[PointClassification] {
        &self.classifications
    }

    /// Count of inlier points.
    #[must_use]
    pub fn inlier_count(&self) -> usize {
        self.classifications
            .iter()
            .filter(|c| matches!(c, PointClassification::Inlier))
            .count()
    }

    /// Count of outlier points (statistical or surface deviation).
    #[must_use]
    pub fn outlier_count(&self) -> usize {
        self.classifications.len() - self.inlier_count()
    }

    /// Extract only inlier points as a new cloud.
    ///
    /// Returns `None` if no inliers exist.
    #[must_use]
    pub fn inliers(&self) -> Option<PointCloud> {
        let inlier_points: Vec<_> = self
            .cloud
            .points()
            .iter()
            .zip(self.classifications.iter())
            .filter(|(_, c)| matches!(c, PointClassification::Inlier))
            .map(|(p, _)| *p)
            .collect();

        PointCloud::from_points(inlier_points)
    }
}

/// Configuration for statistical outlier removal.
#[derive(Debug, Clone, Copy)]
pub struct StatisticalOutlierConfig {
    k_neighbors: PosUsize,
    sigma_threshold: NonNegF64,
}

impl StatisticalOutlierConfig {
    /// Construct a configuration.
    ///
    /// `k_neighbors`: number of nearest neighbors for mean distance.
    /// `sigma_threshold`: points beyond `mean + sigma_threshold * stddev`
    /// are classified as outliers.
    #[must_use]
    pub fn new(k_neighbors: PosUsize, sigma_threshold: NonNegF64) -> Self {
        Self { k_neighbors, sigma_threshold }
    }

    /// The neighbor count.
    #[must_use]
    pub fn k_neighbors(self) -> PosUsize {
        self.k_neighbors
    }

    /// The sigma threshold.
    #[must_use]
    pub fn sigma_threshold(self) -> NonNegF64 {
        self.sigma_threshold
    }
}

/// Detect statistical outliers based on mean distance to k nearest neighbors.
///
/// Returns an [`Io`] that, when run, classifies each point in the cloud.
///
/// # Algorithm
///
/// 1. For each point, compute the mean distance to its k nearest neighbors.
/// 2. Compute the global mean and standard deviation of these distances.
/// 3. Points with `mean_dist > global_mean + sigma * global_stddev` are outliers.
///
/// # Errors
///
/// Returns [`Error::InsufficientPoints`] if the cloud has fewer points
/// than `k_neighbors`.
#[must_use]
pub fn statistical_outlier_removal(
    cloud: &PointCloud,
    config: StatisticalOutlierConfig,
) -> Io<Error, ClassifiedCloud> {
    let cloud = cloud.clone();
    Io::suspend(move || compute_sor(&cloud, config))
}

/// Pure implementation of statistical outlier removal.
fn compute_sor(cloud: &PointCloud, config: StatisticalOutlierConfig) -> Result<ClassifiedCloud, Error> {
    let cloud = cloud.clone();
    let k = config.k_neighbors.value();

    if cloud.len() < k + 1 {
        Err(Error::InsufficientPoints {
            required: k + 1,
            found: cloud.len(),
        })?;
    }

    // Compute mean distance to k nearest neighbors for each point.
    let mean_distances: Vec<f64> = cloud
        .points()
        .iter()
        .map(|point| mean_k_distance(point.position(), &cloud, k))
        .collect();

    // Global statistics.
    #[allow(clippy::cast_precision_loss)]
    let n = mean_distances.len() as f64;
    let global_mean = mean_distances.iter().sum::<f64>() / n;
    let variance = mean_distances
        .iter()
        .map(|d| {
            let diff = d - global_mean;
            diff * diff
        })
        .sum::<f64>()
        / n;
    let global_stddev = variance.sqrt();

    let threshold = global_mean + config.sigma_threshold.value() * global_stddev;

    // Classify each point.
    let classifications = mean_distances
        .iter()
        .map(|&d| {
            if d <= threshold {
                PointClassification::Inlier
            } else {
                let sigma_distance = if global_stddev > 1e-15 {
                    (d - global_mean) / global_stddev
                } else {
                    0.0
                };
                PointClassification::StatisticalOutlier { sigma_distance }
            }
        })
        .collect();

    Ok(ClassifiedCloud {
        cloud,
        classifications,
    })
}

/// Compute the mean distance from a query point to its k nearest neighbors.
fn mean_k_distance(query: Vec3, cloud: &PointCloud, k: usize) -> f64 {
    // Collect all distances, find k smallest (excluding self if distance ~0).
    let distances: Vec<f64> = cloud
        .points()
        .iter()
        .map(|p| p.position().distance(query))
        .filter(|&d| d > 1e-15) // exclude self
        .collect();

    // Take k smallest via partial sort (fold-based).
    let k_smallest = take_k_smallest_f64(&distances, k);

    if k_smallest.is_empty() {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let count = k_smallest.len() as f64;
        k_smallest.iter().sum::<f64>() / count
    }
}

/// Select the k smallest values from a slice, without mutation.
fn take_k_smallest_f64(values: &[f64], k: usize) -> Vec<f64> {
    values.iter().fold(Vec::new(), |acc, &val| {
        if acc.len() < k {
            insert_sorted_f64(&acc, val)
        } else {
            let should_insert = acc.last().is_none_or(|&last| val < last);
            if should_insert {
                insert_sorted_f64(&acc, val).into_iter().take(k).collect()
            } else {
                acc
            }
        }
    })
}

/// Insert a value into a sorted (ascending) vector.
fn insert_sorted_f64(existing: &[f64], val: f64) -> Vec<f64> {
    let pos = existing.iter().position(|&d| d > val).unwrap_or(existing.len());
    existing
        .iter()
        .copied()
        .take(pos)
        .chain(std::iter::once(val))
        .chain(existing.iter().copied().skip(pos))
        .collect()
}

/// Configuration for surface deviation detection.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceDeviationConfig {
    max_deviation: NonNegF64,
}

impl SurfaceDeviationConfig {
    /// Construct a configuration.
    ///
    /// `max_deviation`: the threshold beyond which a point is
    /// classified as deviating from the reference surface.
    #[must_use]
    pub fn new(max_deviation: NonNegF64) -> Self {
        Self { max_deviation }
    }

    /// The deviation threshold.
    #[must_use]
    pub fn max_deviation(self) -> NonNegF64 {
        self.max_deviation
    }
}

/// Detect deviations from a reference surface.
///
/// For each point in `measured`, finds the nearest point in `reference`.
/// If the reference point has a normal, computes the signed distance
/// along that normal; otherwise uses unsigned Euclidean distance.
///
/// # Errors
///
/// Returns [`Error::InsufficientPoints`] if either cloud is empty
/// (which should not happen for valid `PointCloud` values).
#[must_use]
pub fn surface_deviation(
    measured: &PointCloud,
    reference: &PointCloud,
    config: SurfaceDeviationConfig,
) -> Io<Error, ClassifiedCloud> {
    let measured = measured.clone();
    let reference = reference.clone();
    Io::suspend(move || Ok(compute_deviation(&measured, &reference, config)))
}

/// Pure implementation of surface deviation detection.
fn compute_deviation(
    measured: &PointCloud,
    reference: &PointCloud,
    config: SurfaceDeviationConfig,
) -> ClassifiedCloud {
    let measured = measured.clone();
    let threshold = config.max_deviation.value();

    let classifications = measured
        .points()
        .iter()
        .map(|mp| {
            reference
                .nearest_neighbor(mp.position())
                .and_then(|(ri, _dist_sq)| reference.points().get(ri).copied())
                .map_or(PointClassification::Inlier, |rp| {
                    let deviation = rp.normal().map_or_else(
                        || mp.position().distance(rp.position()),
                        |n| (mp.position() - rp.position()).dot(n),
                    );

                    if deviation.abs() <= threshold {
                        PointClassification::Inlier
                    } else {
                        PointClassification::SurfaceDeviation { deviation }
                    }
                })
        })
        .collect();

    ClassifiedCloud {
        cloud: measured,
        classifications,
    }
}
