//! Point cloud types.
//!
//! [`PointNormal`] represents a single 3D point with an optional
//! surface normal.  [`PointCloud`] is a non-empty, ordered collection
//! of such points.  [`Correspondence`] records a nearest-neighbor
//! pairing between two clouds.

use crate::algebra::{RigidTransform, Vec3};

/// A 3D point with an optional surface normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointNormal {
    position: Vec3,
    normal: Option<Vec3>,
}

impl PointNormal {
    /// Construct a point without a normal.
    #[must_use]
    pub fn new(position: Vec3) -> Self {
        Self { position, normal: None }
    }

    /// Construct a point with a surface normal.
    #[must_use]
    pub fn with_normal(position: Vec3, normal: Vec3) -> Self {
        Self { position, normal: Some(normal) }
    }

    /// The 3D position.
    #[must_use]
    pub fn position(&self) -> Vec3 {
        self.position
    }

    /// The surface normal, if present.
    #[must_use]
    pub fn normal(&self) -> Option<Vec3> {
        self.normal
    }

    /// Apply a rigid transform to both position and normal.
    #[must_use]
    pub fn transformed(&self, transform: &RigidTransform) -> Self {
        Self {
            position: transform.apply(self.position),
            normal: self.normal.map(|n| transform.rotation().mul_vec(n)),
        }
    }
}

/// A non-empty, ordered collection of 3D points.
#[derive(Debug, Clone, PartialEq)]
pub struct PointCloud {
    points: Vec<PointNormal>,
}

impl PointCloud {
    /// Construct from a non-empty vector of points.
    ///
    /// Returns `None` if the vector is empty.
    #[must_use]
    pub fn from_points(points: Vec<PointNormal>) -> Option<Self> {
        if points.is_empty() { None } else { Some(Self { points }) }
    }

    /// Number of points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the cloud is empty.
    ///
    /// Always returns `false` for a valid `PointCloud`, since
    /// construction requires at least one point.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Borrow the underlying slice of points.
    #[must_use]
    pub fn points(&self) -> &[PointNormal] {
        &self.points
    }

    /// Centroid (mean position) of the cloud.
    #[must_use]
    pub fn centroid(&self) -> Vec3 {
        let sum = self
            .points
            .iter()
            .map(PointNormal::position)
            .fold(Vec3::zero(), |acc, p| acc + p);
        let n = self.points.len();
        // Safe because PointCloud is always non-empty, and usize->f64
        // is lossless for reasonable cloud sizes.
        #[allow(clippy::cast_precision_loss)]
        let count = n as f64;
        sum.scale(1.0 / count)
    }

    /// Apply a rigid transform to every point, producing a new cloud.
    #[must_use]
    pub fn transformed(&self, transform: &RigidTransform) -> Self {
        Self {
            points: self
                .points
                .iter()
                .map(|p| p.transformed(transform))
                .collect(),
        }
    }

    /// Produce a new cloud with the given normals attached.
    ///
    /// Returns `None` if the number of normals does not match
    /// the number of points.
    #[must_use]
    pub fn with_normals(&self, normals: &[Vec3]) -> Option<Self> {
        if normals.len() == self.points.len() {
            let new_points = self
                .points
                .iter()
                .zip(normals.iter())
                .map(|(p, n)| PointNormal::with_normal(p.position(), *n))
                .collect();
            Some(Self { points: new_points })
        } else {
            None
        }
    }

    /// Find the index and squared distance of the nearest point to `query`.
    ///
    /// Brute-force O(n) scan.  Returns `None` only if the cloud is empty,
    /// which cannot happen for a valid `PointCloud`.
    #[must_use]
    pub fn nearest_neighbor(&self, query: Vec3) -> Option<(usize, f64)> {
        self.points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.position().distance_squared(query)))
            .reduce(|best, candidate| {
                if candidate.1 < best.1 { candidate } else { best }
            })
    }
}

/// A correspondence between a point in a source cloud and a point
/// in a target cloud.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Correspondence {
    source_idx: usize,
    target_idx: usize,
    distance_sq: f64,
}

impl Correspondence {
    /// Construct a new correspondence.
    #[must_use]
    pub fn new(source_idx: usize, target_idx: usize, distance_sq: f64) -> Self {
        Self { source_idx, target_idx, distance_sq }
    }

    /// Index into the source cloud.
    #[must_use]
    pub fn source_idx(&self) -> usize {
        self.source_idx
    }

    /// Index into the target cloud.
    #[must_use]
    pub fn target_idx(&self) -> usize {
        self.target_idx
    }

    /// Squared distance between the paired points.
    #[must_use]
    pub fn distance_sq(&self) -> f64 {
        self.distance_sq
    }
}
