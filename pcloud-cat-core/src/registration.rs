//! Point cloud registration via the Iterative Closest Point (ICP) algorithm.
//!
//! ICP is modeled as a [`Stream::unfold`]: each pull computes one
//! iteration (find correspondences, solve for optimal rigid transform,
//! update state).  The stream terminates when convergence is reached
//! or the iteration budget is exhausted.
//!
//! [`icp_align`] folds the stream to produce a single
//! [`Io<Error, RigidTransform>`] representing the full alignment.

use std::rc::Rc;

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;

use crate::algebra::{Mat3, NonNegF64, PosUsize, RigidTransform, Vec3};
use crate::cloud::{Correspondence, PointCloud, PointNormal};
use crate::error::Error;

/// Which ICP error metric to minimize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcpVariant {
    /// Minimize sum of squared point-to-point distances.
    PointToPoint,
    /// Minimize sum of squared point-to-plane distances (requires
    /// normals on the target cloud).
    PointToPlane,
}

/// Configuration for the ICP algorithm.
#[derive(Debug, Clone, Copy)]
pub struct IcpConfig {
    variant: IcpVariant,
    max_iterations: PosUsize,
    convergence_threshold: NonNegF64,
    max_correspondence_distance: Option<NonNegF64>,
}

impl IcpConfig {
    /// Construct a new configuration.
    #[must_use]
    pub fn new(
        variant: IcpVariant,
        max_iterations: PosUsize,
        convergence_threshold: NonNegF64,
    ) -> Self {
        Self {
            variant,
            max_iterations,
            convergence_threshold,
            max_correspondence_distance: None,
        }
    }

    /// Set a maximum correspondence distance for outlier rejection.
    #[must_use]
    pub fn with_max_distance(self, d: NonNegF64) -> Self {
        Self {
            max_correspondence_distance: Some(d),
            ..self
        }
    }

    /// The ICP variant (point-to-point or point-to-plane).
    #[must_use]
    pub fn variant(self) -> IcpVariant {
        self.variant
    }

    /// Maximum number of iterations.
    #[must_use]
    pub fn max_iterations(self) -> PosUsize {
        self.max_iterations
    }

    /// Convergence threshold on delta-RMSE between steps.
    #[must_use]
    pub fn convergence_threshold(self) -> NonNegF64 {
        self.convergence_threshold
    }

    /// Maximum allowed correspondence distance (squared), if set.
    #[must_use]
    pub fn max_correspondence_distance(self) -> Option<NonNegF64> {
        self.max_correspondence_distance
    }
}

/// The state emitted at each ICP iteration.
#[derive(Debug, Clone)]
pub struct IcpStep {
    iteration: usize,
    current_transform: RigidTransform,
    rmse: f64,
    delta_rmse: f64,
    correspondences_count: usize,
}

impl IcpStep {
    /// The iteration number (0-based).
    #[must_use]
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// The accumulated rigid transform at this step.
    #[must_use]
    pub fn current_transform(&self) -> RigidTransform {
        self.current_transform
    }

    /// The root-mean-square error of correspondences at this step.
    #[must_use]
    pub fn rmse(&self) -> f64 {
        self.rmse
    }

    /// The change in RMSE from the previous step.
    #[must_use]
    pub fn delta_rmse(&self) -> f64 {
        self.delta_rmse
    }

    /// The number of correspondences used at this step.
    #[must_use]
    pub fn correspondences_count(&self) -> usize {
        self.correspondences_count
    }
}

/// Internal state threaded through `Stream::unfold`.
struct IcpState {
    source: PointCloud,
    target: PointCloud,
    config: IcpConfig,
    current_transform: RigidTransform,
    prev_rmse: f64,
    iteration: usize,
}

/// Produce a stream of ICP iteration steps.
///
/// Each pull from the stream computes one ICP iteration.  The stream
/// terminates when convergence is reached (delta-RMSE below threshold)
/// or the iteration budget is exhausted.
///
/// # Examples
///
/// ```rust,ignore
/// use std::rc::Rc;
///
/// let steps: Vec<IcpStep> = icp_stream(source, target, config)
///     .take(10)
///     .collect()
///     .run()?;
/// ```
#[must_use]
pub fn icp_stream(
    source: PointCloud,
    target: PointCloud,
    config: IcpConfig,
) -> Stream<Error, IcpStep> {
    let init = IcpState {
        source,
        target,
        config,
        current_transform: RigidTransform::identity(),
        prev_rmse: f64::MAX,
        iteration: 0,
    };

    Stream::unfold(
        init,
        Rc::new(|state: IcpState| {
            Io::suspend(move || icp_step(state))
        }),
    )
}

/// Run ICP to convergence, returning the final rigid transform.
///
/// This folds the ICP stream, staying inside [`Io`] until `.run()`
/// is called at the boundary.
///
/// # Errors
///
/// Returns [`Error::Algebra`] if the SVD fails, or
/// [`Error::InsufficientPoints`] if the clouds are too small
/// for correspondence matching.
///
/// # Examples
///
/// ```rust,ignore
/// let transform: Result<RigidTransform, Error> =
///     icp_align(source, target, config).run();
/// ```
#[must_use]
pub fn icp_align(
    source: PointCloud,
    target: PointCloud,
    config: IcpConfig,
) -> Io<Error, RigidTransform> {
    icp_stream(source, target, config).fold(
        RigidTransform::identity(),
        Rc::new(|_acc, step: IcpStep| step.current_transform()),
    )
}

/// One step of the ICP algorithm.
///
/// Returns `None` to terminate the stream (converged or budget exhausted),
/// or `Some((step, next_state))` to continue.
fn icp_step(state: IcpState) -> Result<Option<(IcpStep, IcpState)>, Error> {
    if state.iteration >= state.config.max_iterations.value() {
        Ok(None)
    } else {
        // 1. Transform source by current accumulated transform.
        let transformed = state.source.transformed(&state.current_transform);

        // 2. Find correspondences.
        let correspondences = find_correspondences(
            &transformed,
            &state.target,
            state.config.max_correspondence_distance,
        );

        if correspondences.is_empty() {
            Ok(None)
        } else {
            // 3. Compute optimal incremental transform.
            let increment = match state.config.variant {
                IcpVariant::PointToPoint => {
                    solve_point_to_point(&transformed, &state.target, &correspondences)?
                }
                IcpVariant::PointToPlane => {
                    solve_point_to_plane(&transformed, &state.target, &correspondences)?
                }
            };

            // 4. Accumulate: new_transform = increment ∘ current_transform
            let new_transform = state.current_transform.then(&increment);

            // 5. Compute RMSE.
            let rmse = compute_rmse(&correspondences);
            let delta_rmse = (state.prev_rmse - rmse).abs();

            let step = IcpStep {
                iteration: state.iteration,
                current_transform: new_transform,
                rmse,
                delta_rmse,
                correspondences_count: correspondences.len(),
            };

            // 6. Check convergence.
            if delta_rmse < state.config.convergence_threshold.value() && state.iteration > 0 {
                // Emit this final step, then signal termination on next pull.
                let next_state = IcpState {
                    iteration: state.config.max_iterations.value(), // exhaust budget
                    current_transform: new_transform,
                    prev_rmse: rmse,
                    ..state
                };
                Ok(Some((step, next_state)))
            } else {
                let next_state = IcpState {
                    iteration: state.iteration + 1,
                    current_transform: new_transform,
                    prev_rmse: rmse,
                    ..state
                };
                Ok(Some((step, next_state)))
            }
        }
    }
}

/// Find nearest-neighbor correspondences from source to target.
fn find_correspondences(
    source: &PointCloud,
    target: &PointCloud,
    max_dist: Option<NonNegF64>,
) -> Vec<Correspondence> {
    let max_dist_sq = max_dist.map(|d| d.value() * d.value());

    source
        .points()
        .iter()
        .enumerate()
        .filter_map(|(si, sp)| {
            target.nearest_neighbor(sp.position()).and_then(|(ti, dist_sq)| {
                max_dist_sq.map_or(
                    Some(Correspondence::new(si, ti, dist_sq)),
                    |max| {
                        if dist_sq <= max {
                            Some(Correspondence::new(si, ti, dist_sq))
                        } else {
                            None
                        }
                    },
                )
            })
        })
        .collect()
}

/// Solve for the optimal rigid transform (point-to-point).
///
/// Uses the SVD-based closed-form solution:
///   1. Compute centroids of matched subsets.
///   2. Build cross-covariance matrix H.
///   3. SVD of H -> rotation.
///   4. Translation from centroids.
fn solve_point_to_point(
    source: &PointCloud,
    target: &PointCloud,
    correspondences: &[Correspondence],
) -> Result<RigidTransform, Error> {
    let source_pts = source.points();
    let target_pts = target.points();

    #[allow(clippy::cast_precision_loss)]
    let n = correspondences.len() as f64;

    // Centroids of matched subsets.
    let (src_centroid, tgt_centroid) = correspondences.iter().fold(
        (Vec3::zero(), Vec3::zero()),
        |(sa, ta), c| {
            let sp = source_pts.get(c.source_idx()).map_or(Vec3::zero(), PointNormal::position);
            let tp = target_pts.get(c.target_idx()).map_or(Vec3::zero(), PointNormal::position);
            (sa + sp, ta + tp)
        },
    );
    let src_centroid = src_centroid.scale(1.0 / n);
    let tgt_centroid = tgt_centroid.scale(1.0 / n);

    // Cross-covariance matrix H = sum( (s_i - cs) * (t_i - ct)^T )
    let h = correspondences.iter().fold(Mat3::zero(), |acc, c| {
        let sp = source_pts.get(c.source_idx()).map_or(Vec3::zero(), PointNormal::position);
        let tp = target_pts.get(c.target_idx()).map_or(Vec3::zero(), PointNormal::position);
        let ds = sp - src_centroid;
        let dt = tp - tgt_centroid;
        acc + Mat3::outer(ds, dt)
    });

    // SVD of H
    let svd = h.svd().map_err(crate::error::Error::from)?;

    // R = V * U^T, with determinant correction
    let v = svd.vt().transpose();
    let ut = svd.u().transpose();
    let r = v * ut;

    // Correct for reflection
    let rotation = if r.determinant() < 0.0 {
        // Flip sign of last column of V
        let v0 = v.col(0).unwrap_or_else(Vec3::zero);
        let v1 = v.col(1).unwrap_or_else(Vec3::zero);
        let v2 = v.col(2).map_or_else(Vec3::zero, |c| -c);
        Mat3::from_columns(v0, v1, v2) * ut
    } else {
        r
    };

    let translation = tgt_centroid - rotation.mul_vec(src_centroid);

    Ok(RigidTransform::new(rotation, translation))
}

/// Solve for the optimal rigid transform (point-to-plane).
///
/// Uses the linearized small-angle approximation to minimize
/// `sum( ((R*s_i + t - t_i) . n_i)^2 )`.
/// Falls back to point-to-point if target normals are missing.
fn solve_point_to_plane(
    source: &PointCloud,
    target: &PointCloud,
    correspondences: &[Correspondence],
) -> Result<RigidTransform, Error> {
    let target_pts = target.points();

    // Check if target has normals; if not, fall back.
    let all_have_normals = correspondences.iter().all(|c| {
        target_pts.get(c.target_idx()).and_then(PointNormal::normal).is_some()
    });

    if all_have_normals {
        solve_point_to_plane_impl(source, target, correspondences)
    } else {
        solve_point_to_point(source, target, correspondences)
    }
}

/// Actual point-to-plane solver (assumes all target correspondences have normals).
///
/// Builds and solves a 6x6 linear system for small rotation angles
/// (alpha, beta, gamma) and translation (tx, ty, tz).
fn solve_point_to_plane_impl(
    source: &PointCloud,
    target: &PointCloud,
    correspondences: &[Correspondence],
) -> Result<RigidTransform, Error> {
    let source_pts = source.points();
    let target_pts = target.points();

    // Build the 6x6 normal equation: A^T A x = A^T b
    // where each correspondence contributes one row to A and b.
    // Row: [n x s, n] . [alpha, beta, gamma, tx, ty, tz]^T = n . (t - s)
    //
    // We accumulate A^T A (6x6) and A^T b (6x1) directly.
    let (normal_matrix, normal_rhs) = correspondences.iter().fold(
        ([[0.0_f64; 6]; 6], [0.0_f64; 6]),
        |(mat_acc, rhs_acc), c| {
            let s = source_pts.get(c.source_idx()).map_or(Vec3::zero(), PointNormal::position);
            let t = target_pts.get(c.target_idx()).map_or(Vec3::zero(), PointNormal::position);
            let n = target_pts.get(c.target_idx()).and_then(PointNormal::normal).unwrap_or_else(Vec3::zero);

            // Cross product n x s
            let cn = n.cross(s);
            let a_row = [cn.x(), cn.y(), cn.z(), n.x(), n.y(), n.z()];
            let b_val = n.dot(t - s);

            // Accumulate outer product a_row * a_row^T
            let new_mat: [[f64; 6]; 6] = std::array::from_fn(|i| {
                std::array::from_fn(|j| mat_acc[i][j] + a_row[i] * a_row[j])
            });

            // Accumulate a_row * b_val
            let new_rhs: [f64; 6] = std::array::from_fn(|i| rhs_acc[i] + a_row[i] * b_val);

            (new_mat, new_rhs)
        },
    );

    // Solve 6x6 system via Gaussian elimination (no external deps).
    let x = solve_6x6(&normal_matrix, normal_rhs)?;

    // Recover rotation from small angles.
    let (alpha, beta, gamma) = (x[0], x[1], x[2]);
    let translation = Vec3::new(x[3], x[4], x[5]);

    // Small-angle rotation matrix:
    // R ≈ I + [gamma, -beta; -gamma, alpha; beta, -alpha, 0] (skew-symmetric)
    let rotation = Mat3::from_columns(
        Vec3::new(1.0, gamma, -beta),
        Vec3::new(-gamma, 1.0, alpha),
        Vec3::new(beta, -alpha, 1.0),
    );

    // Orthonormalize via polar decomposition: R = U * V^T from SVD(R)
    let svd = rotation.svd().map_err(crate::error::Error::from)?;
    let orthonormal_rotation = svd.u() * svd.vt();

    Ok(RigidTransform::new(orthonormal_rotation, translation))
}

/// Solve a 6x6 linear system Ax = b via Gaussian elimination with partial pivoting.
///
/// Returns the solution vector or an error if the system is singular.
fn solve_6x6(a: &[[f64; 6]; 6], b: [f64; 6]) -> Result<[f64; 6], Error> {
    // Augmented matrix: 6 rows of 7 columns
    let augmented: [[f64; 7]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| if j < 6 { a[i][j] } else { b[i] })
    });

    // Forward elimination with partial pivoting (via try_fold)
    let reduced = (0..6).try_fold(augmented, |aug, col| {
        // Find pivot row
        let pivot_row = (col..6).fold(col, |best, row| {
            if aug[row][col].abs() > aug[best][col].abs() {
                row
            } else {
                best
            }
        });

        if aug[pivot_row][col].abs() < 1e-15 {
            Err(Error::Algebra(crate::error::AlgebraErrorKind::SingularMatrix))
        } else {
            // Swap rows
            let swapped: [[f64; 7]; 6] = std::array::from_fn(|i| match () {
                () if i == col => aug[pivot_row],
                () if i == pivot_row => aug[col],
                () => aug[i],
            });

            // Eliminate below pivot
            let eliminated: [[f64; 7]; 6] = std::array::from_fn(|i| {
                if i <= col {
                    swapped[i]
                } else {
                    let factor = swapped[i][col] / swapped[col][col];
                    std::array::from_fn(|j| swapped[i][j] - factor * swapped[col][j])
                }
            });

            Ok(eliminated)
        }
    });

    // Back substitution
    reduced.map(|aug| {
        (0..6).rev().fold([0.0_f64; 6], |x, i| {
            let sum = ((i + 1)..6).fold(0.0, |acc, j| acc + aug[i][j] * x[j]);
            let xi = (aug[i][6] - sum) / aug[i][i];
            let new_x: [f64; 6] = std::array::from_fn(|j| if j == i { xi } else { x[j] });
            new_x
        })
    })
}

/// Compute the RMSE from a set of correspondences.
fn compute_rmse(correspondences: &[Correspondence]) -> f64 {
    if correspondences.is_empty() {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let n = correspondences.len() as f64;
        let sum_sq: f64 = correspondences.iter().map(Correspondence::distance_sq).sum();
        (sum_sq / n).sqrt()
    }
}
