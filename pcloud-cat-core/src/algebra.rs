//! Linear algebra primitives for 3D point cloud processing.
//!
//! All types use private fields with getter methods and associated
//! function constructors.  Implements standard traits (`Add`, `Sub`,
//! `Mul`, `Neg`) where semantically appropriate.
//!
//! Includes a hand-rolled 3x3 SVD (Jacobi rotations) and 3x3
//! symmetric eigendecomposition (analytical/Cardano) to avoid
//! external linear algebra dependencies.

use crate::error::AlgebraErrorKind;

// ── Validated domain primitives ──────────────────────────────────

/// A non-negative `f64`.  Construction fails for negative or NaN values.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct NonNegF64 {
    value: f64,
}

impl NonNegF64 {
    /// Construct from a value, returning `None` if negative or NaN.
    #[must_use]
    pub fn new(value: f64) -> Option<Self> {
        if value >= 0.0 { Some(Self { value }) } else { None }
    }

    /// The contained value.
    #[must_use]
    pub fn value(self) -> f64 {
        self.value
    }
}

/// A positive (nonzero) `usize`.  Construction fails for zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PosUsize {
    value: usize,
}

impl PosUsize {
    /// Construct from a value, returning `None` if zero.
    #[must_use]
    pub fn new(value: usize) -> Option<Self> {
        if value > 0 { Some(Self { value }) } else { None }
    }

    /// The contained value.
    #[must_use]
    pub fn value(self) -> usize {
        self.value
    }
}

// ── Vec3 ─────────────────────────────────────────────────────────

/// A 3D vector, wrapping `[f64; 3]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    coords: [f64; 3],
}

impl Vec3 {
    /// Construct from three components.
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { coords: [x, y, z] }
    }

    /// The zero vector.
    #[must_use]
    pub fn zero() -> Self {
        Self { coords: [0.0; 3] }
    }

    /// X component.
    #[must_use]
    pub fn x(self) -> f64 {
        let [x, _, _] = self.coords;
        x
    }

    /// Y component.
    #[must_use]
    pub fn y(self) -> f64 {
        let [_, y, _] = self.coords;
        y
    }

    /// Z component.
    #[must_use]
    pub fn z(self) -> f64 {
        let [_, _, z] = self.coords;
        z
    }

    /// Access by index (0, 1, 2).
    #[must_use]
    pub fn get(self, i: usize) -> Option<f64> {
        self.coords.get(i).copied()
    }

    /// Dot product.
    #[must_use]
    pub fn dot(self, other: Self) -> f64 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Cross product.
    #[must_use]
    pub fn cross(self, other: Self) -> Self {
        let [ax, ay, az] = self.coords;
        let [bx, by, bz] = other.coords;
        Self {
            coords: [
                ay * bz - az * by,
                az * bx - ax * bz,
                ax * by - ay * bx,
            ],
        }
    }

    /// Euclidean norm (length).
    #[must_use]
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Squared Euclidean norm.
    #[must_use]
    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    /// Unit vector, or `None` if the norm is near zero.
    #[must_use]
    pub fn normalized(self) -> Option<Self> {
        let n = self.norm();
        if n < 1e-15 { None } else { Some(self.scale(1.0 / n)) }
    }

    /// Scalar multiplication.
    #[must_use]
    pub fn scale(self, s: f64) -> Self {
        let [x, y, z] = self.coords;
        Self {
            coords: [x * s, y * s, z * s],
        }
    }

    /// Squared Euclidean distance to another vector.
    #[must_use]
    pub fn distance_squared(self, other: Self) -> f64 {
        (self - other).norm_squared()
    }

    /// Euclidean distance to another vector.
    #[must_use]
    pub fn distance(self, other: Self) -> f64 {
        (self - other).norm()
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let [ax, ay, az] = self.coords;
        let [bx, by, bz] = rhs.coords;
        Self {
            coords: [ax + bx, ay + by, az + bz],
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let [ax, ay, az] = self.coords;
        let [bx, by, bz] = rhs.coords;
        Self {
            coords: [ax - bx, ay - by, az - bz],
        }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self {
        let [x, y, z] = self.coords;
        Self {
            coords: [-x, -y, -z],
        }
    }
}

impl std::fmt::Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [x, y, z] = self.coords;
        write!(f, "({x}, {y}, {z})")
    }
}

// ── Mat3 ─────────────────────────────────────────────────────────

/// A 3x3 matrix stored in column-major order, wrapping `[f64; 9]`.
///
/// Element at row `r`, column `c` is stored at index `c * 3 + r`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    data: [f64; 9],
}

/// Decompose a column-major `[f64; 9]` into nine named elements.
///
/// `(m00, m10, m20, m01, m11, m21, m02, m12, m22)` where `mRC`
/// is row R, column C.
#[must_use]
fn decompose9(data: [f64; 9]) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let [m00, m10, m20, m01, m11, m21, m02, m12, m22] = data;
    (m00, m10, m20, m01, m11, m21, m02, m12, m22)
}

/// Three singular values stored by name rather than index.
#[derive(Debug, Clone, Copy)]
struct Sigma3 {
    s0: f64,
    s1: f64,
    s2: f64,
}

impl Sigma3 {
    #[must_use]
    fn from_array(a: [f64; 3]) -> Self {
        let [s0, s1, s2] = a;
        Self { s0, s1, s2 }
    }

    #[must_use]
    fn to_array(self) -> [f64; 3] {
        [self.s0, self.s1, self.s2]
    }

    /// Retrieve the value at a position described by a [`Slot3`].
    #[must_use]
    fn pick(self, slot: Slot3) -> f64 {
        match slot {
            Slot3::Zero => self.s0,
            Slot3::One => self.s1,
            Slot3::Two => self.s2,
        }
    }

    /// Negate the value at `slot`.
    #[must_use]
    fn negate_at(self, slot: Slot3) -> Self {
        match slot {
            Slot3::Zero => Self { s0: -self.s0, ..self },
            Slot3::One => Self { s1: -self.s1, ..self },
            Slot3::Two => Self { s2: -self.s2, ..self },
        }
    }
}

/// A type-safe index into a 3-element structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot3 {
    Zero,
    One,
    Two,
}

impl Slot3 {
    /// Convert a `usize` in `0..3` to a [`Slot3`], returning `None`
    /// for out-of-range values.
    #[must_use]
    fn from_usize(v: usize) -> Option<Self> {
        match v {
            0 => Some(Self::Zero),
            1 => Some(Self::One),
            2 => Some(Self::Two),
            _ => None,
        }
    }
}

/// A permutation of three slots, describing a reordering.
#[derive(Debug, Clone, Copy)]
struct Perm3 {
    first: Slot3,
    second: Slot3,
    third: Slot3,
}

impl Perm3 {
    /// Apply this permutation to a [`Sigma3`].
    #[must_use]
    fn permute_sigma(self, s: Sigma3) -> Sigma3 {
        Sigma3 {
            s0: s.pick(self.first),
            s1: s.pick(self.second),
            s2: s.pick(self.third),
        }
    }

    /// Apply this permutation to a matrix's columns, producing a new
    /// matrix whose column `i` is the old column `perm[i]`.
    #[must_use]
    fn permute_cols(self, m: Mat3) -> Mat3 {
        let c0 = pick_col(m, self.first);
        let c1 = pick_col(m, self.second);
        let c2 = pick_col(m, self.third);
        Mat3::from_columns(c0, c1, c2)
    }
}

/// Safely pick a column of `m` by [`Slot3`].
#[must_use]
fn pick_col(m: Mat3, slot: Slot3) -> Vec3 {
    match slot {
        Slot3::Zero => m.col0(),
        Slot3::One  => m.col1(),
        Slot3::Two  => m.col2(),
    }
}

impl Mat3 {
    /// The 3x3 identity matrix.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            data: [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
        }
    }

    /// The 3x3 zero matrix.
    #[must_use]
    pub fn zero() -> Self {
        Self { data: [0.0; 9] }
    }

    /// Construct from three column vectors.
    #[must_use]
    pub fn from_columns(c0: Vec3, c1: Vec3, c2: Vec3) -> Self {
        Self {
            data: [
                c0.x(), c0.y(), c0.z(),
                c1.x(), c1.y(), c1.z(),
                c2.x(), c2.y(), c2.z(),
            ],
        }
    }

    /// Element at row `r`, column `c`.
    #[must_use]
    pub fn at(self, r: usize, c: usize) -> Option<f64> {
        if r < 3 && c < 3 {
            self.data.get(c * 3 + r).copied()
        } else {
            None
        }
    }

    /// Column 0 (infallible).
    #[must_use]
    fn col0(self) -> Vec3 {
        let (m00, m10, m20, _, _, _, _, _, _) = decompose9(self.data);
        Vec3::new(m00, m10, m20)
    }

    /// Column 1 (infallible).
    #[must_use]
    fn col1(self) -> Vec3 {
        let (_, _, _, m01, m11, m21, _, _, _) = decompose9(self.data);
        Vec3::new(m01, m11, m21)
    }

    /// Column 2 (infallible).
    #[must_use]
    fn col2(self) -> Vec3 {
        let (_, _, _, _, _, _, m02, m12, m22) = decompose9(self.data);
        Vec3::new(m02, m12, m22)
    }

    /// Column `c` as a [`Vec3`].
    #[must_use]
    pub fn col(self, c: usize) -> Option<Vec3> {
        Slot3::from_usize(c).map(|s| pick_col(self, s))
    }

    /// Row `r` as a [`Vec3`].
    #[must_use]
    pub fn row(self, r: usize) -> Option<Vec3> {
        let (m00, m10, m20, m01, m11, m21, m02, m12, m22) = decompose9(self.data);
        match r {
            0 => Some(Vec3::new(m00, m01, m02)),
            1 => Some(Vec3::new(m10, m11, m12)),
            2 => Some(Vec3::new(m20, m21, m22)),
            _ => None,
        }
    }

    /// Transpose.
    #[must_use]
    pub fn transpose(self) -> Self {
        let (m00, m10, m20, m01, m11, m21, m02, m12, m22) = decompose9(self.data);
        Self {
            data: [
                m00, m01, m02,
                m10, m11, m12,
                m20, m21, m22,
            ],
        }
    }

    /// Matrix-vector product.
    #[must_use]
    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        let (m00, m10, m20, m01, m11, m21, m02, m12, m22) = decompose9(self.data);
        let vx = v.x();
        let vy = v.y();
        let vz = v.z();
        Vec3::new(
            m00 * vx + m01 * vy + m02 * vz,
            m10 * vx + m11 * vy + m12 * vz,
            m20 * vx + m21 * vy + m22 * vz,
        )
    }

    /// Matrix-matrix product.
    #[must_use]
    pub fn mul_mat(self, other: Self) -> Self {
        let c0 = self.mul_vec(other.col0());
        let c1 = self.mul_vec(other.col1());
        let c2 = self.mul_vec(other.col2());
        Self::from_columns(c0, c1, c2)
    }

    /// Determinant.
    #[must_use]
    pub fn determinant(self) -> f64 {
        let (m00, m10, m20, m01, m11, m21, m02, m12, m22) = decompose9(self.data);
        m00 * (m11 * m22 - m21 * m12)
            - m01 * (m10 * m22 - m20 * m12)
            + m02 * (m10 * m21 - m20 * m11)
    }

    /// Scalar multiplication.
    #[must_use]
    pub fn scale(self, s: f64) -> Self {
        Self {
            data: std::array::from_fn(|i| {
                self.data.get(i).copied().unwrap_or(0.0) * s
            }),
        }
    }

    /// Build a matrix from the outer product of two vectors: a * b^T.
    #[must_use]
    pub fn outer(a: Vec3, b: Vec3) -> Self {
        let ax = a.x();
        let ay = a.y();
        let az = a.z();
        let bx = b.x();
        let by = b.y();
        let bz = b.z();
        Self {
            data: [
                ax * bx, ay * bx, az * bx,
                ax * by, ay * by, az * by,
                ax * bz, ay * bz, az * bz,
            ],
        }
    }

    /// Frobenius norm.
    #[must_use]
    pub fn frobenius_norm(self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// The three diagonal elements `(d00, d11, d22)`.
    #[must_use]
    fn diagonal(self) -> (f64, f64, f64) {
        let (m00, _, _, _, m11, _, _, _, m22) = decompose9(self.data);
        (m00, m11, m22)
    }

    /// SVD: decompose into U * diag(sigma) * V^T.
    ///
    /// Uses one-sided Jacobi rotations for 3x3 matrices.
    ///
    /// # Errors
    ///
    /// Returns [`AlgebraErrorKind::SvdNotConverged`] if the iteration
    /// does not converge within the budget.
    pub fn svd(self) -> Result<SvdResult, AlgebraErrorKind> {
        svd_3x3(self)
    }
}

impl std::ops::Mul<Vec3> for Mat3 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        self.mul_vec(rhs)
    }
}

impl std::ops::Mul for Mat3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        self.mul_mat(rhs)
    }
}

impl std::ops::Add for Mat3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            data: std::array::from_fn(|i| {
                let a = self.data.get(i).copied().unwrap_or(0.0);
                let b = rhs.data.get(i).copied().unwrap_or(0.0);
                a + b
            }),
        }
    }
}

impl std::fmt::Display for Mat3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (m00, m10, m20, m01, m11, m21, m02, m12, m22) = decompose9(self.data);
        writeln!(f, "[{m00}, {m01}, {m02}]")?;
        writeln!(f, "[{m10}, {m11}, {m12}]")?;
        writeln!(f, "[{m20}, {m21}, {m22}]")
    }
}

// ── SVD result ───────────────────────────────────────────────────

/// The result of a 3x3 SVD: `U * diag(sigma) * Vt`.
#[derive(Debug, Clone, Copy)]
pub struct SvdResult {
    u: Mat3,
    sigma: [f64; 3],
    vt: Mat3,
}

impl SvdResult {
    /// The left singular vectors (3x3 orthogonal matrix).
    #[must_use]
    pub fn u(&self) -> Mat3 {
        self.u
    }

    /// The singular values (non-negative, descending).
    #[must_use]
    pub fn sigma(&self) -> [f64; 3] {
        self.sigma
    }

    /// The transposed right singular vectors (3x3 orthogonal matrix).
    #[must_use]
    pub fn vt(&self) -> Mat3 {
        self.vt
    }

    /// Reconstruct the original matrix: U * diag(sigma) * Vt.
    #[must_use]
    pub fn reconstruct(&self) -> Mat3 {
        let [s0, s1, s2] = self.sigma;
        let diag = Mat3 {
            data: [
                s0,  0.0, 0.0,
                0.0, s1,  0.0,
                0.0, 0.0, s2,
            ],
        };
        self.u.mul_mat(diag).mul_mat(self.vt)
    }
}

// ── Rigid transform ──────────────────────────────────────────────

/// A rigid (Euclidean) transform: rotation + translation.
///
/// Applies as `R * p + t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidTransform {
    rotation: Mat3,
    translation: Vec3,
}

impl RigidTransform {
    /// The identity transform (no rotation, no translation).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            rotation: Mat3::identity(),
            translation: Vec3::zero(),
        }
    }

    /// Construct from a rotation matrix and translation vector.
    #[must_use]
    pub fn new(rotation: Mat3, translation: Vec3) -> Self {
        Self { rotation, translation }
    }

    /// The rotation component.
    #[must_use]
    pub fn rotation(&self) -> Mat3 {
        self.rotation
    }

    /// The translation component.
    #[must_use]
    pub fn translation(&self) -> Vec3 {
        self.translation
    }

    /// Apply this transform to a point: `R * p + t`.
    #[must_use]
    pub fn apply(&self, point: Vec3) -> Vec3 {
        self.rotation.mul_vec(point) + self.translation
    }

    /// Compose two transforms: first apply `self`, then `other`.
    ///
    /// `(other . self)(p) = other.R * (self.R * p + self.t) + other.t`
    ///                    `= (other.R * self.R) * p + (other.R * self.t + other.t)`
    #[must_use]
    pub fn then(&self, other: &Self) -> Self {
        Self {
            rotation: other.rotation.mul_mat(self.rotation),
            translation: other.rotation.mul_vec(self.translation) + other.translation,
        }
    }

    /// The inverse transform.
    ///
    /// For a rigid transform with rotation R and translation t,
    /// the inverse has rotation R^T and translation -R^T * t.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let rt = self.rotation.transpose();
        Self {
            rotation: rt,
            translation: -rt.mul_vec(self.translation),
        }
    }
}

// ── 3x3 SVD via one-sided Jacobi rotations ──────────────────────

/// A column-pair designation for Givens rotations on a 3x3 matrix.
///
/// Encodes which two of the three columns participate in the rotation,
/// keeping all index arithmetic inside a single, bounded `match`.
#[derive(Debug, Clone, Copy)]
enum ColPair {
    /// Columns 0 and 1.
    C01,
    /// Columns 0 and 2.
    C02,
    /// Columns 1 and 2.
    C12,
}

impl ColPair {
    /// The (p, q) column indices.
    #[must_use]
    fn indices(self) -> (usize, usize) {
        match self {
            Self::C01 => (0, 1),
            Self::C02 => (0, 2),
            Self::C12 => (1, 2),
        }
    }
}

const ALL_PAIRS: [ColPair; 3] = [ColPair::C01, ColPair::C02, ColPair::C12];

/// Read element at (row, col) from a column-major `[f64; 9]`, returning
/// `0.0` for any out-of-range access (which should never happen in
/// practice because all callers use bounded indices).
#[must_use]
fn cm_get(d: &[f64; 9], row: usize, col: usize) -> f64 {
    d.get(col * 3 + row).copied().unwrap_or(0.0)
}

/// Apply a Givens rotation on the right to columns `pair.p` and `pair.q`.
#[must_use]
#[allow(clippy::many_single_char_names)]
fn apply_right_givens(m: Mat3, pair: ColPair, cos: f64, sin: f64) -> Mat3 {
    let (p, q) = pair.indices();
    let new_data: [f64; 9] = std::array::from_fn(|i| {
        let r = i % 3;
        let c = i / 3;
        match (c == p, c == q) {
            (true, false) => {
                let mp = cm_get(&m.data, r, p);
                let mq = cm_get(&m.data, r, q);
                mp * cos + mq * sin
            }
            (false, true) => {
                let mp = cm_get(&m.data, r, p);
                let mq = cm_get(&m.data, r, q);
                -mp * sin + mq * cos
            }
            _ => cm_get(&m.data, r, c),
        }
    });
    Mat3 { data: new_data }
}

/// Apply a Givens rotation on the right to a matrix V (accumulating rotations).
#[must_use]
fn accumulate_right_givens(v: Mat3, pair: ColPair, cos: f64, sin: f64) -> Mat3 {
    apply_right_givens(v, pair, cos, sin)
}

/// Compute Givens rotation to zero out the `(col_p, col_q)` element of A^T * A.
///
/// Returns (cos, sin) of the rotation angle.
#[must_use]
#[allow(clippy::many_single_char_names)]
fn jacobi_rotation_params(ata: Mat3, pair: ColPair) -> (f64, f64) {
    let (p, q) = pair.indices();
    let app = cm_get(&ata.data, p, p);
    let aqq = cm_get(&ata.data, q, q);
    let apq = cm_get(&ata.data, p, q);

    if apq.abs() < 1e-15 {
        (1.0, 0.0)
    } else {
        let tau = (aqq - app) / (2.0 * apq);
        let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
        let t = sign / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        (c, s)
    }
}

/// Compute the off-diagonal Frobenius norm of a symmetric matrix
/// (measure of how close to diagonal it is).
#[must_use]
fn off_diag_norm(m: Mat3) -> f64 {
    let (_, _, _, a01, _, _, a02, a12, _) = decompose9(m.data);
    (2.0 * (a01 * a01 + a02 * a02 + a12 * a12)).sqrt()
}

/// 3x3 SVD via one-sided Jacobi rotations.
///
/// Decomposes A into U * diag(sigma) * V^T where U and V are orthogonal
/// and sigma contains non-negative singular values in descending order.
fn svd_3x3(a: Mat3) -> Result<SvdResult, AlgebraErrorKind> {
    const MAX_SWEEPS: usize = 100;
    const TOL: f64 = 1e-14;

    // We iteratively diagonalize A^T * A via Jacobi rotations.
    // Each sweep applies rotations to the three off-diagonal pairs.
    let (final_b, final_v) = (0..MAX_SWEEPS).try_fold(
        (a, Mat3::identity()),
        |(b, v), _sweep| {
            let btb = b.transpose().mul_mat(b);
            if off_diag_norm(btb) < TOL {
                Err((b, v)) // converged, break out via Err
            } else {
                // Sweep: rotate all three column-pairs
                let (b2, v2) = ALL_PAIRS.iter().fold((b, v), |(bi, vi), &pair| {
                    let btb_i = bi.transpose().mul_mat(bi);
                    let (cos, sin) = jacobi_rotation_params(btb_i, pair);
                    let bi_new = apply_right_givens(bi, pair, cos, sin);
                    let vi_new = accumulate_right_givens(vi, pair, cos, sin);
                    (bi_new, vi_new)
                });
                Ok((b2, v2))
            }
        },
    )
    .unwrap_or_else(|converged| converged);

    // Check convergence: if we exhausted sweeps without converging
    let btb_final = final_b.transpose().mul_mat(final_b);
    if off_diag_norm(btb_final) >= TOL * 1000.0 {
        Err(AlgebraErrorKind::SvdNotConverged)?;
    }

    // Now B = A * V is approximately column-orthogonal.
    // The singular values are the norms of B's columns.
    // U is obtained by normalizing B's columns.
    let bc0 = final_b.col0();
    let bc1 = final_b.col1();
    let bc2 = final_b.col2();
    let sigma = [bc0.norm(), bc1.norm(), bc2.norm()];

    let u0 = bc0.normalized().unwrap_or_else(|| Vec3::new(1.0, 0.0, 0.0));
    let u1 = bc1.normalized().unwrap_or_else(|| Vec3::new(0.0, 1.0, 0.0));
    let u2 = bc2.normalized().unwrap_or_else(|| Vec3::new(0.0, 0.0, 1.0));
    let u = Mat3::from_columns(u0, u1, u2);

    // Ensure U and V have determinant +1 (proper rotations).
    let (u_fixed, sigma_fixed, v_fixed) = fix_svd_signs(u, Sigma3::from_array(sigma), final_v);

    // Sort singular values in descending order.
    let sorted = sort_svd_descending(u_fixed, sigma_fixed, v_fixed);

    Ok(sorted)
}

/// Ensure U and V both have determinant +1 by flipping the sign of
/// the last column if needed.
#[must_use]
fn fix_svd_signs(u: Mat3, sigma: Sigma3, v: Mat3) -> (Mat3, Sigma3, Mat3) {
    let u_det = u.determinant();
    let v_det = v.determinant();

    let (u2, s2) = if u_det < 0.0 {
        // Flip last column of U and negate last singular value
        let c0 = u.col0();
        let c1 = u.col1();
        let c2 = -u.col2();
        (Mat3::from_columns(c0, c1, c2), sigma.negate_at(Slot3::Two))
    } else {
        (u, sigma)
    };

    let v2 = if v_det < 0.0 {
        let c0 = v.col0();
        let c1 = v.col1();
        let c2 = -v.col2();
        Mat3::from_columns(c0, c1, c2)
    } else {
        v
    };

    // Make all singular values non-negative
    let slots = [Slot3::Zero, Slot3::One, Slot3::Two];
    let (u_final, sigma_final) = slots.iter().fold((u2, s2), |(ui, si), &slot| {
        if si.pick(slot) < 0.0 {
            // Flip the column of U at this slot and negate the sigma entry
            let c0 = ui.col0();
            let c1 = ui.col1();
            let c2 = ui.col2();
            let flipped = match slot {
                Slot3::Zero => Mat3::from_columns(-c0, c1, c2),
                Slot3::One  => Mat3::from_columns(c0, -c1, c2),
                Slot3::Two  => Mat3::from_columns(c0, c1, -c2),
            };
            (flipped, si.negate_at(slot))
        } else {
            (ui, si)
        }
    });

    (u_final, sigma_final, v2)
}

/// Sort three f64 values descending, returning a [`Perm3`] that
/// describes how the original indices map to the sorted order.
#[must_use]
fn sort3_descending(a: f64, b: f64, c: f64) -> Perm3 {
    if a >= b && a >= c {
        if b >= c {
            Perm3 { first: Slot3::Zero, second: Slot3::One,  third: Slot3::Two }
        } else {
            Perm3 { first: Slot3::Zero, second: Slot3::Two,  third: Slot3::One }
        }
    } else if b >= a && b >= c {
        if a >= c {
            Perm3 { first: Slot3::One,  second: Slot3::Zero, third: Slot3::Two }
        } else {
            Perm3 { first: Slot3::One,  second: Slot3::Two,  third: Slot3::Zero }
        }
    } else if a >= b {
        Perm3 { first: Slot3::Two, second: Slot3::Zero, third: Slot3::One }
    } else {
        Perm3 { first: Slot3::Two, second: Slot3::One,  third: Slot3::Zero }
    }
}

/// Sort three f64 values ascending, returning a [`Perm3`].
#[must_use]
fn sort3_ascending(a: f64, b: f64, c: f64) -> Perm3 {
    if a <= b && a <= c {
        if b <= c {
            Perm3 { first: Slot3::Zero, second: Slot3::One,  third: Slot3::Two }
        } else {
            Perm3 { first: Slot3::Zero, second: Slot3::Two,  third: Slot3::One }
        }
    } else if b <= a && b <= c {
        if a <= c {
            Perm3 { first: Slot3::One,  second: Slot3::Zero, third: Slot3::Two }
        } else {
            Perm3 { first: Slot3::One,  second: Slot3::Two,  third: Slot3::Zero }
        }
    } else if a <= b {
        Perm3 { first: Slot3::Two, second: Slot3::Zero, third: Slot3::One }
    } else {
        Perm3 { first: Slot3::Two, second: Slot3::One,  third: Slot3::Zero }
    }
}

/// Sort SVD components so singular values are in descending order.
#[must_use]
fn sort_svd_descending(u: Mat3, sigma: Sigma3, v: Mat3) -> SvdResult {
    let perm = sort3_descending(sigma.s0, sigma.s1, sigma.s2);

    let sorted_sigma = perm.permute_sigma(sigma);
    let sorted_u = perm.permute_cols(u);
    let sorted_v = perm.permute_cols(v);

    SvdResult {
        u: sorted_u,
        sigma: sorted_sigma.to_array(),
        vt: sorted_v.transpose(),
    }
}

// ── 3x3 symmetric eigendecomposition ─────────────────────────────

/// Result of eigendecomposition of a 3x3 symmetric matrix.
#[derive(Debug, Clone, Copy)]
pub struct Eigen3Result {
    eigenvalues: [f64; 3],
    eigenvectors: Mat3,
}

impl Eigen3Result {
    /// Eigenvalues in ascending order.
    #[must_use]
    pub fn eigenvalues(&self) -> [f64; 3] {
        self.eigenvalues
    }

    /// Eigenvectors as columns of a 3x3 matrix, matching the order
    /// of [`eigenvalues`](Self::eigenvalues).
    #[must_use]
    pub fn eigenvectors(&self) -> Mat3 {
        self.eigenvectors
    }
}

/// Eigendecomposition of a 3x3 symmetric matrix via Jacobi iteration.
///
/// Returns eigenvalues in ascending order with corresponding eigenvectors.
///
/// # Errors
///
/// Returns [`AlgebraErrorKind::SvdNotConverged`] if the iteration
/// does not converge.
pub fn symmetric_eigen_3x3(m: Mat3) -> Result<Eigen3Result, AlgebraErrorKind> {
    const MAX_SWEEPS: usize = 100;
    const TOL: f64 = 1e-14;

    let (final_d, final_v) = (0..MAX_SWEEPS).try_fold(
        (m, Mat3::identity()),
        |(d, v), _sweep| {
            if off_diag_norm(d) < TOL {
                Err((d, v))
            } else {
                let (d2, v2) = ALL_PAIRS.iter().fold((d, v), |(di, vi), &pair| {
                    let (cos, sin) = jacobi_rotation_params(di, pair);
                    // D' = G^T * D * G (similarity transform)
                    let dg = apply_right_givens(di, pair, cos, sin);
                    let di_new = apply_left_givens(dg, pair, cos, sin);
                    let vi_new = accumulate_right_givens(vi, pair, cos, sin);
                    (di_new, vi_new)
                });
                Ok((d2, v2))
            }
        },
    )
    .unwrap_or_else(|converged| converged);

    if off_diag_norm(final_d) >= TOL * 1000.0 {
        Err(AlgebraErrorKind::SvdNotConverged)?;
    }

    // Extract eigenvalues from the diagonal
    let (e0, e1, e2) = final_d.diagonal();
    let eigs = Sigma3 { s0: e0, s1: e1, s2: e2 };

    // Sort ascending
    let perm = sort3_ascending(e0, e1, e2);
    let sorted_eigs = perm.permute_sigma(eigs);
    let sorted_vecs = perm.permute_cols(final_v);

    Ok(Eigen3Result {
        eigenvalues: sorted_eigs.to_array(),
        eigenvectors: sorted_vecs,
    })
}

/// Apply a Givens rotation on the left to rows `pair.p` and `pair.q`.
#[must_use]
#[allow(clippy::many_single_char_names)]
fn apply_left_givens(m: Mat3, pair: ColPair, cos: f64, sin: f64) -> Mat3 {
    let (p, q) = pair.indices();
    let new_data: [f64; 9] = std::array::from_fn(|i| {
        let r = i % 3;
        let c = i / 3;
        match (r == p, r == q) {
            (true, false) => {
                let mp = cm_get(&m.data, p, c);
                let mq = cm_get(&m.data, q, c);
                cos * mp + sin * mq
            }
            (false, true) => {
                let mp = cm_get(&m.data, p, c);
                let mq = cm_get(&m.data, q, c);
                -sin * mp + cos * mq
            }
            _ => cm_get(&m.data, r, c),
        }
    });
    Mat3 { data: new_data }
}
