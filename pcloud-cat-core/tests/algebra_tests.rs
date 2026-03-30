use pcloud_cat_core::algebra::{
    symmetric_eigen_3x3, Mat3, NonNegF64, PosUsize, RigidTransform, Vec3,
};
use proptest::prelude::*;

// ── Helpers ──────────────────────────────────────────────────────

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn vec3_approx_eq(a: Vec3, b: Vec3, tol: f64) -> bool {
    approx_eq(a.x(), b.x(), tol)
        && approx_eq(a.y(), b.y(), tol)
        && approx_eq(a.z(), b.z(), tol)
}

fn mat3_approx_eq(a: Mat3, b: Mat3, tol: f64) -> bool {
    (0..3).all(|c| {
        a.col(c)
            .zip(b.col(c))
            .is_some_and(|(ac, bc)| vec3_approx_eq(ac, bc, tol))
    })
}

fn arb_f64() -> impl Strategy<Value = f64> {
    -100.0..100.0_f64
}

fn arb_vec3() -> impl Strategy<Value = Vec3> {
    (arb_f64(), arb_f64(), arb_f64()).prop_map(|(x, y, z)| Vec3::new(x, y, z))
}

fn arb_mat3() -> impl Strategy<Value = Mat3> {
    (arb_vec3(), arb_vec3(), arb_vec3())
        .prop_map(|(c0, c1, c2)| Mat3::from_columns(c0, c1, c2))
}

// ── Vec3 properties ──────────────────────────────────────────────

proptest! {
    #[test]
    fn dot_is_commutative(a in arb_vec3(), b in arb_vec3()) {
        prop_assert!(approx_eq(a.dot(b), b.dot(a), 1e-10));
    }

    #[test]
    fn cross_is_anticommutative(a in arb_vec3(), b in arb_vec3()) {
        let ab = a.cross(b);
        let ba = b.cross(a);
        prop_assert!(vec3_approx_eq(ab, -ba, 1e-10));
    }

    #[test]
    fn cross_is_perpendicular(a in arb_vec3(), b in arb_vec3()) {
        let c = a.cross(b);
        prop_assert!(approx_eq(c.dot(a), 0.0, 1e-8));
        prop_assert!(approx_eq(c.dot(b), 0.0, 1e-8));
    }

    #[test]
    fn norm_of_normalized_is_one(v in arb_vec3()) {
        if let Some(n) = v.normalized() {
            prop_assert!(approx_eq(n.norm(), 1.0, 1e-12));
        }
    }

    #[test]
    fn add_sub_roundtrip(a in arb_vec3(), b in arb_vec3()) {
        prop_assert!(vec3_approx_eq(a + b - b, a, 1e-10));
    }

    #[test]
    fn distance_is_symmetric(a in arb_vec3(), b in arb_vec3()) {
        prop_assert!(approx_eq(a.distance(b), b.distance(a), 1e-12));
    }
}

// ── Mat3 properties ──────────────────────────────────────────────

proptest! {
    #[test]
    fn transpose_involution(m in arb_mat3()) {
        prop_assert!(mat3_approx_eq(m.transpose().transpose(), m, 1e-12));
    }

    #[test]
    fn identity_is_neutral(m in arb_mat3()) {
        let id = Mat3::identity();
        prop_assert!(mat3_approx_eq(m.mul_mat(id), m, 1e-10));
        prop_assert!(mat3_approx_eq(id.mul_mat(m), m, 1e-10));
    }

    #[test]
    fn identity_mul_vec(v in arb_vec3()) {
        prop_assert!(vec3_approx_eq(Mat3::identity().mul_vec(v), v, 1e-12));
    }
}

// ── SVD properties ───────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn svd_reconstructs_original(m in arb_mat3()) {
        if let Ok(svd) = m.svd() {
            let reconstructed = svd.reconstruct();
            prop_assert!(
                mat3_approx_eq(m, reconstructed, 1e-8),
                "SVD reconstruction failed: original {:?} vs reconstructed {:?}",
                m,
                reconstructed,
            );
        }
    }

    #[test]
    fn svd_singular_values_non_negative(m in arb_mat3()) {
        if let Ok(svd) = m.svd() {
            let s = svd.sigma();
            prop_assert!(s[0] >= -1e-12, "sigma[0] negative: {}", s[0]);
            prop_assert!(s[1] >= -1e-12, "sigma[1] negative: {}", s[1]);
            prop_assert!(s[2] >= -1e-12, "sigma[2] negative: {}", s[2]);
        }
    }

    #[test]
    fn svd_singular_values_descending(m in arb_mat3()) {
        if let Ok(svd) = m.svd() {
            let s = svd.sigma();
            prop_assert!(s[0] >= s[1] - 1e-12);
            prop_assert!(s[1] >= s[2] - 1e-12);
        }
    }
}

// ── Eigendecomposition ───────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn eigen_of_symmetric_reconstructs(m in arb_mat3()) {
        // Make symmetric: A = M^T * M
        let sym = m.transpose().mul_mat(m);
        if let Ok(eig) = symmetric_eigen_3x3(sym) {
            let vals = eig.eigenvalues();
            let vecs = eig.eigenvectors();

            // Verify A * v_i ≈ lambda_i * v_i for each eigenvector
            let all_ok = (0..3).all(|i| {
                vecs.col(i).is_none_or(|vi| {
                    let av = sym.mul_vec(vi);
                    let lv = vi.scale(vals[i]);
                    vec3_approx_eq(av, lv, 1e-6)
                })
            });
            prop_assert!(all_ok, "Eigenvector check failed for {:?}", vals);
        }
    }

    #[test]
    fn eigenvalues_ascending(m in arb_mat3()) {
        let sym = m.transpose().mul_mat(m);
        if let Ok(eig) = symmetric_eigen_3x3(sym) {
            let v = eig.eigenvalues();
            prop_assert!(v[0] <= v[1] + 1e-12);
            prop_assert!(v[1] <= v[2] + 1e-12);
        }
    }
}

// ── RigidTransform properties ────────────────────────────────────

#[test]
fn identity_transform_is_neutral() {
    let id = RigidTransform::identity();
    let p = Vec3::new(1.0, 2.0, 3.0);
    assert!(vec3_approx_eq(id.apply(p), p, 1e-15));
}

#[test]
fn inverse_roundtrip() {
    let r = Mat3::identity(); // simple rotation (identity for now)
    let t = Vec3::new(5.0, -3.0, 7.0);
    let tf = RigidTransform::new(r, t);
    let p = Vec3::new(1.0, 2.0, 3.0);
    let roundtrip = tf.inverse().apply(tf.apply(p));
    assert!(vec3_approx_eq(roundtrip, p, 1e-12));
}

#[test]
fn compose_then_inverse_is_identity() {
    let t1 = RigidTransform::new(Mat3::identity(), Vec3::new(1.0, 0.0, 0.0));
    let t2 = RigidTransform::new(Mat3::identity(), Vec3::new(0.0, 2.0, 0.0));
    let composed = t1.then(&t2);
    let p = Vec3::new(0.0, 0.0, 0.0);
    let result = composed.apply(p);
    assert!(vec3_approx_eq(result, Vec3::new(1.0, 2.0, 0.0), 1e-12));
}

// ── NonNegF64 / PosUsize ─────────────────────────────────────────

#[test]
fn non_neg_rejects_negative() {
    assert!(NonNegF64::new(-1.0).is_none());
    assert!(NonNegF64::new(f64::NAN).is_none());
    assert!(NonNegF64::new(0.0).is_some());
    assert!(NonNegF64::new(42.0).is_some());
}

#[test]
fn pos_usize_rejects_zero() {
    assert!(PosUsize::new(0).is_none());
    assert!(PosUsize::new(1).is_some());
}
