use pcloud_cat_core::algebra::{Mat3, NonNegF64, PosUsize, RigidTransform, Vec3};
use pcloud_cat_core::cloud::{PointCloud, PointNormal};
use pcloud_cat_core::registration::{icp_align, icp_stream, IcpConfig, IcpVariant};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn vec3_approx_eq(a: Vec3, b: Vec3, tol: f64) -> bool {
    approx_eq(a.x(), b.x(), tol)
        && approx_eq(a.y(), b.y(), tol)
        && approx_eq(a.z(), b.z(), tol)
}

fn make_config() -> IcpConfig {
    IcpConfig::new(
        IcpVariant::PointToPoint,
        PosUsize::new(50).unwrap_or_else(|| PosUsize::new(1).expect("1 > 0")),
        NonNegF64::new(1e-10).unwrap_or_else(|| NonNegF64::new(0.0).expect("0 >= 0")),
    )
}

fn simple_cloud() -> PointCloud {
    let pts = vec![
        PointNormal::new(Vec3::new(0.0, 0.0, 0.0)),
        PointNormal::new(Vec3::new(1.0, 0.0, 0.0)),
        PointNormal::new(Vec3::new(0.0, 1.0, 0.0)),
        PointNormal::new(Vec3::new(0.0, 0.0, 1.0)),
        PointNormal::new(Vec3::new(1.0, 1.0, 0.0)),
        PointNormal::new(Vec3::new(1.0, 0.0, 1.0)),
        PointNormal::new(Vec3::new(0.0, 1.0, 1.0)),
        PointNormal::new(Vec3::new(1.0, 1.0, 1.0)),
    ];
    PointCloud::from_points(pts).expect("non-empty")
}

#[test]
fn icp_identical_clouds_converges_immediately() {
    let cloud = simple_cloud();
    let config = make_config();
    let result = icp_align(cloud.clone(), cloud, config).run();
    match result {
        Ok(tf) => {
            let p = Vec3::new(1.0, 2.0, 3.0);
            assert!(
                vec3_approx_eq(tf.apply(p), p, 1e-6),
                "Expected identity transform, got {tf:?}",
            );
        }
        Err(e) => panic!("ICP failed: {e}"),
    }
}

#[test]
fn icp_recovers_pure_translation() {
    let cloud = simple_cloud();
    let offset = Vec3::new(0.5, -0.3, 0.2);
    let transform = RigidTransform::new(Mat3::identity(), offset);
    let shifted = cloud.transformed(&transform);

    let config = make_config();
    let result = icp_align(shifted, cloud, config).run();
    match result {
        Ok(tf) => {
            // The recovered transform should approximately undo the shift.
            let test_point = Vec3::new(1.0, 1.0, 1.0);
            let original = test_point;
            let shifted_pt = transform.apply(test_point);
            let recovered = tf.apply(shifted_pt);
            assert!(
                vec3_approx_eq(recovered, original, 0.1),
                "Expected recovery near {original:?}, got {recovered:?}"
            );
        }
        Err(e) => panic!("ICP failed: {e}"),
    }
}

#[test]
fn icp_stream_emits_steps() {
    let cloud = simple_cloud();
    let config = IcpConfig::new(
        IcpVariant::PointToPoint,
        PosUsize::new(5).unwrap_or_else(|| PosUsize::new(1).expect("1 > 0")),
        NonNegF64::new(1e-10).unwrap_or_else(|| NonNegF64::new(0.0).expect("0 >= 0")),
    );
    let steps = icp_stream(cloud.clone(), cloud, config)
        .take(3)
        .collect()
        .run();
    match steps {
        Ok(s) => assert!(!s.is_empty(), "Expected at least one step"),
        Err(e) => panic!("ICP stream failed: {e}"),
    }
}

#[test]
fn icp_rmse_is_non_negative() {
    let cloud = simple_cloud();
    let offset = Vec3::new(0.1, 0.0, 0.0);
    let shifted = cloud.transformed(&RigidTransform::new(Mat3::identity(), offset));

    let config = make_config();
    let steps = icp_stream(shifted, cloud, config).collect().run();
    match steps {
        Ok(s) => {
            assert!(
                s.iter().all(|step| step.rmse() >= 0.0),
                "All RMSE values should be non-negative",
            );
        }
        Err(e) => panic!("ICP stream failed: {e}"),
    }
}
