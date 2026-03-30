use pcloud_cat_core::algebra::{NonNegF64, PosUsize, Vec3};
use pcloud_cat_core::anomaly::{
    statistical_outlier_removal, surface_deviation, PointClassification,
    StatisticalOutlierConfig, SurfaceDeviationConfig,
};
use pcloud_cat_core::cloud::{PointCloud, PointNormal};
use pcloud_cat_core::normal::{estimate_normals, NormalEstimationConfig};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn pos(n: usize) -> PosUsize {
    PosUsize::new(n).expect("positive")
}

fn nonneg(v: f64) -> NonNegF64 {
    NonNegF64::new(v).expect("non-negative")
}

// ── Normal estimation ────────────────────────────────────────────

fn planar_cloud() -> PointCloud {
    // Points on the z=0 plane (XY grid)
    let pts: Vec<PointNormal> = (-3..=3)
        .flat_map(|x| {
            (-3..=3).map(move |y| {
                #[allow(clippy::cast_precision_loss)]
                PointNormal::new(Vec3::new(f64::from(x), f64::from(y), 0.0))
            })
        })
        .collect();
    PointCloud::from_points(pts).expect("non-empty")
}

#[test]
fn normals_on_plane_are_vertical() {
    let cloud = planar_cloud();
    let config = NormalEstimationConfig::new(pos(6));
    let result = estimate_normals(&cloud, config).run();
    match result {
        Ok(cloud_with_normals) => {
            cloud_with_normals.points().iter().for_each(|p| {
                if let Some(n) = p.normal() {
                    // Normal should be approximately (0, 0, ±1)
                    assert!(
                        approx_eq(n.x().abs(), 0.0, 0.15)
                            && approx_eq(n.y().abs(), 0.0, 0.15)
                            && approx_eq(n.z().abs(), 1.0, 0.15),
                        "Expected vertical normal, got ({}, {}, {})",
                        n.x(),
                        n.y(),
                        n.z(),
                    );
                }
            });
        }
        Err(e) => panic!("Normal estimation failed: {e}"),
    }
}

#[test]
fn normal_estimation_insufficient_points_error() {
    // Cloud with 3 points, but k=10
    let pts = vec![
        PointNormal::new(Vec3::new(0.0, 0.0, 0.0)),
        PointNormal::new(Vec3::new(1.0, 0.0, 0.0)),
        PointNormal::new(Vec3::new(0.0, 1.0, 0.0)),
    ];
    let cloud = PointCloud::from_points(pts).expect("non-empty");
    let config = NormalEstimationConfig::new(pos(10));
    let result = estimate_normals(&cloud, config).run();
    assert!(result.is_err(), "Should fail with insufficient points");
}

// ── Statistical outlier removal ──────────────────────────────────

fn cloud_with_outlier() -> PointCloud {
    // Regular grid of 27 points + 1 far-away outlier
    let grid: Vec<PointNormal> = (0..3)
        .flat_map(|x| {
            (0..3).flat_map(move |y| {
                (0..3).map(move |z| {
                    #[allow(clippy::cast_precision_loss)]
                    PointNormal::new(Vec3::new(f64::from(x), f64::from(y), f64::from(z)))
                })
            })
        })
        .collect();

    let outlier = PointNormal::new(Vec3::new(100.0, 100.0, 100.0));
    let all: Vec<PointNormal> = grid.into_iter().chain(std::iter::once(outlier)).collect();
    PointCloud::from_points(all).expect("non-empty")
}

#[test]
fn sor_flags_extreme_outlier() {
    let cloud = cloud_with_outlier();
    let config = StatisticalOutlierConfig::new(pos(5), nonneg(2.0));
    let result = statistical_outlier_removal(&cloud, config).run();
    match result {
        Ok(classified) => {
            assert!(
                classified.outlier_count() >= 1,
                "Expected at least 1 outlier, found {}",
                classified.outlier_count(),
            );
            // The last point (100, 100, 100) should be an outlier
            let last = classified
                .classifications()
                .last()
                .expect("non-empty classifications");
            assert!(
                matches!(last, PointClassification::StatisticalOutlier { .. }),
                "Expected the extreme point to be classified as outlier, got {last:?}",
            );
        }
        Err(e) => panic!("SOR failed: {e}"),
    }
}

#[test]
fn sor_uniform_cloud_no_outliers() {
    // A perfectly uniform grid should have few or no outliers at a generous threshold
    let pts: Vec<PointNormal> = (0..5)
        .flat_map(|x| {
            (0..5).flat_map(move |y| {
                (0..5).map(move |z| {
                    #[allow(clippy::cast_precision_loss)]
                    PointNormal::new(Vec3::new(f64::from(x), f64::from(y), f64::from(z)))
                })
            })
        })
        .collect();
    let cloud = PointCloud::from_points(pts).expect("non-empty");
    let config = StatisticalOutlierConfig::new(pos(5), nonneg(3.0));
    let result = statistical_outlier_removal(&cloud, config).run();
    match result {
        Ok(classified) => {
            // Most points should be inliers in a uniform grid
            #[allow(clippy::cast_precision_loss)]
            let inlier_ratio =
                classified.inlier_count() as f64 / classified.cloud().len() as f64;
            assert!(
                inlier_ratio > 0.8,
                "Expected mostly inliers, got ratio {inlier_ratio}",
            );
        }
        Err(e) => panic!("SOR failed: {e}"),
    }
}

// ── Surface deviation ────────────────────────────────────────────

#[test]
fn identical_clouds_no_deviation() {
    let cloud = planar_cloud();
    let config = SurfaceDeviationConfig::new(nonneg(0.01));
    let result = surface_deviation(&cloud, &cloud, config).run();
    match result {
        Ok(classified) => {
            assert_eq!(
                classified.outlier_count(),
                0,
                "Identical clouds should have zero deviations",
            );
        }
        Err(e) => panic!("Surface deviation failed: {e}"),
    }
}

#[test]
fn shifted_cloud_detects_deviation() {
    let reference = planar_cloud();
    // Shift measured cloud up by 1.0 in Z
    let measured_pts: Vec<PointNormal> = reference
        .points()
        .iter()
        .map(|p| PointNormal::new(p.position() + Vec3::new(0.0, 0.0, 1.0)))
        .collect();
    let measured = PointCloud::from_points(measured_pts).expect("non-empty");

    // Threshold of 0.5 should flag all points (deviation = 1.0)
    let config = SurfaceDeviationConfig::new(nonneg(0.5));
    let result = surface_deviation(&measured, &reference, config).run();
    match result {
        Ok(classified) => {
            assert_eq!(
                classified.outlier_count(),
                classified.cloud().len(),
                "All points should be deviating",
            );
        }
        Err(e) => panic!("Surface deviation failed: {e}"),
    }
}
