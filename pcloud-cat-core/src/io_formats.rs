//! Point cloud file I/O.
//!
//! Load and save point clouds in PLY (ASCII) format using
//! [`Resource`]-managed file handles and [`Io`]-wrapped parsing.

use comp_cat_rs::effect::io::Io;

use crate::algebra::Vec3;
use crate::cloud::{PointCloud, PointNormal};
use crate::error::{Error, IoErrorKind};

/// Supported point cloud file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointCloudFormat {
    /// Stanford PLY format (ASCII variant).
    Ply,
}

/// Load a point cloud from a file.
///
/// Uses [`Resource`] for bracket-based file handle management,
/// ensuring the file is closed even if parsing fails.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be opened or parsed.
///
/// # Examples
///
/// ```rust,ignore
/// let cloud: Result<PointCloud, Error> =
///     load_cloud("scan.ply", PointCloudFormat::Ply).run();
/// ```
#[must_use]
pub fn load_cloud(path: &str, format: PointCloudFormat) -> Io<Error, PointCloud> {
    let path = path.to_owned();
    Io::suspend(move || {
        std::fs::read_to_string(&path).map_err(Error::from)
    })
    .flat_map(move |contents| {
        Io::suspend(move || match format {
            PointCloudFormat::Ply => parse_ply_ascii(&contents),
        })
    })
}

/// Save a point cloud to a file.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be written.
#[must_use]
pub fn save_cloud(cloud: &PointCloud, path: &str, format: PointCloudFormat) -> Io<Error, ()> {
    let cloud = cloud.clone();
    let path = path.to_owned();
    Io::suspend(move || match format {
        PointCloudFormat::Ply => write_ply_ascii(&cloud, &path),
    })
}

/// Parse an ASCII PLY string into a [`PointCloud`].
fn parse_ply_ascii(contents: &str) -> Result<PointCloud, Error> {
    let lines: Vec<&str> = contents.lines().collect();

    // Validate header
    lines.first().filter(|&&l| l.trim() == "ply").ok_or_else(|| {
        Error::Io(IoErrorKind::ParseError {
            line: 1,
            detail: "missing PLY magic".to_owned(),
        })
    })?;

    // Find vertex count
    let vertex_count = lines
        .iter()
        .enumerate()
        .find(|(_, l)| l.starts_with("element vertex"))
        .and_then(|(_, l)| l.split_whitespace().nth(2))
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or_else(|| {
            Error::Io(IoErrorKind::ParseError {
                line: 0,
                detail: "missing element vertex count".to_owned(),
            })
        })?;

    // Find which properties exist and their order
    let header_end = lines
        .iter()
        .position(|l| l.trim() == "end_header")
        .ok_or_else(|| {
            Error::Io(IoErrorKind::ParseError {
                line: 0,
                detail: "missing end_header".to_owned(),
            })
        })?;

    // Collect property names in order
    let properties: Vec<&str> = lines[..header_end]
        .iter()
        .filter(|l| l.starts_with("property"))
        .filter_map(|l| l.split_whitespace().nth(2))
        .collect();

    let x_idx = properties.iter().position(|&p| p == "x");
    let y_idx = properties.iter().position(|&p| p == "y");
    let z_idx = properties.iter().position(|&p| p == "z");
    let normal_indices = (
        properties.iter().position(|&p| p == "nx"),
        properties.iter().position(|&p| p == "ny"),
        properties.iter().position(|&p| p == "nz"),
    );
    let has_normals = normal_indices.0.is_some()
        && normal_indices.1.is_some()
        && normal_indices.2.is_some();

    // Parse vertex data
    let data_start = header_end + 1;
    let points: Result<Vec<PointNormal>, Error> = lines[data_start..]
        .iter()
        .take(vertex_count)
        .enumerate()
        .map(|(i, line)| {
            let vals: Vec<f64> = line
                .split_whitespace()
                .filter_map(|s| s.parse::<f64>().ok())
                .collect();

            let get_val = |idx: Option<usize>| -> Option<f64> {
                idx.and_then(|j| vals.get(j).copied())
            };

            let pos = x_idx
                .zip(y_idx)
                .zip(z_idx)
                .and_then(|((xi, yi), zi)| {
                    get_val(Some(xi))
                        .zip(get_val(Some(yi)))
                        .zip(get_val(Some(zi)))
                        .map(|((x, y), z)| Vec3::new(x, y, z))
                })
                .ok_or_else(|| {
                    Error::Io(IoErrorKind::ParseError {
                        line: data_start + i + 1,
                        detail: "missing x/y/z coordinates".to_owned(),
                    })
                })?;

            let normal = if has_normals {
                normal_indices.0
                    .zip(normal_indices.1)
                    .zip(normal_indices.2)
                    .and_then(|((nxi, nyi), nzi)| {
                        get_val(Some(nxi))
                            .zip(get_val(Some(nyi)))
                            .zip(get_val(Some(nzi)))
                            .map(|((vnx, vny), vnz)| Vec3::new(vnx, vny, vnz))
                    })
            } else {
                None
            };

            Ok(normal.map_or_else(
                || PointNormal::new(pos),
                |n| PointNormal::with_normal(pos, n),
            ))
        })
        .collect();

    points.and_then(|pts| {
        PointCloud::from_points(pts).ok_or(Error::InsufficientPoints {
            required: 1,
            found: 0,
        })
    })
}

/// Write a [`PointCloud`] to a PLY ASCII file.
fn write_ply_ascii(cloud: &PointCloud, path: &str) -> Result<(), Error> {
    let has_normals = cloud.points().iter().all(|p| p.normal().is_some());

    let header = {
        let prop_lines = if has_normals {
            "property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz"
        } else {
            "property float x\nproperty float y\nproperty float z"
        };
        format!(
            "ply\nformat ascii 1.0\nelement vertex {}\n{}\nend_header\n",
            cloud.len(),
            prop_lines,
        )
    };

    let body: String = cloud
        .points()
        .iter()
        .map(|p| {
            let pos = p.position();
            p.normal().map_or_else(
                || format!("{} {} {}", pos.x(), pos.y(), pos.z()),
                |n| {
                    format!(
                        "{} {} {} {} {} {}",
                        pos.x(), pos.y(), pos.z(),
                        n.x(), n.y(), n.z(),
                    )
                },
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let content = format!("{header}{body}\n");
    std::fs::write(path, content).map_err(Error::from)
}
