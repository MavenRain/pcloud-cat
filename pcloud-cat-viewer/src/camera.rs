//! Orbit camera for 3D point cloud viewing.
//!
//! [`OrbitCamera`] is immutable; rotation, zoom, and pan return
//! a new camera state.

use crate::algebra::{Mat4, Vec3f};

/// An orbit camera that rotates around a target point.
///
/// All methods return a new `OrbitCamera` rather than mutating.
#[derive(Debug, Clone, Copy)]
pub struct OrbitCamera {
    target: Vec3f,
    distance: f32,
    azimuth: f32,
    elevation: f32,
    fov_y: f32,
    aspect: f32,
    near: f32,
    far: f32,
}

impl OrbitCamera {
    /// Construct a camera looking at `target` from `distance` away.
    #[must_use]
    pub fn new(target: Vec3f, distance: f32) -> Self {
        Self {
            target,
            distance,
            azimuth: 0.0,
            elevation: 0.3,
            fov_y: std::f32::consts::FRAC_PI_4,
            aspect: 1.0,
            near: 0.1,
            far: 1000.0,
        }
    }

    /// Set the viewport aspect ratio (width / height).
    #[must_use]
    pub fn with_aspect(self, aspect: f32) -> Self {
        Self { aspect, ..self }
    }

    /// Rotate the camera by delta angles (radians).
    #[must_use]
    pub fn rotate(self, delta_azimuth: f32, delta_elevation: f32) -> Self {
        let max_elev = std::f32::consts::FRAC_PI_2 - 0.01;
        let new_elevation = (self.elevation + delta_elevation).clamp(-max_elev, max_elev);
        Self {
            azimuth: self.azimuth + delta_azimuth,
            elevation: new_elevation,
            ..self
        }
    }

    /// Zoom by changing the distance (positive delta zooms out).
    #[must_use]
    pub fn zoom(self, delta: f32) -> Self {
        Self {
            distance: (self.distance + delta).max(0.1),
            ..self
        }
    }

    /// Pan the target point in the camera's local XY plane.
    #[must_use]
    pub fn pan(self, dx: f32, dy: f32) -> Self {
        let right = Vec3f::new(self.azimuth.cos(), 0.0, self.azimuth.sin());
        let up = Vec3f::unit_y();
        Self {
            target: self.target + right.scale(dx) + up.scale(dy),
            ..self
        }
    }

    /// Compute the combined view-projection matrix.
    #[must_use]
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// The view matrix.
    #[must_use]
    fn view_matrix(&self) -> Mat4 {
        let cos_az = self.azimuth.cos();
        let sin_az = self.azimuth.sin();
        let cos_el = self.elevation.cos();
        let sin_el = self.elevation.sin();

        let eye = self.target
            + Vec3f::new(
                self.distance * cos_el * sin_az,
                self.distance * sin_el,
                self.distance * cos_el * cos_az,
            );

        Mat4::look_at(eye, self.target, Vec3f::unit_y())
    }

    /// The perspective projection matrix.
    #[must_use]
    fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective(self.fov_y, self.aspect, self.near, self.far)
    }
}
