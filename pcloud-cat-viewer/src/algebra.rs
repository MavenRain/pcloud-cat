//! GPU-oriented linear algebra types (f32).
//!
//! [`Vec3f`], [`Vec4f`], and [`Mat4`] use named fields internally
//! (no array indexing).  [`Mat4::as_array`] produces the column-major
//! `[f32; 16]` needed at the WebGL boundary.

// ── Vec3f ────────────────────────────────────────────────────────

/// A 3-component f32 vector with named fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3f {
    vx: f32,
    vy: f32,
    vz: f32,
}

impl Vec3f {
    /// Construct from three components.
    #[must_use]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { vx: x, vy: y, vz: z }
    }

    /// The zero vector.
    #[must_use]
    pub fn zero() -> Self {
        Self { vx: 0.0, vy: 0.0, vz: 0.0 }
    }

    /// The positive-Y unit vector.
    #[must_use]
    pub fn unit_y() -> Self {
        Self { vx: 0.0, vy: 1.0, vz: 0.0 }
    }

    /// X component.
    #[must_use]
    pub fn x(self) -> f32 {
        self.vx
    }

    /// Y component.
    #[must_use]
    pub fn y(self) -> f32 {
        self.vy
    }

    /// Z component.
    #[must_use]
    pub fn z(self) -> f32 {
        self.vz
    }

    /// Dot product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.vx * rhs.vx + self.vy * rhs.vy + self.vz * rhs.vz
    }

    /// Cross product.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            vx: self.vy * rhs.vz - self.vz * rhs.vy,
            vy: self.vz * rhs.vx - self.vx * rhs.vz,
            vz: self.vx * rhs.vy - self.vy * rhs.vx,
        }
    }

    /// Squared length.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Unit vector, falling back to +Z if near zero.
    #[must_use]
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            Self { vx: 0.0, vy: 0.0, vz: 1.0 }
        } else {
            self.scale(1.0 / len)
        }
    }

    /// Scalar multiplication.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self {
            vx: self.vx * s,
            vy: self.vy * s,
            vz: self.vz * s,
        }
    }
}

impl std::ops::Add for Vec3f {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            vx: self.vx + rhs.vx,
            vy: self.vy + rhs.vy,
            vz: self.vz + rhs.vz,
        }
    }
}

impl std::ops::Sub for Vec3f {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            vx: self.vx - rhs.vx,
            vy: self.vy - rhs.vy,
            vz: self.vz - rhs.vz,
        }
    }
}

impl std::ops::Neg for Vec3f {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            vx: -self.vx,
            vy: -self.vy,
            vz: -self.vz,
        }
    }
}

// ── Vec4f ────────────────────────────────────────────────────────

/// A 4-component f32 vector (one column of a [`Mat4`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4f {
    vx: f32,
    vy: f32,
    vz: f32,
    vw: f32,
}

impl Vec4f {
    /// Construct from four components.
    #[must_use]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { vx: x, vy: y, vz: z, vw: w }
    }

    /// X component.
    #[must_use]
    pub fn x(self) -> f32 {
        self.vx
    }

    /// Y component.
    #[must_use]
    pub fn y(self) -> f32 {
        self.vy
    }

    /// Z component.
    #[must_use]
    pub fn z(self) -> f32 {
        self.vz
    }

    /// W component.
    #[must_use]
    pub fn w(self) -> f32 {
        self.vw
    }

    /// Dot product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.vx * rhs.vx + self.vy * rhs.vy + self.vz * rhs.vz + self.vw * rhs.vw
    }
}

// ── Mat4 ─────────────────────────────────────────────────────────

/// A 4x4 column-major matrix stored as four [`Vec4f`] columns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    c0: Vec4f,
    c1: Vec4f,
    c2: Vec4f,
    c3: Vec4f,
}

impl Mat4 {
    /// Construct from four column vectors.
    #[must_use]
    pub fn from_columns(c0: Vec4f, c1: Vec4f, c2: Vec4f, c3: Vec4f) -> Self {
        Self { c0, c1, c2, c3 }
    }

    /// Column 0.
    #[must_use]
    pub fn col0(self) -> Vec4f {
        self.c0
    }

    /// Column 1.
    #[must_use]
    pub fn col1(self) -> Vec4f {
        self.c1
    }

    /// Column 2.
    #[must_use]
    pub fn col2(self) -> Vec4f {
        self.c2
    }

    /// Column 3.
    #[must_use]
    pub fn col3(self) -> Vec4f {
        self.c3
    }

    /// Row `r` as a [`Vec4f`] (extracts one component from each column).
    #[must_use]
    fn row(self, pick: fn(Vec4f) -> f32) -> Vec4f {
        Vec4f::new(
            pick(self.c0),
            pick(self.c1),
            pick(self.c2),
            pick(self.c3),
        )
    }

    /// Matrix-matrix product.
    #[must_use]
    pub fn mul_mat(self, rhs: Self) -> Self {
        let row_x = self.row(Vec4f::x);
        let row_y = self.row(Vec4f::y);
        let row_z = self.row(Vec4f::z);
        let row_w = self.row(Vec4f::w);

        let mul_col = |col: Vec4f| -> Vec4f {
            Vec4f::new(
                row_x.dot(col),
                row_y.dot(col),
                row_z.dot(col),
                row_w.dot(col),
            )
        };

        Self {
            c0: mul_col(rhs.c0),
            c1: mul_col(rhs.c1),
            c2: mul_col(rhs.c2),
            c3: mul_col(rhs.c3),
        }
    }

    /// Flatten to a column-major `[f32; 16]` for the WebGL boundary.
    #[must_use]
    pub fn as_array(self) -> [f32; 16] {
        [
            self.c0.vx, self.c0.vy, self.c0.vz, self.c0.vw,
            self.c1.vx, self.c1.vy, self.c1.vz, self.c1.vw,
            self.c2.vx, self.c2.vy, self.c2.vz, self.c2.vw,
            self.c3.vx, self.c3.vy, self.c3.vz, self.c3.vw,
        ]
    }

    /// Perspective projection matrix.
    #[must_use]
    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y / 2.0).tan();
        let nf = 1.0 / (near - far);
        Self {
            c0: Vec4f::new(f / aspect, 0.0, 0.0, 0.0),
            c1: Vec4f::new(0.0, f, 0.0, 0.0),
            c2: Vec4f::new(0.0, 0.0, (far + near) * nf, -1.0),
            c3: Vec4f::new(0.0, 0.0, 2.0 * far * near * nf, 0.0),
        }
    }

    /// Look-at view matrix.
    #[must_use]
    pub fn look_at(eye: Vec3f, target: Vec3f, up: Vec3f) -> Self {
        let fwd = (target - eye).normalized();
        let right = fwd.cross(up).normalized();
        let cam_up = right.cross(fwd);

        Self {
            c0: Vec4f::new(right.x(), cam_up.x(), -fwd.x(), 0.0),
            c1: Vec4f::new(right.y(), cam_up.y(), -fwd.y(), 0.0),
            c2: Vec4f::new(right.z(), cam_up.z(), -fwd.z(), 0.0),
            c3: Vec4f::new(
                -right.dot(eye),
                -cam_up.dot(eye),
                fwd.dot(eye),
                1.0,
            ),
        }
    }
}

impl std::ops::Mul for Mat4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.mul_mat(rhs)
    }
}
