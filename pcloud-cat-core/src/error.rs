//! Error types for point cloud operations.
//!
//! A single [`Error`] enum covers all failure modes in the library:
//! algebra, I/O, registration convergence, and insufficient data.

/// Describes why a linear algebra operation failed.
#[derive(Debug, Clone)]
pub enum AlgebraErrorKind {
    /// A matrix was singular (determinant near zero).
    SingularMatrix,
    /// SVD did not converge within the iteration budget.
    SvdNotConverged,
}

impl std::fmt::Display for AlgebraErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SingularMatrix => write!(f, "singular matrix"),
            Self::SvdNotConverged => write!(f, "SVD did not converge"),
        }
    }
}

/// Describes why file I/O failed.
#[derive(Debug)]
pub enum IoErrorKind {
    /// The underlying OS I/O operation failed.
    Io(std::io::Error),
    /// The file could not be parsed.
    ParseError {
        /// The 1-based line number where parsing failed.
        line: usize,
        /// A human-readable description of the failure.
        detail: String,
    },
}

impl std::fmt::Display for IoErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::ParseError { line, detail } => {
                write!(f, "parse error at line {line}: {detail}")
            }
        }
    }
}

/// All errors that can arise in point cloud operations.
#[derive(Debug)]
pub enum Error {
    /// A linear algebra operation failed.
    Algebra(AlgebraErrorKind),
    /// The point cloud has too few points for the requested operation.
    InsufficientPoints {
        /// The minimum number of points required.
        required: usize,
        /// The number of points actually present.
        found: usize,
    },
    /// ICP did not converge within the step budget.
    ConvergenceFailure {
        /// The number of steps taken before giving up.
        steps: usize,
        /// The RMSE at the final step.
        residual: f64,
    },
    /// File I/O failed.
    Io(IoErrorKind),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Algebra(kind) => write!(f, "algebra error: {kind}"),
            Self::InsufficientPoints { required, found } => {
                write!(
                    f,
                    "insufficient points: required {required}, found {found}"
                )
            }
            Self::ConvergenceFailure { steps, residual } => {
                write!(
                    f,
                    "ICP did not converge after {steps} steps (residual: {residual})"
                )
            }
            Self::Io(kind) => write!(f, "{kind}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(IoErrorKind::Io(e)) => Some(e),
            _ => None,
        }
    }
}

impl From<AlgebraErrorKind> for Error {
    fn from(kind: AlgebraErrorKind) -> Self {
        Self::Algebra(kind)
    }
}

impl From<IoErrorKind> for Error {
    fn from(kind: IoErrorKind) -> Self {
        Self::Io(kind)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(IoErrorKind::Io(e))
    }
}
