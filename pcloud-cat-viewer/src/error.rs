//! Viewer-specific error types.

/// All errors that can arise in the WASM viewer.
#[derive(Debug, Clone)]
pub enum ViewerError {
    /// The WebGL2 rendering context was lost or could not be acquired.
    ContextLost,
    /// A shader failed to compile.
    ShaderCompilationFailed(String),
    /// A GPU buffer could not be created.
    BufferCreationFailed,
    /// The canvas element was not found in the DOM.
    CanvasNotFound(String),
    /// A uniform location could not be found.
    UniformNotFound(String),
}

impl std::fmt::Display for ViewerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ContextLost => write!(f, "WebGL2 context lost"),
            Self::ShaderCompilationFailed(msg) => {
                write!(f, "shader compilation failed: {msg}")
            }
            Self::BufferCreationFailed => write!(f, "GPU buffer creation failed"),
            Self::CanvasNotFound(id) => write!(f, "canvas not found: {id}"),
            Self::UniformNotFound(name) => {
                write!(f, "uniform not found: {name}")
            }
        }
    }
}

impl std::error::Error for ViewerError {}

impl From<ViewerError> for wasm_bindgen::JsValue {
    fn from(err: ViewerError) -> Self {
        wasm_bindgen::JsValue::from_str(&err.to_string())
    }
}
