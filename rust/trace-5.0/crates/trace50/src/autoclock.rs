//! STEVE / Recursive AutoClock spectral temporal deconvolution for TRACE-5.0.
//!
//! The implementation lives in the shared workspace crate `aeon-autoclock`;
//! this module re-exports it so existing `crate::autoclock::*` paths keep
//! working.
pub use aeon_autoclock::*;
