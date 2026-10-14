/// Exact Cyclic Jacobi Eigensolver for real symmetric K x K matrices.
///
/// The implementation lives in the shared workspace crate `aeon-manifold`;
/// this module re-exports it so `crate::jacobi::symmetric_jacobi_eigh` and
/// `rhizaeon::jacobi::symmetric_jacobi_eigh` keep resolving.
pub use aeon_manifold::{symmetric_jacobi_eigh, Eigendecomposition};
