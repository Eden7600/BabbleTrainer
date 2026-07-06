// Fast data path: rayon-parallel decode up front + background-prefetch batchers so the GPU stays
// fed (batch size unchanged — larger batches hurt expression quality, an optimization-dynamics
// effect). Split into: frame store + sample build (frames), mixup/paired/unlabeled indices (index),
// worker threads (prefetch), eval + init helpers (evalhelp), and the fit_* drivers (train).
mod evalhelp;
mod frames;
mod index;
mod prefetch;
mod train;

pub use frames::{build_samples, Frames};
pub use train::{fit_expr_staged, fit_gaze_fast, FastCfg};
