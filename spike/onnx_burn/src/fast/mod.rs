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

use burn::backend::wgpu::{MemoryConfiguration, RuntimeOptions};
use cubecl_runtime::memory_management::{MemoryPoolOptions, PoolType};

const MB: u64 = 1024 * 1024;

// Capped-pool memory config: the cubecl default (SubSlices) sizes its pages from
// max_storage_buffer_binding_size (GBs) and NEVER deallocates them, so each device server holds
// far more VRAM than its live tensors (measured: gaze holds ~2 GB for ~500 MB of need). Here:
// small sliced pools sized to the actual working set, finite dealloc periods so idle pages are
// returned, and an exclusive catch-all so big one-offs (e.g. the resident frame store) allocate
// exactly-sized pages instead of pinning a giant sliced page.
pub fn pool_opts(page_mb: u64) -> RuntimeOptions {
    let page = page_mb * MB;
    RuntimeOptions {
        memory_config: MemoryConfiguration::Custom {
            pool_options: vec![
                MemoryPoolOptions { pool_type: PoolType::ExclusivePages { max_alloc_size: 0 }, dealloc_period: None },
                MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages { page_size: page / 4, max_slice_size: page / 16 },
                    dealloc_period: Some(4000),
                },
                MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages { page_size: page, max_slice_size: page },
                    dealloc_period: Some(2000),
                },
                // Exclusive buckets for big allocations (activations, frame store): pages are
                // sized ~to the request (NOT the bucket; a u64::MAX catch-all would try to
                // allocate cur_avg_size = MAX/2 bytes on first use and poison the fusion stream),
                // and idle pages are returned. Top bucket = 4 GB, the wgpu binding limit region.
                MemoryPoolOptions { pool_type: PoolType::ExclusivePages { max_alloc_size: 128 * MB }, dealloc_period: Some(1000) },
                MemoryPoolOptions { pool_type: PoolType::ExclusivePages { max_alloc_size: 256 * MB }, dealloc_period: Some(1000) },
                MemoryPoolOptions { pool_type: PoolType::ExclusivePages { max_alloc_size: 512 * MB }, dealloc_period: Some(500) },
                MemoryPoolOptions { pool_type: PoolType::ExclusivePages { max_alloc_size: 1024 * MB }, dealloc_period: Some(500) },
                MemoryPoolOptions { pool_type: PoolType::ExclusivePages { max_alloc_size: 4096 * MB }, dealloc_period: Some(300) },
            ],
        },
        ..Default::default()
    }
}
