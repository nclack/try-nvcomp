#[allow(non_camel_case_types)]
#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bindings::*;

pub mod benchmark;
pub mod compressors;
pub mod data;

pub const CHUNK_SIZE_U16: usize = 1024 * 1024 / 2; // 1MB = 512K u16s
pub const NUM_CHUNKS: usize = 1000;
pub const NVCOMP_SUCCESS: nvcompStatus_t = nvcompStatus_t_nvcompSuccess;

// Re-export commonly used items for convenience
pub use benchmark::{run_compression_benchmark, run_decompression_benchmark};
pub use compressors::{Compressor, Lz4Compressor, ZstdCompressor};
pub use data::{
    create_test_data, create_uncompressed_test_data, generate_sample_data, BufferHeader,
    CompressedData,
};
