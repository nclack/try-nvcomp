use bytesize::ByteSize;
use cudarc::driver::safe::{CudaSlice, CudaContext};
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::DevicePtr;
use std::error::Error;
use tracing::info;

use crate::{bindings::*, CHUNK_SIZE_U16, NUM_CHUNKS, NVCOMP_SUCCESS};

pub trait Compressor {
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, Box<dyn Error>>;
    fn compress_on_gpu(
        &self,
        gpu_uncompressed_ptr_array: CUdeviceptr,
        gpu_uncompressed_sizes: CUdeviceptr,
        gpu_compressed_buffer_sizes: CUdeviceptr,
        gpu_actual_compressed_sizes: CUdeviceptr,
        gpu_compressed_ptr_array: CUdeviceptr,
        gpu_statuses: CUdeviceptr,
        stream: cudaStream_t,
    ) -> Result<(), Box<dyn Error>>;
    fn decompress_on_gpu(
        &self,
        gpu_compressed_ptr_array: CUdeviceptr,
        gpu_compressed_sizes: CUdeviceptr,
        gpu_decompressed_buffer_sizes: CUdeviceptr,
        gpu_actual_decompressed_sizes: CUdeviceptr,
        gpu_decompressed_ptr_array: CUdeviceptr,
        gpu_statuses: CUdeviceptr,
        stream: cudaStream_t,
    ) -> Result<(), Box<dyn Error>>;
    fn algorithm_name(&self) -> &'static str;
}

#[derive(Clone)]
pub struct ZstdCompressor {
    temp_buffer_ptr: Option<CUdeviceptr>,
    temp_buffer_size: usize,
}

impl ZstdCompressor {
    pub fn new() -> Self {
        Self {
            temp_buffer_ptr: None,
            temp_buffer_size: 0,
        }
    }

    pub fn init(&mut self) -> Result<(), Box<dyn Error>> {
        let mut temp_bytes: usize = 0;
        let format_opts = nvcompBatchedZstdOpts_t { reserved: 0 };
        unsafe {
            // Get temp size for compression (which is typically larger)
            let status = nvcompBatchedZstdCompressGetTempSize(
                NUM_CHUNKS,
                CHUNK_SIZE_U16 * 2,
                format_opts,
                &mut temp_bytes as *mut usize,
            );

            if status != NVCOMP_SUCCESS {
                return Err("Failed to get compression temp size".into());
            }

            // Also check decompression temp size and use the larger one
            let mut decomp_temp_bytes: usize = 0;
            let decomp_status = nvcompBatchedZstdDecompressGetTempSize(
                NUM_CHUNKS,
                CHUNK_SIZE_U16 * 2,
                &mut decomp_temp_bytes as *mut usize,
            );

            if decomp_status == NVCOMP_SUCCESS {
                temp_bytes = temp_bytes.max(decomp_temp_bytes);
            }
        }

        if temp_bytes > 0 {
            let device = CudaContext::new(0)?;
            let stream = device.default_stream();
            let temp_buffer: CudaSlice<u8> = stream.alloc_zeros(temp_bytes)?;
            self.temp_buffer_ptr = Some(temp_buffer.device_ptr(&stream).0);
            self.temp_buffer_size = temp_bytes;
            std::mem::forget(temp_buffer); // Prevent automatic deallocation
            info!(
                "nvCOMP Zstd temp memory allocated: {}",
                ByteSize(temp_bytes.try_into().unwrap())
            );
        }

        Ok(())
    }
}

impl Compressor for ZstdCompressor {
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
        Ok(zstd::encode_all(data, 3)?)
    }

    fn compress_on_gpu(
        &self,
        gpu_uncompressed_ptr_array: CUdeviceptr,
        gpu_uncompressed_sizes: CUdeviceptr,
        _gpu_compressed_buffer_sizes: CUdeviceptr,
        gpu_actual_compressed_sizes: CUdeviceptr,
        gpu_compressed_ptr_array: CUdeviceptr,
        _gpu_statuses: CUdeviceptr,
        stream: cudaStream_t,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            let format_opts = nvcompBatchedZstdOpts_t { reserved: 0 };
            // Perform batched compression
            let status = nvcompBatchedZstdCompressAsync(
                gpu_uncompressed_ptr_array as *const *const std::ffi::c_void,
                gpu_uncompressed_sizes as *const usize,
                CHUNK_SIZE_U16 * 2, // max_uncompressed_chunk_bytes
                NUM_CHUNKS,
                self.temp_buffer_ptr
                    .map_or(std::ptr::null_mut(), |ptr| ptr as *mut std::ffi::c_void),
                self.temp_buffer_size,
                gpu_compressed_ptr_array as *const *mut std::ffi::c_void,
                gpu_actual_compressed_sizes as *mut usize,
                format_opts,
                stream,
            );

            if status != NVCOMP_SUCCESS {
                return Err(
                    format!("nvCOMP Zstd compression failed with status: {}", status).into(),
                );
            }

            info!("nvCOMP Zstd compression completed successfully");
        }
        Ok(())
    }

    fn decompress_on_gpu(
        &self,
        gpu_compressed_ptr_array: CUdeviceptr,
        gpu_compressed_sizes: CUdeviceptr,
        gpu_decompressed_buffer_sizes: CUdeviceptr,
        gpu_actual_decompressed_sizes: CUdeviceptr,
        gpu_decompressed_ptr_array: CUdeviceptr,
        gpu_statuses: CUdeviceptr,
        stream: cudaStream_t,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            // Perform batched decompression
            let status = nvcompBatchedZstdDecompressAsync(
                gpu_compressed_ptr_array as *const *const std::ffi::c_void,
                gpu_compressed_sizes as *const usize,
                gpu_decompressed_buffer_sizes as *const usize,
                gpu_actual_decompressed_sizes as *mut usize,
                NUM_CHUNKS,
                self.temp_buffer_ptr
                    .map_or(std::ptr::null_mut(), |ptr| ptr as *mut std::ffi::c_void),
                self.temp_buffer_size,
                gpu_decompressed_ptr_array as *const *mut std::ffi::c_void,
                gpu_statuses as *mut nvcompStatus_t,
                stream,
            );

            if status != NVCOMP_SUCCESS {
                return Err(
                    format!("nvCOMP Zstd decompression failed with status: {}", status).into(),
                );
            }

            info!("nvCOMP Zstd decompression completed successfully");
        }
        Ok(())
    }

    fn algorithm_name(&self) -> &'static str {
        "Zstd"
    }
}

#[derive(Clone)]
pub struct Lz4Compressor {
    temp_buffer_ptr: Option<CUdeviceptr>,
    temp_buffer_size: usize,
}

impl Lz4Compressor {
    pub fn new() -> Self {
        Self {
            temp_buffer_ptr: None,
            temp_buffer_size: 0,
        }
    }

    pub fn init(&mut self) -> Result<(), Box<dyn Error>> {
        let mut temp_bytes: usize = 0;
        let format_opts = nvcompBatchedLZ4Opts_t {
            data_type: nvcompType_t_NVCOMP_TYPE_UCHAR,
        };
        unsafe {
            // Get temp size for compression (which is typically larger)
            let status = nvcompBatchedLZ4CompressGetTempSize(
                NUM_CHUNKS,
                CHUNK_SIZE_U16 * 2,
                format_opts,
                &mut temp_bytes as *mut usize,
            );

            if status != NVCOMP_SUCCESS {
                return Err("Failed to get compression temp size".into());
            }

            // Also check decompression temp size and use the larger one
            let mut decomp_temp_bytes: usize = 0;
            let decomp_status = nvcompBatchedLZ4DecompressGetTempSize(
                NUM_CHUNKS,
                CHUNK_SIZE_U16 * 2,
                &mut decomp_temp_bytes as *mut usize,
            );

            if decomp_status == NVCOMP_SUCCESS {
                temp_bytes = temp_bytes.max(decomp_temp_bytes);
            }
        }

        if temp_bytes > 0 {
            let device = CudaContext::new(0)?;
            let stream = device.default_stream();
            let temp_buffer: CudaSlice<u8> = stream.alloc_zeros(temp_bytes)?;
            self.temp_buffer_ptr = Some(temp_buffer.device_ptr(&stream).0);
            self.temp_buffer_size = temp_bytes;
            std::mem::forget(temp_buffer); // Prevent automatic deallocation
            info!(
                "nvCOMP LZ4 temp memory allocated: {}",
                ByteSize(temp_bytes.try_into().unwrap())
            );
        }

        Ok(())
    }
}

impl Compressor for Lz4Compressor {
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
        Ok(lz4::block::compress(
            data,
            Some(lz4::block::CompressionMode::DEFAULT),
            false,
        )?)
    }

    fn compress_on_gpu(
        &self,
        gpu_uncompressed_ptr_array: CUdeviceptr,
        gpu_uncompressed_sizes: CUdeviceptr,
        _gpu_compressed_buffer_sizes: CUdeviceptr,
        gpu_actual_compressed_sizes: CUdeviceptr,
        gpu_compressed_ptr_array: CUdeviceptr,
        _gpu_statuses: CUdeviceptr,
        stream: cudaStream_t,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            let format_opts = nvcompBatchedLZ4Opts_t {
                data_type: nvcompType_t_NVCOMP_TYPE_UCHAR,
            };
            // Perform batched compression
            let status = nvcompBatchedLZ4CompressAsync(
                gpu_uncompressed_ptr_array as *const *const std::ffi::c_void,
                gpu_uncompressed_sizes as *const usize,
                CHUNK_SIZE_U16 * 2, // max_uncompressed_chunk_bytes
                NUM_CHUNKS,
                self.temp_buffer_ptr
                    .map_or(std::ptr::null_mut(), |ptr| ptr as *mut std::ffi::c_void),
                self.temp_buffer_size,
                gpu_compressed_ptr_array as *const *mut std::ffi::c_void,
                gpu_actual_compressed_sizes as *mut usize,
                format_opts,
                stream,
            );

            if status != NVCOMP_SUCCESS {
                return Err(
                    format!("nvCOMP LZ4 compression failed with status: {}", status).into(),
                );
            }

            info!("nvCOMP LZ4 compression completed successfully");
        }
        Ok(())
    }

    fn decompress_on_gpu(
        &self,
        gpu_compressed_ptr_array: CUdeviceptr,
        gpu_compressed_sizes: CUdeviceptr,
        gpu_decompressed_buffer_sizes: CUdeviceptr,
        gpu_actual_decompressed_sizes: CUdeviceptr,
        gpu_decompressed_ptr_array: CUdeviceptr,
        gpu_statuses: CUdeviceptr,
        stream: cudaStream_t,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            // Perform batched decompression
            let status = nvcompBatchedLZ4DecompressAsync(
                gpu_compressed_ptr_array as *const *const std::ffi::c_void,
                gpu_compressed_sizes as *const usize,
                gpu_decompressed_buffer_sizes as *const usize,
                gpu_actual_decompressed_sizes as *mut usize,
                NUM_CHUNKS,
                self.temp_buffer_ptr
                    .map_or(std::ptr::null_mut(), |ptr| ptr as *mut std::ffi::c_void),
                self.temp_buffer_size,
                gpu_decompressed_ptr_array as *const *mut std::ffi::c_void,
                gpu_statuses as *mut nvcompStatus_t,
                stream,
            );

            if status != NVCOMP_SUCCESS {
                return Err(
                    format!("nvCOMP LZ4 decompression failed with status: {}", status).into(),
                );
            }

            info!("nvCOMP LZ4 decompression completed successfully");
        }
        Ok(())
    }

    fn algorithm_name(&self) -> &'static str {
        "LZ4"
    }
}