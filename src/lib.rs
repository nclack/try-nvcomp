use bytesize::ByteSize;
use cudarc::driver::safe::{CudaContext, CudaSlice};
use cudarc::driver::sys::{CUdeviceptr, CUevent_flags};
use cudarc::driver::DevicePtr;
use rand::Rng;
use std::error::Error;
use std::time::Instant;
use tracing::{debug, info, span, Level};

#[allow(non_camel_case_types)]
#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bindings::*;

pub const CHUNK_SIZE_U16: usize = 1024 * 1024 / 2; // 1MB = 512K u16s
pub const NUM_CHUNKS: usize = 1000;
pub const NVCOMP_SUCCESS: nvcompStatus_t = nvcompStatus_t_nvcompSuccess;

pub trait Compressor {
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, Box<dyn Error>>;
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

pub fn generate_sample_data() -> Vec<u16> {
    let mut rng = rand::thread_rng();
    (0..CHUNK_SIZE_U16)
        .map(|_| rng.gen_range(0..1024))
        .collect()
}

pub fn create_test_data<C: Compressor>(
    compressor: &C,
) -> Result<(Vec<Vec<u8>>, Vec<Vec<u8>>), Box<dyn Error>> {
    info!(
        "Generating {} chunks of {} u16 data",
        NUM_CHUNKS,
        ByteSize((CHUNK_SIZE_U16 * 2).try_into().unwrap())
    );

    // Generate 3 different patterns
    let patterns = vec![
        generate_sample_data(),
        (0..CHUNK_SIZE_U16).map(|i| (i % 65536) as u16).collect(),
        vec![42u16; CHUNK_SIZE_U16],
    ];

    let mut compressed_chunks = Vec::new();
    let mut original_chunks = Vec::new();

    for i in 0..NUM_CHUNKS {
        let pattern = &patterns[i % patterns.len()];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                pattern.as_ptr() as *const u8,
                pattern.len() * std::mem::size_of::<u16>(),
            )
        };

        // Store original data for verification
        original_chunks.push(bytes.to_vec());

        let compressed = compressor.compress_data(bytes)?;
        info!(
            "Chunk {}: compressed {} -> {} (ratio: {:.2})",
            i,
            ByteSize(bytes.len().try_into().unwrap()),
            ByteSize(compressed.len().try_into().unwrap()),
            bytes.len() as f32 / compressed.len() as f32
        );
        compressed_chunks.push(compressed);
    }

    Ok((compressed_chunks, original_chunks))
}

pub fn run_benchmark<C: Compressor>(compressor: C) -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    let _span = span!(
        Level::INFO,
        "nvcomp_benchmark",
        algorithm = compressor.algorithm_name()
    )
    .entered();

    info!(
        "Starting nvCOMP {} parallel decompression example",
        compressor.algorithm_name()
    );

    let (compressed_data, original_data) = create_test_data(&compressor)?;
    decompress_on_gpu(&compressor, &compressed_data, &original_data)?;

    info!(
        "{} example completed successfully",
        compressor.algorithm_name()
    );
    Ok(())
}

fn decompress_on_gpu<C: Compressor>(
    compressor: &C,
    compressed_chunks: &[Vec<u8>],
    original_chunks: &[Vec<u8>],
) -> Result<(), Box<dyn Error>> {
    let device = CudaContext::new(0)?;
    let stream = device.default_stream();

    // Create CUDA events for precise timing
    let start_event = device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let copy_start_event = device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let copy_end_event = device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let decomp_events = vec![
        device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
        device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
        device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
        device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
        device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
        device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
        device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
    ];
    let end_event = device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

    info!(
        "Starting GPU decompression of {} chunks",
        compressed_chunks.len()
    );
    let start = Instant::now();

    // Record start event
    start_event.record(&stream)?;
    copy_start_event.record(&stream)?;

    // Allocate individual GPU buffers for each compressed chunk
    let mut gpu_compressed_chunks: Vec<CudaSlice<u8>> = Vec::new();
    let mut gpu_compressed_ptrs: Vec<CUdeviceptr> = Vec::new();
    let mut compressed_sizes: Vec<usize> = Vec::new();

    for chunk in compressed_chunks {
        let gpu_chunk: CudaSlice<u8> = stream.memcpy_stod(chunk)?;
        let ptr = gpu_chunk.device_ptr(&stream).0;
        gpu_compressed_ptrs.push(ptr);
        compressed_sizes.push(chunk.len());
        gpu_compressed_chunks.push(gpu_chunk);
    }

    // Allocate GPU buffers for decompressed chunks
    let mut gpu_decompressed_chunks: Vec<CudaSlice<u8>> = Vec::new();
    let mut gpu_decompressed_ptrs: Vec<CUdeviceptr> = Vec::new();
    let chunk_size_bytes = CHUNK_SIZE_U16 * 2; // u16 = 2 bytes

    for _ in 0..NUM_CHUNKS {
        let gpu_chunk: CudaSlice<u8> = stream.alloc_zeros(chunk_size_bytes)?;
        let ptr = gpu_chunk.device_ptr(&stream).0;
        gpu_decompressed_ptrs.push(ptr);
        gpu_decompressed_chunks.push(gpu_chunk);
    }

    // Copy sizes to GPU
    let gpu_compressed_sizes: CudaSlice<usize> = stream.memcpy_stod(&compressed_sizes)?;
    let gpu_decompressed_buffer_sizes: CudaSlice<usize> =
        stream.memcpy_stod(&vec![chunk_size_bytes; NUM_CHUNKS])?;
    let gpu_actual_decompressed_sizes: CudaSlice<usize> = stream.alloc_zeros(NUM_CHUNKS)?;
    let gpu_statuses: CudaSlice<nvcompStatus_t> = stream.alloc_zeros(NUM_CHUNKS)?;

    // Copy pointer arrays to GPU
    let gpu_compressed_ptr_array: CudaSlice<CUdeviceptr> =
        stream.memcpy_stod(&gpu_compressed_ptrs)?;
    let gpu_decompressed_ptr_array: CudaSlice<CUdeviceptr> =
        stream.memcpy_stod(&gpu_decompressed_ptrs)?;

    // Record end of copy operations
    copy_end_event.record(&stream)?;

    // Synchronize to ensure all copies are complete before decompression
    stream.synchronize()?;
    let copy_time = start.elapsed();
    info!(
        "Data copy to GPU (CPU time): {:.4}ms",
        copy_time.as_secs_f64() * 1000.0
    );

    // Calculate total sizes for reporting
    let total_compressed_size: usize = compressed_sizes.iter().sum();
    let total_decompressed_size = NUM_CHUNKS * chunk_size_bytes;
    let overall_compression_ratio = total_decompressed_size as f32 / total_compressed_size as f32;

    info!(
        "nvCOMP {} batched decompression would be called here with:",
        compressor.algorithm_name()
    );
    info!("  {} chunks of compressed data", NUM_CHUNKS);
    info!(
        "  Total compressed data: {} (pointers at {:?})",
        ByteSize(total_compressed_size.try_into().unwrap()),
        gpu_compressed_ptr_array.device_ptr(&stream).0
    );
    info!(
        "  Total decompressed data: {} (pointers at {:?})",
        ByteSize(total_decompressed_size.try_into().unwrap()),
        gpu_decompressed_ptr_array.device_ptr(&stream).0
    );
    info!(
        "  Overall compression ratio: {:.2}x",
        overall_compression_ratio,
    );

    // Record decompression start event (after data copy)
    // Repeat a few of these on the stream
    decomp_events[0].record(&stream)?;
    for i in 1..decomp_events.len() {
        // Call algorithm-specific decompression
        compressor.decompress_on_gpu(
            gpu_compressed_ptr_array.device_ptr(&stream).0,
            gpu_compressed_sizes.device_ptr(&stream).0,
            gpu_decompressed_buffer_sizes.device_ptr(&stream).0,
            gpu_actual_decompressed_sizes.device_ptr(&stream).0,
            gpu_decompressed_ptr_array.device_ptr(&stream).0,
            gpu_statuses.device_ptr(&stream).0,
            stream.cu_stream() as cudaStream_t,
        )?;
        decomp_events[i].record(&stream)?;
    }

    end_event.record(&stream)?;
    stream.synchronize()?;

    // Calculate GPU timings using CUDA events
    info!("Calculating GPU timings using CUDA events...");

    // Get timing from CUDA events
    let total_gpu_time = start_event.elapsed_ms(&end_event)?;
    let copy_gpu_time = copy_start_event.elapsed_ms(&copy_end_event)?;
    let decomp_gpu_time = {
        (1..decomp_events.len())
            .filter_map(|i| decomp_events[i - 1].elapsed_ms(&decomp_events[i]).ok())
            .min_by(f32::total_cmp)
            .unwrap()
    };

    info!("Total GPU time: {:.4}ms", total_gpu_time);
    info!("Copy GPU time: {:.4}ms", copy_gpu_time);
    info!("Decomp GPU time: {:.4}ms", decomp_gpu_time);

    // For verification, copy some results back
    info!("Copying status results from GPU...");
    let host_statuses: Vec<nvcompStatus_t> = stream.memcpy_dtov(&gpu_statuses)?;
    info!("Copying size results from GPU...");
    let host_actual_sizes: Vec<usize> = stream.memcpy_dtov(&gpu_actual_decompressed_sizes)?;
    info!("Results copied successfully");

    // Calculate throughputs
    let copy_throughput_gb_s =
        (total_compressed_size as f64 / 1e9) / (copy_gpu_time as f64 / 1000.0);
    let decomp_throughput_gb_s =
        (total_decompressed_size as f64 / 1e9) / (decomp_gpu_time as f64 / 1000.0);

    info!(
        "GPU {} decompression completed:",
        compressor.algorithm_name()
    );
    info!(
        "  Copy to GPU: {:.4}ms ({:.2} GB/s)",
        copy_gpu_time, copy_throughput_gb_s
    );
    info!(
        "  Decompression: {:.4}ms ({:.2} GB/s)",
        decomp_gpu_time, decomp_throughput_gb_s
    );
    info!("  Total GPU time: {:.4}ms", total_gpu_time);
    info!("  Status check: {:?}", &host_statuses[..5]);
    info!("  Size check: {:?}", &host_actual_sizes[..5]);

    // Verify decompression results
    info!("Verifying decompressed data...");
    let mut verification_passed = 0;
    let mut verification_failed = 0;

    for (i, (gpu_chunk, expected)) in gpu_decompressed_chunks
        .iter()
        .zip(original_chunks.iter())
        .enumerate()
    {
        if host_statuses[i] == NVCOMP_SUCCESS {
            debug!("Verifying chunk {}: copying from GPU...", i);
            let decompressed_data: Vec<u8> = stream.memcpy_dtov(gpu_chunk)?;
            debug!("Chunk {} copied, comparing data...", i);

            if decompressed_data == *expected {
                verification_passed += 1;
            } else {
                verification_failed += 1;
                debug!(
                    "Chunk {} verification FAILED: lengths {} vs {}",
                    i,
                    decompressed_data.len(),
                    expected.len()
                );
                if decompressed_data.len() == expected.len() {
                    let mismatches = decompressed_data
                        .iter()
                        .zip(expected.iter())
                        .enumerate()
                        .filter(|(_, (a, b))| a != b)
                        .take(5)
                        .collect::<Vec<_>>();
                    debug!("  First few mismatches: {:?}", mismatches);
                }
            }
        } else {
            verification_failed += 1;
            debug!(
                "Chunk {} decompression failed with status {}",
                i, host_statuses[i]
            );
        }
    }

    info!(
        "Verification complete: {} passed, {} failed",
        verification_passed, verification_failed
    );

    if verification_failed > 0 {
        return Err(format!("{} chunks failed verification", verification_failed).into());
    }

    Ok(())
}

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
        unsafe {
            let status = nvcompBatchedZstdDecompressGetTempSize(
                NUM_CHUNKS,
                CHUNK_SIZE_U16 * 2,
                &mut temp_bytes as *mut usize,
            );

            if status != NVCOMP_SUCCESS {
                return Err("Failed to get temp size".into());
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
        unsafe {
            let status = nvcompBatchedLZ4DecompressGetTempSize(
                NUM_CHUNKS,
                CHUNK_SIZE_U16 * 2,
                &mut temp_bytes as *mut usize,
            );

            if status != NVCOMP_SUCCESS {
                return Err("Failed to get temp size".into());
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
