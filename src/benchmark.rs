use bytesize::ByteSize;
use cudarc::driver::safe::{CudaContext, CudaSlice};
use cudarc::driver::sys::{CUdeviceptr, CUevent_flags};
use cudarc::driver::DevicePtr;
use std::error::Error;
use std::time::Instant;
use tracing::{debug, info};

use crate::{
    bindings::*,
    compressors::Compressor,
    data::{create_test_data, create_uncompressed_test_data},
    CHUNK_SIZE_U16, NUM_CHUNKS, NVCOMP_SUCCESS,
};

/// Common CUDA events and timing utilities for benchmarks
struct BenchmarkEvents {
    start_event: cudarc::driver::safe::CudaEvent,
    copy_start_event: cudarc::driver::safe::CudaEvent,
    copy_end_event: cudarc::driver::safe::CudaEvent,
    operation_events: Vec<cudarc::driver::safe::CudaEvent>,
    d2h_start_event: cudarc::driver::safe::CudaEvent,
    d2h_end_event: cudarc::driver::safe::CudaEvent,
    end_event: cudarc::driver::safe::CudaEvent,
}

impl BenchmarkEvents {
    fn new(device: &std::sync::Arc<CudaContext>) -> Result<Self, Box<dyn Error>> {
        let create_event = || device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT));

        Ok(BenchmarkEvents {
            start_event: create_event()?,
            copy_start_event: create_event()?,
            copy_end_event: create_event()?,
            operation_events: (0..7)
                .map(|_| create_event())
                .collect::<Result<Vec<_>, _>>()?,
            d2h_start_event: create_event()?,
            d2h_end_event: create_event()?,
            end_event: create_event()?,
        })
    }

    fn calculate_timings(&self) -> Result<BenchmarkTimings, Box<dyn Error>> {
        let total_gpu_time = self.start_event.elapsed_ms(&self.end_event)?;
        let copy_gpu_time = self.copy_start_event.elapsed_ms(&self.copy_end_event)?;
        let operation_gpu_time = (1..self.operation_events.len())
            .filter_map(|i| {
                self.operation_events[i - 1]
                    .elapsed_ms(&self.operation_events[i])
                    .ok()
            })
            .min_by(f32::total_cmp)
            .unwrap();
        let d2h_gpu_time = self.d2h_start_event.elapsed_ms(&self.d2h_end_event)?;

        Ok(BenchmarkTimings {
            total_gpu_time,
            copy_gpu_time,
            operation_gpu_time,
            d2h_gpu_time,
        })
    }
}

/// Timing results from benchmark events
struct BenchmarkTimings {
    total_gpu_time: f32,
    copy_gpu_time: f32,
    operation_gpu_time: f32,
    d2h_gpu_time: f32,
}

/// Validation results for benchmarks
#[derive(Default)]
struct ValidationResults {
    passed: usize,
    failed: usize,
}

impl ValidationResults {
    fn log_and_check(&self, operation: &str) -> Result<(), Box<dyn Error>> {
        info!(
            "{} validation complete: {} passed, {} failed",
            operation, self.passed, self.failed
        );

        if self.failed > 0 {
            Err(format!("{} chunks failed {}", self.failed, operation).into())
        } else {
            Ok(())
        }
    }
}

/// Validates compression by decompressing GPU results and comparing with originals
fn validate_compression_chunks<C: Compressor>(
    compressor: &C,
    compressed_chunks: &[Vec<u8>],
    original_chunks: &[Vec<u8>],
    statuses: &[nvcompStatus_t],
) -> ValidationResults {
    let mut validation = ValidationResults::default();

    for (i, (compressed_chunk, expected)) in compressed_chunks
        .iter()
        .zip(original_chunks.iter())
        .enumerate()
    {
        let success = statuses[i] == NVCOMP_SUCCESS
            && !compressed_chunk.is_empty()
            && compressor.compress_data(expected).is_ok()
            && decompress_chunk_by_algorithm(
                compressor.algorithm_name(),
                compressed_chunk,
                expected.len(),
            )
            .map_or(false, |decompressed| {
                let matches = compare_byte_arrays(&decompressed, expected);
                if !matches {
                    debug!(
                        "Chunk {} validation FAILED: decompressed length {} vs expected {}",
                        i,
                        decompressed.len(),
                        expected.len()
                    );
                }
                matches
            });

        if success {
            validation.passed += 1;
        } else {
            validation.failed += 1;
            if statuses[i] != NVCOMP_SUCCESS {
                debug!("Chunk {} compression failed with status {}", i, statuses[i]);
            }
        }
    }

    validation
}

/// Validates decompression by comparing GPU results with original data
fn validate_decompression_chunks(
    decompressed_buffer: &[u8],
    original_chunks: &[Vec<u8>],
    statuses: &[nvcompStatus_t],
    actual_sizes: &[usize],
    chunk_size_bytes: usize,
) -> ValidationResults {
    let mut validation = ValidationResults::default();

    for (i, expected) in original_chunks.iter().enumerate() {
        let success = statuses[i] == NVCOMP_SUCCESS && {
            debug!("Verifying chunk {}: extracting from unified buffer...", i);
            let chunk_offset = i * chunk_size_bytes;
            let actual_size = actual_sizes[i];
            let chunk_data = &decompressed_buffer[chunk_offset..chunk_offset + actual_size];
            debug!("Chunk {} extracted, comparing data...", i);

            let matches = compare_byte_arrays(chunk_data, expected);
            if !matches {
                debug!(
                    "Chunk {} verification FAILED: lengths {} vs {}",
                    i,
                    chunk_data.len(),
                    expected.len()
                );
                if chunk_data.len() == expected.len() {
                    let mismatches = chunk_data
                        .iter()
                        .zip(expected.iter())
                        .enumerate()
                        .filter(|(_, (a, b))| a != b)
                        .take(5)
                        .collect::<Vec<_>>();
                    debug!("  First few mismatches: {:?}", mismatches);
                }
            }
            matches
        };

        if success {
            validation.passed += 1;
        } else {
            validation.failed += 1;
            if statuses[i] != NVCOMP_SUCCESS {
                debug!(
                    "Chunk {} decompression failed with status {}",
                    i, statuses[i]
                );
            }
        }
    }

    validation
}

/// Decompresses a chunk using the appropriate algorithm
fn decompress_chunk_by_algorithm(
    algorithm: &str,
    compressed_chunk: &[u8],
    expected_size: usize,
) -> Result<Vec<u8>, Box<dyn Error>> {
    match algorithm {
        "Zstd" => zstd::decode_all(compressed_chunk).map_err(|e| e.into()),
        "LZ4" => lz4::block::decompress(compressed_chunk, Some(expected_size as i32))
            .map_err(|e| e.into()),
        _ => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Unknown algorithm",
        )) as Box<dyn Error>),
    }
}

/// Compares two byte arrays for equality
fn compare_byte_arrays(a: &[u8], b: &[u8]) -> bool {
    a == b
}

/// Unified buffer memory layout calculator
struct UnifiedBufferLayout {
    pub max_chunk_size: usize,
    pub total_buffer_size: usize,
}

impl UnifiedBufferLayout {
    fn new_for_compression(total_data_size: usize, max_chunk_size: usize) -> Self {
        let chunk_offsets_size = NUM_CHUNKS * std::mem::size_of::<usize>();
        let chunk_sizes_size = NUM_CHUNKS * std::mem::size_of::<usize>();
        let header_size = std::mem::size_of::<crate::data::BufferHeader>();
        let align_to_8 = |size: usize| (size + 7) & !7;

        let aligned_data_size = align_to_8(total_data_size);
        let chunk_offsets_offset = aligned_data_size;
        let chunk_sizes_offset = align_to_8(chunk_offsets_offset + chunk_offsets_size);
        let header_offset = align_to_8(chunk_sizes_offset + chunk_sizes_size);
        let total_buffer_size = header_offset + header_size;

        Self {
            total_buffer_size,
            max_chunk_size,
        }
    }

    fn new_for_decompression(chunk_size_bytes: usize) -> Self {
        Self {
            total_buffer_size: NUM_CHUNKS * chunk_size_bytes,
            max_chunk_size: chunk_size_bytes,
        }
    }
}

/// GPU memory transfer manager for batch operations
struct GpuMemoryManager;

impl GpuMemoryManager {
    fn transfer_compression_metadata(
        stream: &std::sync::Arc<cudarc::driver::safe::CudaStream>,
        chunk_sizes: &[usize],
        max_compressed_size: usize,
        uncompressed_ptrs: &[CUdeviceptr],
        compressed_ptrs: &[CUdeviceptr],
        gpu_buffers: &mut CompressionGpuBuffers,
    ) -> Result<(), Box<dyn Error>> {
        stream.memcpy_htod(chunk_sizes, &mut gpu_buffers.uncompressed_sizes)?;
        stream.memcpy_htod(
            &vec![max_compressed_size; NUM_CHUNKS],
            &mut gpu_buffers.compressed_buffer_sizes,
        )?;
        stream.memcpy_htod(uncompressed_ptrs, &mut gpu_buffers.uncompressed_ptr_array)?;
        stream.memcpy_htod(compressed_ptrs, &mut gpu_buffers.compressed_ptr_array)?;
        Ok(())
    }

    fn transfer_decompression_metadata(
        stream: &std::sync::Arc<cudarc::driver::safe::CudaStream>,
        chunk_size_bytes: usize,
        compressed_ptrs: &[CUdeviceptr],
        decompressed_ptrs: &[CUdeviceptr],
        gpu_buffers: &mut DecompressionGpuBuffers,
    ) -> Result<(), Box<dyn Error>> {
        stream.memcpy_htod(
            &vec![chunk_size_bytes; NUM_CHUNKS],
            &mut gpu_buffers.decompressed_buffer_sizes,
        )?;
        stream.memcpy_htod(compressed_ptrs, &mut gpu_buffers.compressed_ptr_array)?;
        stream.memcpy_htod(decompressed_ptrs, &mut gpu_buffers.decompressed_ptr_array)?;
        Ok(())
    }
}

/// Pre-allocates common GPU buffers for compression operations
struct CompressionGpuBuffers {
    unified_compressed_buffer: CudaSlice<u8>,
    compressed_ptrs: Vec<CUdeviceptr>,
    uncompressed_sizes: CudaSlice<usize>,
    compressed_buffer_sizes: CudaSlice<usize>,
    actual_compressed_sizes: CudaSlice<usize>,
    statuses: CudaSlice<nvcompStatus_t>,
    uncompressed_ptr_array: CudaSlice<CUdeviceptr>,
    compressed_ptr_array: CudaSlice<CUdeviceptr>,
}

impl CompressionGpuBuffers {
    fn new(
        stream: std::sync::Arc<cudarc::driver::safe::CudaStream>,
        max_compressed_size: usize,
        total_unified_buffer_size: usize,
    ) -> Result<Self, Box<dyn Error>> {
        // Pre-allocate unified GPU buffer
        let unified_compressed_buffer: CudaSlice<u8> =
            stream.alloc_zeros(total_unified_buffer_size)?;
        let gpu_base_ptr = unified_compressed_buffer.device_ptr(&stream).0;

        // Calculate pointers into the unified buffer for each chunk
        let mut compressed_ptrs: Vec<CUdeviceptr> = Vec::new();
        for i in 0..NUM_CHUNKS {
            let chunk_offset = i * max_compressed_size;
            compressed_ptrs.push(gpu_base_ptr + chunk_offset as u64);
        }

        // Pre-allocate metadata arrays on GPU
        let uncompressed_sizes: CudaSlice<usize> = stream.alloc_zeros(NUM_CHUNKS)?;
        let compressed_buffer_sizes: CudaSlice<usize> = stream.alloc_zeros(NUM_CHUNKS)?;
        let actual_compressed_sizes: CudaSlice<usize> = stream.alloc_zeros(NUM_CHUNKS)?;
        let statuses: CudaSlice<nvcompStatus_t> = stream.alloc_zeros(NUM_CHUNKS)?;

        // Pre-allocate pointer arrays on GPU
        let uncompressed_ptr_array: CudaSlice<CUdeviceptr> = stream.alloc_zeros(NUM_CHUNKS)?;
        let compressed_ptr_array: CudaSlice<CUdeviceptr> = stream.alloc_zeros(NUM_CHUNKS)?;

        Ok(CompressionGpuBuffers {
            unified_compressed_buffer,
            compressed_ptrs,
            uncompressed_sizes,
            compressed_buffer_sizes,
            actual_compressed_sizes,
            statuses,
            uncompressed_ptr_array,
            compressed_ptr_array,
        })
    }
}

/// Pre-allocates common GPU buffers for decompression operations
struct DecompressionGpuBuffers {
    unified_decompressed_buffer: CudaSlice<u8>,
    decompressed_ptrs: Vec<CUdeviceptr>,
    decompressed_buffer_sizes: CudaSlice<usize>,
    actual_decompressed_sizes: CudaSlice<usize>,
    statuses: CudaSlice<nvcompStatus_t>,
    compressed_ptr_array: CudaSlice<CUdeviceptr>,
    decompressed_ptr_array: CudaSlice<CUdeviceptr>,
}

impl DecompressionGpuBuffers {
    fn new(
        stream: std::sync::Arc<cudarc::driver::safe::CudaStream>,
        chunk_size_bytes: usize,
        total_decompressed_size: usize,
    ) -> Result<Self, Box<dyn Error>> {
        // Pre-allocate unified decompressed buffer
        let unified_decompressed_buffer: CudaSlice<u8> =
            stream.alloc_zeros(total_decompressed_size)?;
        let gpu_decompressed_base_ptr = unified_decompressed_buffer.device_ptr(&stream).0;

        // Calculate pointers into the unified decompressed buffer for each chunk
        let mut decompressed_ptrs: Vec<CUdeviceptr> = Vec::new();
        for i in 0..NUM_CHUNKS {
            let chunk_offset = i * chunk_size_bytes;
            decompressed_ptrs.push(gpu_decompressed_base_ptr + chunk_offset as u64);
        }

        // Pre-allocate metadata arrays on GPU
        let decompressed_buffer_sizes: CudaSlice<usize> = stream.alloc_zeros(NUM_CHUNKS)?;
        let actual_decompressed_sizes: CudaSlice<usize> = stream.alloc_zeros(NUM_CHUNKS)?;
        let statuses: CudaSlice<nvcompStatus_t> = stream.alloc_zeros(NUM_CHUNKS)?;

        // Pre-allocate pointer arrays on GPU
        let compressed_ptr_array: CudaSlice<CUdeviceptr> = stream.alloc_zeros(NUM_CHUNKS)?;
        let decompressed_ptr_array: CudaSlice<CUdeviceptr> = stream.alloc_zeros(NUM_CHUNKS)?;

        Ok(DecompressionGpuBuffers {
            unified_decompressed_buffer,
            decompressed_ptrs,
            decompressed_buffer_sizes,
            actual_decompressed_sizes,
            statuses,
            compressed_ptr_array,
            decompressed_ptr_array,
        })
    }
}

#[tracing::instrument(skip(compressor), fields(algorithm = compressor.algorithm_name()))]
pub fn run_decompression_benchmark<C: Compressor>(compressor: C) -> Result<(), Box<dyn Error>> {
    info!(
        "Starting nvCOMP {} parallel decompression example",
        compressor.algorithm_name()
    );

    // Create fresh test data for each run (matching compression pattern exactly)
    let (compressed_data, original_data) = create_test_data(&compressor)?;
    decompress_on_gpu(&compressor, &compressed_data, &original_data)?;

    info!(
        "{} example completed successfully",
        compressor.algorithm_name()
    );
    Ok(())
}

#[tracing::instrument(skip(compressor), fields(algorithm = compressor.algorithm_name()))]
pub fn run_compression_benchmark<C: Compressor>(compressor: C) -> Result<(), Box<dyn Error>> {
    info!(
        "Starting nvCOMP {} parallel compression benchmark",
        compressor.algorithm_name()
    );

    let original_data = create_uncompressed_test_data();
    compress_on_gpu(&compressor, &original_data)?;

    info!(
        "{} compression benchmark completed successfully",
        compressor.algorithm_name()
    );
    Ok(())
}

fn compress_on_gpu<C: Compressor>(
    compressor: &C,
    original_chunks: &[Vec<u8>],
) -> Result<(), Box<dyn Error>> {
    let device = CudaContext::new(0)?;
    let stream = device.default_stream();

    // Create CUDA events for precise timing
    let events = BenchmarkEvents::new(&device)?;

    info!(
        "Starting GPU compression of {} chunks",
        original_chunks.len()
    );

    // Pre-calculate sizes and layout BEFORE timing starts
    let chunk_sizes: Vec<usize> = original_chunks.iter().map(|chunk| chunk.len()).collect();
    let max_compressed_size = chunk_sizes[0] * 2; // Conservative upper bound per chunk
    let total_compressed_buffer_size = max_compressed_size * NUM_CHUNKS;
    let layout =
        UnifiedBufferLayout::new_for_compression(total_compressed_buffer_size, max_compressed_size);

    // Pre-allocate ALL GPU buffers BEFORE timing starts
    let mut gpu_buffers = CompressionGpuBuffers::new(
        stream.clone(),
        max_compressed_size,
        layout.total_buffer_size,
    )?;

    // Create PINNED unified uncompressed buffer BEFORE timing starts for optimal H→D transfer
    let total_uncompressed_size: usize = chunk_sizes.iter().sum();
    let mut pinned_unified_buffer: cudarc::driver::safe::PinnedHostSlice<u8> =
        unsafe { device.alloc_pinned(total_uncompressed_size)? };

    // Pack all uncompressed chunks into single contiguous pinned buffer
    {
        let buffer_slice = pinned_unified_buffer.as_mut_slice()?;
        let mut offset = 0;
        for chunk in original_chunks {
            buffer_slice[offset..offset + chunk.len()].copy_from_slice(chunk);
            offset += chunk.len();
        }
    }

    let _start = Instant::now();

    // Record start event - timing begins HERE
    events.start_event.record(&stream)?;
    events.copy_start_event.record(&stream)?;

    // Transfer the ENTIRE pinned unified uncompressed buffer to GPU in ONE memcpy operation!
    let gpu_unified_uncompressed_buffer: CudaSlice<u8> =
        stream.memcpy_stod(&pinned_unified_buffer)?;
    let gpu_uncompressed_base_ptr = gpu_unified_uncompressed_buffer.device_ptr(&stream).0;

    // Calculate pointers into the unified uncompressed buffer for each chunk
    let mut gpu_uncompressed_ptrs: Vec<CUdeviceptr> = Vec::new();
    let mut offset = 0;
    for &chunk_size in &chunk_sizes {
        gpu_uncompressed_ptrs.push(gpu_uncompressed_base_ptr + offset as u64);
        offset += chunk_size;
    }

    // Copy pre-calculated metadata to GPU (only data transfer, no allocation)
    let compressed_ptrs = gpu_buffers.compressed_ptrs.clone();
    GpuMemoryManager::transfer_compression_metadata(
        &stream,
        &chunk_sizes,
        max_compressed_size,
        &gpu_uncompressed_ptrs,
        &compressed_ptrs,
        &mut gpu_buffers,
    )?;

    // Record end of copy operations
    events.copy_end_event.record(&stream)?;

    info!(
        "nvCOMP {} batched compression will be called with:",
        compressor.algorithm_name()
    );
    info!("  {} chunks of uncompressed data", NUM_CHUNKS);
    info!(
        "  Total uncompressed data: {}",
        ByteSize(total_uncompressed_size.try_into().unwrap())
    );
    info!(
        "  Unified pinned input buffer size: {}",
        ByteSize(total_uncompressed_size.try_into().unwrap())
    );
    info!(
        "  Unified output buffer size: {} (includes metadata)",
        ByteSize(layout.total_buffer_size.try_into().unwrap())
    );

    // Record compression start events and perform compression
    events.operation_events[0].record(&stream)?;
    for i in 1..events.operation_events.len() {
        // Call algorithm-specific compression
        compressor.compress_on_gpu(
            gpu_buffers.uncompressed_ptr_array.device_ptr(&stream).0,
            gpu_buffers.uncompressed_sizes.device_ptr(&stream).0,
            gpu_buffers.compressed_buffer_sizes.device_ptr(&stream).0,
            gpu_buffers.actual_compressed_sizes.device_ptr(&stream).0,
            gpu_buffers.compressed_ptr_array.device_ptr(&stream).0,
            gpu_buffers.statuses.device_ptr(&stream).0,
            stream.cu_stream() as cudaStream_t,
        )?;
        events.operation_events[i].record(&stream)?;
    }

    events.end_event.record(&stream)?;

    // Copy results back to host with timing
    info!("Copying compression results from GPU...");
    let host_statuses: Vec<nvcompStatus_t> = stream.memcpy_dtov(&gpu_buffers.statuses)?;
    let host_actual_sizes: Vec<usize> = stream.memcpy_dtov(&gpu_buffers.actual_compressed_sizes)?;

    // Pre-allocate pinned host memory OUTSIDE the timed section for optimal performance
    let mut pinned_host_buffer: cudarc::driver::safe::PinnedHostSlice<u8> =
        unsafe { device.alloc_pinned(layout.total_buffer_size)? };

    // Time the D→H transfer of compressed data - single unified buffer copy
    events.d2h_start_event.record(&stream)?;

    // Copy entire unified buffer in ONE memcpy operation to pre-allocated pinned memory
    stream.memcpy_dtoh(
        &gpu_buffers.unified_compressed_buffer,
        &mut pinned_host_buffer,
    )?;

    events.d2h_end_event.record(&stream)?;

    // Convert pinned memory to regular slice for processing
    let unified_host_buffer = pinned_host_buffer.as_slice()?;

    // Parse individual chunks from the unified buffer using unsafe code
    let mut compressed_host_data: Vec<Vec<u8>> = Vec::new();
    for i in 0..NUM_CHUNKS {
        if host_statuses[i] == NVCOMP_SUCCESS {
            let actual_size = host_actual_sizes[i];
            let chunk_offset = i * max_compressed_size;

            // Extract chunk data from unified buffer
            let chunk_data = &unified_host_buffer[chunk_offset..chunk_offset + actual_size];
            compressed_host_data.push(chunk_data.to_vec());
        } else {
            compressed_host_data.push(Vec::new());
        }
    }

    // Synchronize once after all GPU operations are queued
    stream.synchronize()?;
    info!("Results copied successfully");

    // Calculate GPU timings using CUDA events
    info!("Calculating GPU timings using CUDA events...");
    let timings = events.calculate_timings()?;

    info!("Total GPU time: {:.4}ms", timings.total_gpu_time);
    info!("Copy GPU time: {:.4}ms", timings.copy_gpu_time);
    info!("Compression GPU time: {:.4}ms", timings.operation_gpu_time);

    // Calculate total compressed size and compression ratio
    let total_compressed_size: usize = host_actual_sizes
        .iter()
        .zip(host_statuses.iter())
        .filter_map(|(&size, &status)| {
            if status == NVCOMP_SUCCESS {
                Some(size)
            } else {
                None
            }
        })
        .sum();
    let compression_ratio = total_uncompressed_size as f32 / total_compressed_size as f32;

    // Calculate D→H transfer time and throughput
    let transferred_size = unified_host_buffer.len();
    let d2h_throughput_gb_s =
        (transferred_size as f64 / 1e9) / (timings.d2h_gpu_time as f64 / 1000.0);

    // Calculate throughputs
    let copy_throughput_gb_s =
        (total_uncompressed_size as f64 / 1e9) / (timings.copy_gpu_time as f64 / 1000.0);
    let comp_throughput_gb_s =
        (total_uncompressed_size as f64 / 1e9) / (timings.operation_gpu_time as f64 / 1000.0);

    info!("GPU {} compression completed:", compressor.algorithm_name());
    info!(
        "  Compression ratio: {:.2}x ({} -> {})",
        compression_ratio,
        ByteSize(total_uncompressed_size.try_into().unwrap()),
        ByteSize(total_compressed_size.try_into().unwrap())
    );
    info!(
        "  Copy to GPU (H→D): {:.4}ms ({:.2} GB/s) [pinned unified buffer]",
        timings.copy_gpu_time, copy_throughput_gb_s
    );
    info!(
        "  Compression: {:.4}ms ({:.2} GB/s)",
        timings.operation_gpu_time, comp_throughput_gb_s
    );
    info!(
        "  Copy from GPU (D→H): {:.4}ms ({:.2} GB/s) [unified buffer: {}]",
        timings.d2h_gpu_time,
        d2h_throughput_gb_s,
        ByteSize(transferred_size.try_into().unwrap())
    );
    info!("  Total GPU time: {:.4}ms", timings.total_gpu_time);
    info!("  Status check: {:?}", &host_statuses[..5]);
    info!("  Size check: {:?}", &host_actual_sizes[..5]);

    // Validate by decompressing on CPU
    info!("Validating compressed data by decompressing on CPU...");
    let validation = validate_compression_chunks(
        compressor,
        &compressed_host_data,
        original_chunks,
        &host_statuses,
    );

    validation.log_and_check("validation")?;

    Ok(())
}

fn decompress_on_gpu<C: Compressor>(
    compressor: &C,
    compressed_data: &crate::data::CompressedData,
    original_chunks: &[Vec<u8>],
) -> Result<(), Box<dyn Error>> {
    let device = CudaContext::new(0)?;
    let stream = device.default_stream();

    // Create CUDA events for precise timing
    let events = BenchmarkEvents::new(&device)?;

    info!(
        "Starting GPU decompression of {} chunks (single unified buffer)",
        compressed_data.header.chunk_count
    );
    info!(
        "Unified buffer size: {} bytes",
        compressed_data.buffer.len()
    );
    info!(
        "Compressed data size: {} bytes",
        compressed_data.header.compressed_data_size
    );
    info!(
        "Chunk offsets at offset: {}",
        compressed_data.header.chunk_offsets_offset
    );
    info!(
        "Chunk sizes at offset: {}",
        compressed_data.header.chunk_sizes_offset
    );

    // Pre-extract metadata from the host buffer BEFORE timing starts
    let buffer_slice = compressed_data.buffer.as_slice()?;
    let offsets_bytes = &buffer_slice[compressed_data.header.chunk_offsets_offset
        ..compressed_data.header.chunk_offsets_offset
            + (compressed_data.header.chunk_count * std::mem::size_of::<usize>())];
    let sizes_bytes = &buffer_slice[compressed_data.header.chunk_sizes_offset
        ..compressed_data.header.chunk_sizes_offset
            + (compressed_data.header.chunk_count * std::mem::size_of::<usize>())];

    // Convert byte slices back to usize arrays
    let temp_offsets_buffer: &[usize] = unsafe {
        std::slice::from_raw_parts(
            offsets_bytes.as_ptr() as *const usize,
            compressed_data.header.chunk_count,
        )
    };
    let temp_sizes_buffer: &[usize] = unsafe {
        std::slice::from_raw_parts(
            sizes_bytes.as_ptr() as *const usize,
            compressed_data.header.chunk_count,
        )
    };

    let compressed_sizes = temp_sizes_buffer;
    let chunk_size_bytes = CHUNK_SIZE_U16 * 2; // u16 = 2 bytes
    let layout = UnifiedBufferLayout::new_for_decompression(chunk_size_bytes);

    // Pre-allocate ALL GPU buffers BEFORE timing starts
    let mut gpu_buffers =
        DecompressionGpuBuffers::new(stream.clone(), chunk_size_bytes, layout.total_buffer_size)?;

    // Pre-allocate pinned host memory OUTSIDE the timed section for optimal performance
    let mut pinned_decompressed_buffer: cudarc::driver::safe::PinnedHostSlice<u8> =
        unsafe { device.alloc_pinned(layout.total_buffer_size)? };

    let _start = Instant::now();

    // Record start event - timing begins HERE
    events.start_event.record(&stream)?;
    events.copy_start_event.record(&stream)?;

    // Transfer the ENTIRE unified buffer to GPU in ONE memcpy operation!
    // (Now using fresh CompressedData with fresh pinned buffer for each run)
    let gpu_unified_buffer: CudaSlice<u8> = stream.memcpy_stod(&compressed_data.buffer)?;
    let gpu_base_ptr = gpu_unified_buffer.device_ptr(&stream).0;

    // Parse the unified buffer - extract metadata from GPU memory
    let compressed_data_gpu_ptr = gpu_base_ptr;
    let _chunk_offsets_gpu_ptr = gpu_base_ptr + compressed_data.header.chunk_offsets_offset as u64;
    let chunk_sizes_gpu_ptr = gpu_base_ptr + compressed_data.header.chunk_sizes_offset as u64;

    // Create pointer array pointing into the compressed data section
    let mut gpu_compressed_ptrs: Vec<CUdeviceptr> = Vec::new();
    for &offset in temp_offsets_buffer {
        gpu_compressed_ptrs.push(compressed_data_gpu_ptr + offset as u64);
    }

    // Copy pre-calculated metadata to GPU (only data transfer, no allocation)
    let decompressed_ptrs = gpu_buffers.decompressed_ptrs.clone();
    GpuMemoryManager::transfer_decompression_metadata(
        &stream,
        chunk_size_bytes,
        &gpu_compressed_ptrs,
        &decompressed_ptrs,
        &mut gpu_buffers,
    )?;

    // Sizes are already on GPU as part of the unified buffer!
    let gpu_compressed_sizes_ptr = chunk_sizes_gpu_ptr;

    // Record end of copy operations
    events.copy_end_event.record(&stream)?;

    // Calculate total sizes for reporting
    let total_compressed_size: usize = compressed_sizes.iter().sum();
    let overall_compression_ratio = layout.total_buffer_size as f32 / total_compressed_size as f32;

    info!(
        "nvCOMP {} batched decompression would be called here with:",
        compressor.algorithm_name()
    );
    info!("  {} chunks of compressed data", NUM_CHUNKS);
    info!(
        "  Total compressed data: {} (pointers at {:?})",
        ByteSize(total_compressed_size.try_into().unwrap()),
        gpu_buffers.compressed_ptr_array.device_ptr(&stream).0
    );
    info!(
        "  Total decompressed data: {} (pointers at {:?})",
        ByteSize(layout.total_buffer_size.try_into().unwrap()),
        gpu_buffers.decompressed_ptr_array.device_ptr(&stream).0
    );
    info!(
        "  Overall compression ratio: {:.2}x",
        overall_compression_ratio,
    );

    // Record decompression start event (after data copy)
    // Repeat a few of these on the stream
    events.operation_events[0].record(&stream)?;
    for i in 1..events.operation_events.len() {
        // Call algorithm-specific decompression
        compressor.decompress_on_gpu(
            gpu_buffers.compressed_ptr_array.device_ptr(&stream).0,
            gpu_compressed_sizes_ptr as CUdeviceptr,
            gpu_buffers.decompressed_buffer_sizes.device_ptr(&stream).0,
            gpu_buffers.actual_decompressed_sizes.device_ptr(&stream).0,
            gpu_buffers.decompressed_ptr_array.device_ptr(&stream).0,
            gpu_buffers.statuses.device_ptr(&stream).0,
            stream.cu_stream() as cudaStream_t,
        )?;
        events.operation_events[i].record(&stream)?;
    }

    events.end_event.record(&stream)?;

    // Copy results back to host with timing
    info!("Copying decompression results from GPU...");
    let host_statuses: Vec<nvcompStatus_t> = stream.memcpy_dtov(&gpu_buffers.statuses)?;
    let host_actual_sizes: Vec<usize> =
        stream.memcpy_dtov(&gpu_buffers.actual_decompressed_sizes)?;

    // Time the D→H transfer of decompressed data - single unified buffer copy
    events.d2h_start_event.record(&stream)?;

    // Copy entire unified decompressed buffer in ONE memcpy operation to pre-allocated pinned memory
    stream.memcpy_dtoh(
        &gpu_buffers.unified_decompressed_buffer,
        &mut pinned_decompressed_buffer,
    )?;

    events.d2h_end_event.record(&stream)?;

    // Synchronize once after all GPU operations are queued
    stream.synchronize()?;
    info!("Results copied successfully");

    // Calculate GPU timings using CUDA events
    info!("Calculating GPU timings using CUDA events...");
    let timings = events.calculate_timings()?;

    info!("Total GPU time: {:.4}ms", timings.total_gpu_time);
    info!("Copy GPU time: {:.4}ms", timings.copy_gpu_time);
    info!("Decomp GPU time: {:.4}ms", timings.operation_gpu_time);

    // Calculate D→H transfer time and throughput
    let transferred_size = layout.total_buffer_size;
    let d2h_throughput_gb_s =
        (transferred_size as f64 / 1e9) / (timings.d2h_gpu_time as f64 / 1000.0);

    // Calculate throughputs
    let copy_throughput_gb_s =
        (total_compressed_size as f64 / 1e9) / (timings.copy_gpu_time as f64 / 1000.0);
    let decomp_throughput_gb_s =
        (layout.total_buffer_size as f64 / 1e9) / (timings.operation_gpu_time as f64 / 1000.0);

    info!(
        "GPU {} decompression completed:",
        compressor.algorithm_name()
    );
    info!(
        "  Copy to GPU (pinned): {:.4}ms ({:.2} GB/s)",
        timings.copy_gpu_time, copy_throughput_gb_s
    );
    info!(
        "  Decompression: {:.4}ms ({:.2} GB/s)",
        timings.operation_gpu_time, decomp_throughput_gb_s
    );
    info!(
        "  Copy from GPU (D→H): {:.4}ms ({:.2} GB/s) [unified buffer: {}]",
        timings.d2h_gpu_time,
        d2h_throughput_gb_s,
        ByteSize(transferred_size.try_into().unwrap())
    );
    info!("  Total GPU time: {:.4}ms", timings.total_gpu_time);
    info!("  Status check: {:?}", &host_statuses[..5]);
    info!("  Size check: {:?}", &host_actual_sizes[..5]);

    // Verify decompression results using the unified buffer
    info!("Verifying decompressed data...");

    // Convert pinned memory to regular slice for processing
    let unified_decompressed_buffer = pinned_decompressed_buffer.as_slice()?;

    let verification = validate_decompression_chunks(
        unified_decompressed_buffer,
        &original_chunks,
        &host_statuses,
        &host_actual_sizes,
        layout.max_chunk_size,
    );

    verification.log_and_check("verification")?;

    Ok(())
}
