use bytesize::ByteSize;
use cudarc::driver::safe::{CudaContext, CudaSlice};
use cudarc::driver::sys::{CUdeviceptr, CUevent_flags};
use cudarc::driver::DevicePtr;
use std::error::Error;
use std::time::Instant;
use tracing::{debug, info, span, Level};

use crate::{
    bindings::*,
    compressors::Compressor,
    data::{create_test_data, create_uncompressed_test_data, CompressedData},
    CHUNK_SIZE_U16, NUM_CHUNKS, NVCOMP_SUCCESS,
};

pub fn run_decompression_benchmark<C: Compressor>(compressor: C) -> Result<(), Box<dyn Error>> {
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

pub fn run_compression_benchmark<C: Compressor>(compressor: C) -> Result<(), Box<dyn Error>> {
    let _span = span!(
        Level::INFO,
        "nvcomp_compression_benchmark",
        algorithm = compressor.algorithm_name()
    )
    .entered();

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
    let start_event = device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let copy_start_event = device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let copy_end_event = device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let comp_events = vec![
        device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
        device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
        device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
        device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
        device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
        device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
        device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
    ];
    let d2h_start_event = device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let d2h_end_event = device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let end_event = device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

    info!(
        "Starting GPU compression of {} chunks",
        original_chunks.len()
    );

    let _start = Instant::now();

    // Record start event
    start_event.record(&stream)?;
    copy_start_event.record(&stream)?;

    // Transfer uncompressed data to GPU
    let mut gpu_uncompressed_chunks: Vec<CudaSlice<u8>> = Vec::new();
    let mut gpu_uncompressed_ptrs: Vec<CUdeviceptr> = Vec::new();
    let chunk_sizes: Vec<usize> = original_chunks.iter().map(|chunk| chunk.len()).collect();

    for chunk in original_chunks {
        let gpu_chunk: CudaSlice<u8> = stream.memcpy_stod(chunk)?;
        let ptr = gpu_chunk.device_ptr(&stream).0;
        gpu_uncompressed_ptrs.push(ptr);
        gpu_uncompressed_chunks.push(gpu_chunk);
    }

    // Allocate GPU buffers for compressed output
    let mut gpu_compressed_chunks: Vec<CudaSlice<u8>> = Vec::new();
    let mut gpu_compressed_ptrs: Vec<CUdeviceptr> = Vec::new();
    let max_compressed_size = chunk_sizes[0] * 2; // Conservative upper bound

    for _ in 0..NUM_CHUNKS {
        let gpu_chunk: CudaSlice<u8> = stream.alloc_zeros(max_compressed_size)?;
        let ptr = gpu_chunk.device_ptr(&stream).0;
        gpu_compressed_ptrs.push(ptr);
        gpu_compressed_chunks.push(gpu_chunk);
    }

    // Create metadata arrays on GPU
    let gpu_uncompressed_sizes: CudaSlice<usize> = stream.memcpy_stod(&chunk_sizes)?;
    let gpu_compressed_buffer_sizes: CudaSlice<usize> =
        stream.memcpy_stod(&vec![max_compressed_size; NUM_CHUNKS])?;
    let gpu_actual_compressed_sizes: CudaSlice<usize> = stream.alloc_zeros(NUM_CHUNKS)?;
    let gpu_statuses: CudaSlice<nvcompStatus_t> = stream.alloc_zeros(NUM_CHUNKS)?;

    // Copy pointer arrays to GPU
    let gpu_uncompressed_ptr_array: CudaSlice<CUdeviceptr> =
        stream.memcpy_stod(&gpu_uncompressed_ptrs)?;
    let gpu_compressed_ptr_array: CudaSlice<CUdeviceptr> =
        stream.memcpy_stod(&gpu_compressed_ptrs)?;

    // Record end of copy operations
    copy_end_event.record(&stream)?;

    // Calculate total sizes for reporting
    let total_uncompressed_size: usize = chunk_sizes.iter().sum();

    info!(
        "nvCOMP {} batched compression will be called with:",
        compressor.algorithm_name()
    );
    info!("  {} chunks of uncompressed data", NUM_CHUNKS);
    info!(
        "  Total uncompressed data: {}",
        ByteSize(total_uncompressed_size.try_into().unwrap())
    );

    // Record compression start events and perform compression
    comp_events[0].record(&stream)?;
    for i in 1..comp_events.len() {
        // Call algorithm-specific compression
        compressor.compress_on_gpu(
            gpu_uncompressed_ptr_array.device_ptr(&stream).0,
            gpu_uncompressed_sizes.device_ptr(&stream).0,
            gpu_compressed_buffer_sizes.device_ptr(&stream).0,
            gpu_actual_compressed_sizes.device_ptr(&stream).0,
            gpu_compressed_ptr_array.device_ptr(&stream).0,
            gpu_statuses.device_ptr(&stream).0,
            stream.cu_stream() as cudaStream_t,
        )?;
        comp_events[i].record(&stream)?;
    }

    end_event.record(&stream)?;

    // Copy results back to host with timing
    info!("Copying compression results from GPU...");
    let host_statuses: Vec<nvcompStatus_t> = stream.memcpy_dtov(&gpu_statuses)?;
    let host_actual_sizes: Vec<usize> = stream.memcpy_dtov(&gpu_actual_compressed_sizes)?;

    // Time the D→H transfer of compressed data
    d2h_start_event.record(&stream)?;
    let mut compressed_host_data: Vec<Vec<u8>> = Vec::new();
    for (i, gpu_chunk) in gpu_compressed_chunks.iter().enumerate() {
        if host_statuses[i] == NVCOMP_SUCCESS {
            let actual_size = host_actual_sizes[i];
            let mut host_chunk: Vec<u8> = stream.memcpy_dtov(gpu_chunk)?;
            host_chunk.truncate(actual_size);
            compressed_host_data.push(host_chunk);
        } else {
            compressed_host_data.push(Vec::new());
        }
    }
    d2h_end_event.record(&stream)?;
    
    // Synchronize once after all GPU operations are queued
    stream.synchronize()?;
    info!("Results copied successfully");

    // Calculate GPU timings using CUDA events
    info!("Calculating GPU timings using CUDA events...");

    let total_gpu_time = start_event.elapsed_ms(&end_event)?;
    let copy_gpu_time = copy_start_event.elapsed_ms(&copy_end_event)?;
    let comp_gpu_time = {
        (1..comp_events.len())
            .filter_map(|i| comp_events[i - 1].elapsed_ms(&comp_events[i]).ok())
            .min_by(f32::total_cmp)
            .unwrap()
    };

    info!("Total GPU time: {:.4}ms", total_gpu_time);
    info!("Copy GPU time: {:.4}ms", copy_gpu_time);
    info!("Compression GPU time: {:.4}ms", comp_gpu_time);

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
    let d2h_gpu_time = d2h_start_event.elapsed_ms(&d2h_end_event)?;
    let d2h_throughput_gb_s =
        (total_compressed_size as f64 / 1e9) / (d2h_gpu_time as f64 / 1000.0);

    // Calculate throughputs
    let copy_throughput_gb_s =
        (total_uncompressed_size as f64 / 1e9) / (copy_gpu_time as f64 / 1000.0);
    let comp_throughput_gb_s =
        (total_uncompressed_size as f64 / 1e9) / (comp_gpu_time as f64 / 1000.0);

    info!("GPU {} compression completed:", compressor.algorithm_name());
    info!(
        "  Compression ratio: {:.2}x ({} -> {})",
        compression_ratio,
        ByteSize(total_uncompressed_size.try_into().unwrap()),
        ByteSize(total_compressed_size.try_into().unwrap())
    );
    info!(
        "  Copy to GPU (H→D): {:.4}ms ({:.2} GB/s)",
        copy_gpu_time, copy_throughput_gb_s
    );
    info!(
        "  Compression: {:.4}ms ({:.2} GB/s)",
        comp_gpu_time, comp_throughput_gb_s
    );
    info!(
        "  Copy from GPU (D→H): {:.4}ms ({:.2} GB/s)",
        d2h_gpu_time, d2h_throughput_gb_s
    );
    info!("  Total GPU time: {:.4}ms", total_gpu_time);
    info!("  Status check: {:?}", &host_statuses[..5]);
    info!("  Size check: {:?}", &host_actual_sizes[..5]);

    // Validate by decompressing on CPU
    info!("Validating compressed data by decompressing on CPU...");
    let mut validation_passed = 0;
    let mut validation_failed = 0;

    for (i, (compressed_chunk, expected)) in compressed_host_data
        .iter()
        .zip(original_chunks.iter())
        .enumerate()
    {
        if host_statuses[i] == NVCOMP_SUCCESS && !compressed_chunk.is_empty() {
            match compressor.compress_data(expected) {
                Ok(_cpu_compressed) => {
                    // For validation, we decompress the GPU result and compare with original
                    match (|| match compressor.algorithm_name() {
                        "Zstd" => {
                            zstd::decode_all(compressed_chunk.as_slice()).map_err(|e| e.into())
                        }
                        "LZ4" => {
                            lz4::block::decompress(compressed_chunk, Some(expected.len() as i32))
                                .map_err(|e| e.into())
                        }
                        _ => Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "Unknown algorithm",
                        )) as Box<dyn Error>),
                    })() {
                        Ok(decompressed) => {
                            if decompressed == *expected {
                                validation_passed += 1;
                            } else {
                                validation_failed += 1;
                                debug!(
                                    "Chunk {} validation FAILED: decompressed length {} vs expected {}",
                                    i, decompressed.len(), expected.len()
                                );
                            }
                        }
                        Err(_) => {
                            validation_failed += 1;
                            debug!("Chunk {} decompression failed during validation", i);
                        }
                    }
                }
                Err(_) => {
                    validation_failed += 1;
                    debug!("Chunk {} CPU compression failed", i);
                }
            }
        } else {
            validation_failed += 1;
            debug!(
                "Chunk {} compression failed with status {}",
                i, host_statuses[i]
            );
        }
    }

    info!(
        "Validation complete: {} passed, {} failed",
        validation_passed, validation_failed
    );

    if validation_failed > 0 {
        return Err(format!("{} chunks failed validation", validation_failed).into());
    }

    Ok(())
}

fn decompress_on_gpu<C: Compressor>(
    compressor: &C,
    compressed_data: &CompressedData,
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
    let _start = Instant::now();

    // Record start event
    start_event.record(&stream)?;
    copy_start_event.record(&stream)?;

    // Transfer the ENTIRE unified buffer to GPU in ONE memcpy operation!
    let gpu_unified_buffer: CudaSlice<u8> = stream.memcpy_stod(&compressed_data.buffer)?;
    let gpu_base_ptr = gpu_unified_buffer.device_ptr(&stream).0;

    // Parse the unified buffer - extract metadata from GPU memory
    let compressed_data_gpu_ptr = gpu_base_ptr;
    let _chunk_offsets_gpu_ptr = gpu_base_ptr + compressed_data.header.chunk_offsets_offset as u64;
    let chunk_sizes_gpu_ptr = gpu_base_ptr + compressed_data.header.chunk_sizes_offset as u64;

    // We need the metadata to create pointer arrays - extract from the host buffer
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

    // Create pointer array pointing into the compressed data section
    let mut gpu_compressed_ptrs: Vec<CUdeviceptr> = Vec::new();
    for &offset in temp_offsets_buffer {
        gpu_compressed_ptrs.push(compressed_data_gpu_ptr + offset as u64);
    }

    let compressed_sizes = temp_sizes_buffer;

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

    // Sizes are already on GPU as part of the unified buffer!
    let gpu_compressed_sizes_ptr = chunk_sizes_gpu_ptr;
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
            gpu_compressed_sizes_ptr as CUdeviceptr,
            gpu_decompressed_buffer_sizes.device_ptr(&stream).0,
            gpu_actual_decompressed_sizes.device_ptr(&stream).0,
            gpu_decompressed_ptr_array.device_ptr(&stream).0,
            gpu_statuses.device_ptr(&stream).0,
            stream.cu_stream() as cudaStream_t,
        )?;
        decomp_events[i].record(&stream)?;
    }

    end_event.record(&stream)?;

    // For verification, copy some results back
    info!("Copying status results from GPU...");
    let host_statuses: Vec<nvcompStatus_t> = stream.memcpy_dtov(&gpu_statuses)?;
    info!("Copying size results from GPU...");
    let host_actual_sizes: Vec<usize> = stream.memcpy_dtov(&gpu_actual_decompressed_sizes)?;
    
    // Synchronize once after all GPU operations are queued
    stream.synchronize()?;
    info!("Results copied successfully");

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
        "  Copy to GPU (pinned): {:.4}ms ({:.2} GB/s)",
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