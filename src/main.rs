use bytesize::ByteSize;
use cudarc::driver::safe::{CudaContext, CudaSlice};
use cudarc::driver::sys::{CUdeviceptr, CUevent_flags};
use cudarc::driver::DevicePtr;
use rand::Rng;
use std::time::Instant;
use tracing::{debug, info, span, Level};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

const CHUNK_SIZE_U16: usize = 1024 * 1024 / 2; // 1MB = 512K u16s
const NUM_CHUNKS: usize = 1000;
const NVCOMP_SUCCESS: i32 = 0;

fn generate_sample_data() -> Vec<u16> {
    let mut rng = rand::thread_rng();
    (0..CHUNK_SIZE_U16)
        .map(|_| rng.gen_range(0..1000))
        .collect()
}

fn create_test_data() -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
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

        let compressed = zstd::encode_all(bytes, 3).unwrap();
        info!(
            "Chunk {}: compressed {} -> {} (ratio: {:.2})",
            i,
            ByteSize(bytes.len().try_into().unwrap()),
            ByteSize(compressed.len().try_into().unwrap()),
            bytes.len() as f32 / compressed.len() as f32
        );
        compressed_chunks.push(compressed);
    }

    (compressed_chunks, original_chunks)
}

fn decompress_on_gpu(
    compressed_chunks: &[Vec<u8>],
    original_chunks: &[Vec<u8>],
) -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaContext::new(0)?;
    let stream = device.default_stream();

    // Create CUDA events for precise timing
    let start_event = device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let decomp_start_event = device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let decomp_end_event = device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let end_event = device.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

    info!(
        "Starting GPU decompression of {} chunks",
        compressed_chunks.len()
    );
    let start = Instant::now();

    // Record start event
    start_event.record(&stream)?;

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
    let gpu_statuses: CudaSlice<i32> = stream.alloc_zeros(NUM_CHUNKS)?;

    // Copy pointer arrays to GPU
    let gpu_compressed_ptr_array: CudaSlice<CUdeviceptr> =
        stream.memcpy_stod(&gpu_compressed_ptrs)?;
    let gpu_decompressed_ptr_array: CudaSlice<CUdeviceptr> =
        stream.memcpy_stod(&gpu_decompressed_ptrs)?;

    // Record decompression start event (after data copy)
    decomp_start_event.record(&stream)?;

    // Synchronize to ensure all copies are complete before decompression
    stream.synchronize()?;
    let copy_time = start.elapsed();
    info!(
        "Data copy to GPU: {:.4}ms",
        copy_time.as_secs_f64() * 1000.0
    );

    // This is where you'd call nvCOMP functions:
    // Calculate total sizes for reporting
    let total_compressed_size: usize = compressed_sizes.iter().sum();
    let total_decompressed_size = NUM_CHUNKS * chunk_size_bytes;
    let overall_compression_ratio = total_decompressed_size as f32 / total_compressed_size as f32;

    info!("nvCOMP batched decompression would be called here with:");
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

    // Real nvCOMP call structure:
    unsafe {
        // Get temp memory requirements
        let mut temp_bytes: usize = 0;
        let status = nvcompBatchedZstdDecompressGetTempSize(
            NUM_CHUNKS,
            chunk_size_bytes,
            &mut temp_bytes as *mut usize,
        );

        if status != NVCOMP_SUCCESS {
            return Err("Failed to get temp size".into());
        }

        info!(
            "nvCOMP temp memory required: {}",
            ByteSize(temp_bytes.try_into().unwrap())
        );

        // Allocate temp memory if needed
        let gpu_temp: Option<CudaSlice<u8>> = if temp_bytes > 0 {
            Some(stream.alloc_zeros(temp_bytes)?)
        } else {
            None
        };

        // Perform batched decompression
        let status = nvcompBatchedZstdDecompressAsync(
            gpu_compressed_ptr_array.device_ptr(&stream).0 as *const *const std::ffi::c_void,
            gpu_compressed_sizes.device_ptr(&stream).0 as *const usize,
            gpu_decompressed_buffer_sizes.device_ptr(&stream).0 as *const usize,
            gpu_actual_decompressed_sizes.device_ptr(&stream).0 as *mut usize,
            NUM_CHUNKS,
            gpu_temp.as_ref().map_or(std::ptr::null_mut(), |t| {
                t.device_ptr(&stream).0 as *mut std::ffi::c_void
            }),
            temp_bytes,
            gpu_decompressed_ptr_array.device_ptr(&stream).0 as *const *mut std::ffi::c_void,
            gpu_statuses.device_ptr(&stream).0 as *mut i32,
            stream.cu_stream() as cudaStream_t,
        );

        if status != NVCOMP_SUCCESS {
            return Err(format!("nvCOMP decompression failed with status: {}", status).into());
        }

        info!("nvCOMP decompression completed successfully");
    }

    // Record decompression end event
    decomp_end_event.record(&stream)?;

    // Record total end event
    end_event.record(&stream)?;

    // Wait for events to complete before measuring
    decomp_end_event.synchronize()?;
    end_event.synchronize()?;

    // Ensure all GPU work is complete
    stream.synchronize()?;

    // Calculate GPU timings using CUDA events
    info!("Calculating GPU timings using CUDA events...");

    // Get precise timing from CUDA events
    let total_gpu_time = start_event.elapsed_ms(&end_event)?;
    let decomp_gpu_time = decomp_start_event.elapsed_ms(&decomp_end_event)?;
    let copy_gpu_time = total_gpu_time - decomp_gpu_time;

    info!("Total GPU time: {:.4}ms", total_gpu_time);
    info!("Decomp GPU time: {:.4}ms", decomp_gpu_time);
    info!("Copy GPU time: {:.4}ms", copy_gpu_time);

    // For verification, copy some results back
    info!("Copying status results from GPU...");
    let host_statuses: Vec<i32> = stream.memcpy_dtov(&gpu_statuses)?;
    info!("Copying size results from GPU...");
    let host_actual_sizes: Vec<usize> = stream.memcpy_dtov(&gpu_actual_decompressed_sizes)?;
    info!("Results copied successfully");

    let total_decompressed_size = NUM_CHUNKS * chunk_size_bytes;

    info!("GPU decompression completed:");
    info!("  Copy to GPU: {:.4}ms", copy_gpu_time);
    info!("  Decompression: {:.4}ms", decomp_gpu_time);
    info!("  Total GPU time: {:.4}ms", total_gpu_time);
    info!(
        "  GPU throughput: {:.2} GB/s",
        (total_decompressed_size as f64 / 1e9) / (total_gpu_time as f64 / 1000.0)
    );
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let _span = span!(Level::INFO, "nvcomp_example").entered();

    info!("Starting nvCOMP parallel decompression example");

    let (compressed_data, original_data) = create_test_data();
    decompress_on_gpu(&compressed_data, &original_data)?;

    info!("Example completed successfully");
    Ok(())
}
