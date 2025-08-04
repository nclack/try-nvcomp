use bytesize::ByteSize;
use cudarc::driver::safe::{CudaContext, PinnedHostSlice};
use rand::Rng;
use std::error::Error;
use tracing::debug;

use crate::{compressors::Compressor, CHUNK_SIZE_U16, NUM_CHUNKS};

pub fn generate_sample_data() -> Vec<u16> {
    let mut rng = rand::thread_rng();
    (0..CHUNK_SIZE_U16)
        .map(|_| rng.gen_range(0..1024))
        .collect()
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BufferHeader {
    pub compressed_data_size: usize,
    pub chunk_count: usize,
    pub chunk_offsets_offset: usize, // Offset to chunk_offsets array
    pub chunk_sizes_offset: usize,   // Offset to chunk_sizes array
}

pub struct CompressedData {
    pub buffer: PinnedHostSlice<u8>, // Single unified buffer containing everything
    pub header: BufferHeader,
}

pub fn create_test_data<C: Compressor>(
    compressor: &C,
) -> Result<(CompressedData, Vec<Vec<u8>>), Box<dyn Error>> {
    // Generate 3 different patterns
    let patterns = vec![
        generate_sample_data(),
        (0..CHUNK_SIZE_U16).map(|i| (i % 65536) as u16).collect(),
        vec![42u16; CHUNK_SIZE_U16],
    ];

    let mut original_chunks = Vec::new();
    let mut chunk_offsets = Vec::new();
    let mut chunk_sizes = Vec::new();
    let mut total_size = 0;

    // First pass: calculate total size needed
    for i in 0..NUM_CHUNKS {
        let pattern = &patterns[i % patterns.len()];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                pattern.as_ptr() as *const u8,
                pattern.len() * std::mem::size_of::<u16>(),
            )
        };
        original_chunks.push(bytes.to_vec());
        let compressed = compressor.compress_data(bytes)?;
        total_size += compressed.len();
    }

    // Calculate unified buffer layout with proper alignment:
    // [compressed_data] [padding] [chunk_offsets: usize * NUM_CHUNKS] [chunk_sizes: usize * NUM_CHUNKS] [header: BufferHeader]
    let chunk_offsets_size = NUM_CHUNKS * std::mem::size_of::<usize>();
    let chunk_sizes_size = NUM_CHUNKS * std::mem::size_of::<usize>();
    let header_size = std::mem::size_of::<BufferHeader>();

    // Align to 8-byte boundaries for LZ4 compatibility
    let align_to_8 = |size: usize| (size + 7) & !7;

    let aligned_compressed_size = align_to_8(total_size);
    let chunk_offsets_offset = aligned_compressed_size;
    let chunk_sizes_offset = align_to_8(chunk_offsets_offset + chunk_offsets_size);
    let header_offset = align_to_8(chunk_sizes_offset + chunk_sizes_size);
    let total_buffer_size = header_offset + header_size;

    // Calculate offsets within the unified buffer
    let _compressed_data_offset = 0;

    // Allocate single unified pinned buffer
    let device = CudaContext::new(0)?;
    let mut pinned_buffer: PinnedHostSlice<u8> = unsafe { device.alloc_pinned(total_buffer_size)? };

    // Second pass: compress directly into pinned memory
    let mut offset = 0;
    original_chunks.clear(); // Clear and rebuild

    for i in 0..NUM_CHUNKS {
        let pattern = &patterns[i % patterns.len()];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                pattern.as_ptr() as *const u8,
                pattern.len() * std::mem::size_of::<u16>(),
            )
        };
        original_chunks.push(bytes.to_vec());
        let compressed = compressor.compress_data(bytes)?;

        // Copy compressed data to pinned memory
        let slice = pinned_buffer.as_mut_slice()?;
        slice[offset..offset + compressed.len()].copy_from_slice(&compressed);

        chunk_offsets.push(offset);
        chunk_sizes.push(compressed.len());
        offset += compressed.len();

        debug!(
            "Chunk {}: compressed {} -> {} (ratio: {:.2})",
            i,
            ByteSize(bytes.len().try_into().unwrap()),
            ByteSize(compressed.len().try_into().unwrap()),
            bytes.len() as f32 / compressed.len() as f32
        );
    }

    // Pack metadata arrays into the unified buffer with proper alignment
    let slice = pinned_buffer.as_mut_slice()?;

    // Zero out padding areas for clean alignment
    if chunk_offsets_offset > total_size {
        slice[total_size..chunk_offsets_offset].fill(0);
    }

    let chunk_offsets_bytes = unsafe {
        std::slice::from_raw_parts(chunk_offsets.as_ptr() as *const u8, chunk_offsets_size)
    };
    slice[chunk_offsets_offset..chunk_offsets_offset + chunk_offsets_size]
        .copy_from_slice(chunk_offsets_bytes);

    // Zero padding between chunk_offsets and chunk_sizes if needed
    let chunk_offsets_end = chunk_offsets_offset + chunk_offsets_size;
    if chunk_sizes_offset > chunk_offsets_end {
        slice[chunk_offsets_end..chunk_sizes_offset].fill(0);
    }

    let chunk_sizes_bytes =
        unsafe { std::slice::from_raw_parts(chunk_sizes.as_ptr() as *const u8, chunk_sizes_size) };
    slice[chunk_sizes_offset..chunk_sizes_offset + chunk_sizes_size]
        .copy_from_slice(chunk_sizes_bytes);

    // Zero padding between chunk_sizes and header if needed
    let chunk_sizes_end = chunk_sizes_offset + chunk_sizes_size;
    if header_offset > chunk_sizes_end {
        slice[chunk_sizes_end..header_offset].fill(0);
    }

    // Pack header into the unified buffer
    let header = BufferHeader {
        compressed_data_size: aligned_compressed_size, // Use aligned size for pointer calculations
        chunk_count: NUM_CHUNKS,
        chunk_offsets_offset,
        chunk_sizes_offset,
    };

    let header_bytes = unsafe {
        std::slice::from_raw_parts(&header as *const BufferHeader as *const u8, header_size)
    };
    slice[header_offset..header_offset + header_size].copy_from_slice(header_bytes);

    let compressed_data = CompressedData {
        buffer: pinned_buffer,
        header,
    };

    Ok((compressed_data, original_chunks))
}

pub fn create_uncompressed_test_data() -> Vec<Vec<u8>> {
    // Generate 3 different patterns
    let patterns = vec![
        generate_sample_data(),
        (0..CHUNK_SIZE_U16).map(|i| (i % 65536) as u16).collect(),
        vec![42u16; CHUNK_SIZE_U16],
    ];

    let mut chunks = Vec::new();
    for i in 0..NUM_CHUNKS {
        let pattern = &patterns[i % patterns.len()];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                pattern.as_ptr() as *const u8,
                pattern.len() * std::mem::size_of::<u16>(),
            )
        };
        chunks.push(bytes.to_vec());
    }
    chunks
}
