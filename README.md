# try-nvcomp

GPU decompression benchmarks using NVIDIA nvCOMP library for Zstd and LZ4 compression algorithms.

## Prerequisites

- **CUDA Runtime Library**: CUDA 12.6+ 
- **nvCOMP Library**: NVIDIA nvCOMP v4.2+ 
- **Windows or Linux**: For non-standard installation paths, modify `build.rs`.

## Build

```bash
cargo build --release
```

## Run

The project includes two benchmarking binaries:

### Zstd Decompression Test
```bash
cargo run --release --bin zstd_test
```

### LZ4 Decompression Test  
```bash
cargo run --release --bin lz4_test
```

## What it does

Both tests:
1. Generate 1000 chunks of 1MB test data (u16 arrays) with different patterns
2. Compress the data on CPU using the respective algorithm (Zstd or LZ4)
3. Pack compressed data, metadata, and headers into a single pinned memory buffer, aligning to 8-bytes
4. Transfer the entire unified buffer to GPU memory in one operation
5. Decompress all chunks in parallel on GPU using nvCOMP batched operations
6. Verify decompressed data matches original
7. Report GPU copy and decompression throughput in GB/s

The benchmarks measure both memory transfer and GPU decompression performance
using CUDA events for precise timing. The unified buffer approach minimizes
GPU memory transfers by consolidating all data into a single pinned memory
allocation transferred via one memcpy operation.

Patterns:
- Uniformly distributed random over 10 bits.
- Constant value
- Linear ramp over 16 bits

## Benchmark Results

Environment: Windows 11
Hardware configuration:
- **GPU**: NVIDIA GeForce RTX 5080 (16GB VRAM)
- **CPU**: AMD Ryzen 5 7600X 6-Core Processor (12 threads)
- **RAM**: 64GB
- **NVIDIA Driver**: 576.57
- **CUDA**: 12.9
- **nvCOMP**: v4.2

### Unified Buffer Performance (3 runs each):

**Copy to GPU (Pinned Memory):**
- Zstd: **11.78 GB/s** (runs: 10.50, 11.71, 11.78 GB/s)
- LZ4: **18.54 GB/s** (runs: 18.39, 18.54, 18.00 GB/s)

**GPU Decompression:**
- Zstd: **49.15 GB/s** (runs: 47.86, 45.52, 49.15 GB/s)
- LZ4: **93.65 GB/s** (runs: 93.47, 93.39, 93.65 GB/s)

_Note_: Running in WSL effects throughput by 20-30%.
        This could be differences in the cuda version (12.6 vs 12.9), drivers
        (575.57 vs 576.57), or due to virtualization.

