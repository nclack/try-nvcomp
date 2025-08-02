# try-nvcomp

GPU decompression benchmarks using NVIDIA nvCOMP library for Zstd and LZ4 compression algorithms.

## Prerequisites

- **CUDA Runtime Library**: This project requires CUDA 12.9+ to be installed
- **nvCOMP Library**: NVIDIA nvCOMP v4.2+ must be installed
- **Windows**: Currently only builds on Windows. For Linux or non-standard installation paths, modify `build.rs`

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
3. Transfer compressed data to GPU memory
4. Decompress all chunks in parallel on GPU using nvCOMP batched operations
5. Verify decompressed data matches original
6. Report GPU throughput in GB/s

The benchmarks measure pure GPU decompression performance using CUDA events for
precise timing of the decompression step.

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

Maximum GPU Throughput (3 runs each):
- Zstd decompression: **8.75 GB/s** (runs: 8.17, 8.75 GB/s, 8.63 GB/s)
- LZ4 decompression: **9.62 GB/s** (runs: 9.12, 7.76, 9.62 GB/s)

_Note_: Running in WSL effects throughput by 20-30%.
        This could be differences in the cuda version (12.6 vs 12.9), drivers (575.57 vs 576.57), or due to virtualization.
        Max Zstd 6.37 GB/s, lz4 7.66 GB/s.

