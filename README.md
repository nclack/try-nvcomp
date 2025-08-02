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
3. Transfer compressed data to GPU memory
4. Decompress all chunks in parallel on GPU using nvCOMP batched operations
5. Verify decompressed data matches original
6. Report GPU throughput in GB/s

The benchmarks measures just the pure GPU decompression performance for a single
batch decompression call.

It uses CUDA events timestamps for precise timing of the decompression step.

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
- Zstd decompression: **7.53 GB/s** (runs: 7.23, 7.36, 7.53 GB/s)
- LZ4 decompression: **7.41 GB/s** (runs: 7.41, 7.27, 7.09 GB/s)

_Note_: Running in WSL effects throughput by 20-30%.
        This could be differences in the cuda version (12.6 vs 12.9), drivers
        (575.57 vs 576.57), or due to virtualization. Max Zstd 6.37 GB/s, lz4
        7.66 GB/s.

