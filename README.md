# try-nvcomp

GPU compression and decompression benchmarks using NVIDIA nvCOMP library for Zstd and LZ4 compression algorithms.

## Prerequisites

- **CUDA Runtime Library**: CUDA 12.6+ 
- **nvCOMP Library**: NVIDIA nvCOMP v4.2+ 
- **Windows or Linux**: For non-standard installation paths, modify `build.rs`.

## Build

```bash
cargo build --release
```

## Run

### Unified Benchmark Tool

Run compression or decompression benchmarks:

```bash
# Compression benchmarks
cargo run --release --bin benchmark compress zstd
cargo run --release --bin benchmark compress lz4

# Decompression benchmarks  
cargo run --release --bin benchmark decompress zstd
cargo run --release --bin benchmark decompress lz4

# Multiple runs for statistical analysis
cargo run --release --bin benchmark compress zstd -r 5
```

**Usage:** `benchmark <compress|decompress> [zstd|lz4] [-r runs] [-v]`

## What it does

### Compression Benchmark:
1. Generates 1000 chunks of 1MB test data (u16 arrays) with different patterns
2. Transfers uncompressed data to GPU memory
3. Compresses all chunks in parallel on GPU using nvCOMP batched operations
4. Transfers compressed results back to host
5. Validates results by decompressing on CPU and comparing with the original
6. Reports compression throughput and compression ratio

### Decompression Benchmark:
1. Generates and compresses test data on CPU using the respective algorithm
2. Packs compressed data, metadata, and headers into a single pinned memory buffer
3. Transfers the entire unified buffer to GPU memory in one operation
4. Decompresses all chunks in parallel on GPU using nvCOMP batched operations
5. Verifies decompressed data matches original
6. Reports GPU copy and decompression throughput

Both benchmarks use CUDA events for timing measurements and use minimum-time 
sampling across multiple iterations to measure the peak performance.

**Test Data Patterns:**
- Uniformly distributed random over 10 bits
- Constant value (42)
- Linear ramp over 16 bits

## Benchmark Results

**Environment:** Windows 11  
**Hardware Configuration:**
- **GPU**: NVIDIA GeForce RTX 5080 (16GB VRAM)
- **CPU**: AMD Ryzen 5 7600X 6-Core Processor (12 threads)
- **RAM**: 64GB
- **NVIDIA Driver**: 576.57
- **CUDA**: 12.9
- **nvCOMP**: v4.2

### GPU Compression Performance (3 runs each):

**Copy to GPU:**
- Zstd: **37.22 GB/s** (runs: 30.47, 37.22, 36.82 GB/s)
- LZ4: **37.63 GB/s** (runs: 37.55, 37.63, 36.59 GB/s)

**GPU Compression:**
- Zstd: **5.70 GB/s** (runs: 5.60, 5.70, 5.54 GB/s)
- LZ4: **18.30 GB/s** (runs: 18.02, 18.29, 18.30 GB/s)

**Copy from GPU:**
- Zstd: **57.40 GB/s** (runs: 57.40, 57.35, 57.38 GB/s)
- LZ4: **57.34 GB/s** (runs: 57.34, 57.12, 57.14 GB/s)

**Output Compression Ratios:**
- Zstd: **3.03x** (1048.6 MB → 345.8 MB)
- LZ4: **1.49x** (1048.6 MB → 702.0 MB)

### GPU Decompression Performance (3 runs each):

**Input Compression Ratios:**
- Zstd: **3.45x** (304.1 MB → 1048.6 MB decompressed)
- LZ4: **1.49x** (703.5 MB → 1048.6 MB decompressed)

**Copy to GPU:**
- Zstd: **37.09 GB/s** (runs: 37.09, 36.72, 37.09 GB/s)
- LZ4: **37.28 GB/s** (runs: 36.97, 35.36, 37.28 GB/s)

**GPU Decompression:**
- Zstd: **51.24 GB/s** (runs: 51.24, 50.39, 50.57 GB/s)
- LZ4: **93.43 GB/s** (runs: 93.13, 93.43, 93.41 GB/s)

**Copy from GPU:**
- Zstd: **57.39 GB/s** (runs: 56.79, 57.39, 57.38 GB/s)
- LZ4: **57.36 GB/s** (runs: 53.76, 57.13, 57.36 GB/s)

**Notes:**
- Compression throughput measures the uncompressed data processing rate on the GPU
- Using pinned, single transfers for DtoH or HtoD is important for copy bandwidth.
- All measurements use CUDA events for timing
- Results represent peak performance (minimum time across multiple runs)
- Running in WSL reduces copy bandwidth by ~50% but maintains the same decompression performance

