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
5. Validates results by decompressing on CPU and comparing with original
6. Reports compression throughput and compression ratio

### Decompression Benchmark:
1. Generates and compresses test data on CPU using the respective algorithm
2. Packs compressed data, metadata, and headers into a single pinned memory buffer
3. Transfers the entire unified buffer to GPU memory in one operation
4. Decompresses all chunks in parallel on GPU using nvCOMP batched operations
5. Verifies decompressed data matches original
6. Reports GPU copy and decompression throughput

Both benchmarks use CUDA events for precise GPU-only timing measurements and
implement minimum-time sampling across multiple iterations to measure peak
performance capabilities.

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

**GPU Compression:**
- Zstd: **5.71 GB/s** (runs: 5.71, 5.71, 5.71 GB/s)
- LZ4: **18.72 GB/s** (runs: 18.71, 18.72, 18.69 GB/s)

### GPU Decompression Performance (3 runs each):

**Copy to GPU (Pinned Memory):**
- Zstd: **11.62 GB/s** (runs: 11.62, 11.41, 10.79 GB/s)
- LZ4: **19.29 GB/s** (runs: 17.71, 19.29, 19.08 GB/s)

**GPU Decompression:**
- Zstd: **49.67 GB/s** (runs: 46.24, 49.67, 48.66 GB/s)
- LZ4: **94.26 GB/s** (runs: 94.26, 94.08, 94.04 GB/s)

**Notes:**
- Compression throughput measures uncompressed data processing rate on GPU
- Decompression results based on unified buffer approach (single memcpy operation)
- All measurements use CUDA events for precise GPU-only timing
- Results represent peak performance (minimum time across multiple runs)
- Running in WSL reduces copy bandwidth by ~50% but maintains same decompression performance

