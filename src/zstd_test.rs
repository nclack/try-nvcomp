use try_nvcomp::{run_benchmark, ZstdCompressor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compressor = ZstdCompressor::new();
    run_benchmark(compressor)
}