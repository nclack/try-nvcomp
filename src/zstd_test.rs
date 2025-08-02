use try_nvcomp::{run_benchmark, ZstdCompressor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut compressor = ZstdCompressor::new();
    compressor.init()?;
    run_benchmark(compressor)
}