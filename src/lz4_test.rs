use try_nvcomp::{run_benchmark, Lz4Compressor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compressor = Lz4Compressor::new();
    run_benchmark(compressor)
}