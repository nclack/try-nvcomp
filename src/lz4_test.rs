use try_nvcomp::{run_benchmark, Lz4Compressor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut compressor = Lz4Compressor::new();
    compressor.init()?;
    run_benchmark(compressor)
}