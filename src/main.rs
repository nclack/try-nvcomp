use clap::{Parser, ValueEnum};
use std::error::Error;
use try_nvcomp::{run_decompression_benchmark, run_compression_benchmark, Lz4Compressor, ZstdCompressor};

#[derive(Debug, Clone, ValueEnum)]
enum Algorithm {
    /// ZSTD compression algorithm
    Zstd,
    /// LZ4 compression algorithm  
    Lz4,
}

#[derive(Debug, Clone, ValueEnum)]
enum BenchmarkType {
    /// Run compression benchmark (CPU->GPU compression + validation)
    Compress,
    /// Run decompression benchmark (GPU decompression + validation)
    Decompress,
}

#[derive(Parser)]
#[command(
    name = "nvcomp_benchmark",
    about = "nvCOMP GPU compression/decompression benchmark tool",
    long_about = "A benchmark tool for testing GPU-accelerated compression and decompression using NVIDIA nvCOMP library. Supports ZSTD and LZ4 algorithms."
)]
struct Args {
    /// Type of benchmark to run (compress, decompress )
    #[arg(value_enum)]
    benchmark: BenchmarkType,

    /// Compression algorithm to benchmark (zstd, lz4)
    #[arg(value_enum, default_value_t = Algorithm::Zstd)]
    algorithm: Algorithm,

    /// Number of repetitions for more stable measurements
    #[arg(short, long, default_value_t = 1)]
    runs: u32,

    /// Enable verbose debug output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // Initialize tracing subscriber with appropriate level
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(if args.verbose {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;

    println!("nvCOMP Benchmark Tool");
    println!("Algorithm: {:?}", args.algorithm);
    println!("Benchmark: {:?}", args.benchmark);
    println!("Runs: {}", args.runs);
    println!("Verbose: {}", args.verbose);
    println!();

    // Initialize compressor once outside the run loop
    match args.algorithm {
        Algorithm::Zstd => {
            let mut compressor = ZstdCompressor::new();
            compressor.init()?;
            
            // Run the specified number of times
            for run in 1..=args.runs {
                if args.runs > 1 {
                    println!("=== Run {}/{} ===", run, args.runs);
                }
                run_benchmarks(&args.benchmark, compressor.clone())?;
                if args.runs > 1 && run < args.runs {
                    println!();
                }
            }
        }
        Algorithm::Lz4 => {
            let mut compressor = Lz4Compressor::new();
            compressor.init()?;
            
            // Run the specified number of times
            for run in 1..=args.runs {
                if args.runs > 1 {
                    println!("=== Run {}/{} ===", run, args.runs);
                }
                run_benchmarks(&args.benchmark, compressor.clone())?;
                if args.runs > 1 && run < args.runs {
                    println!();
                }
            }
        }
    }

    Ok(())
}

fn run_benchmarks<C>(benchmark_type: &BenchmarkType, compressor: C) -> Result<(), Box<dyn Error>>
where
    C: try_nvcomp::Compressor + Clone,
{
    match benchmark_type {
        BenchmarkType::Compress => {
            println!("Running compression benchmark...");
            run_compression_benchmark(compressor)?;
        }
        BenchmarkType::Decompress => {
            println!("Running decompression benchmark...");
            run_decompression_benchmark(compressor)?;
        }
    }
    Ok(())
}
