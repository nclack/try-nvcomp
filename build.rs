use std::env;
use std::path::PathBuf;

fn main() {
    if cfg!(target_os = "windows") {
        configure_windows();
    } else if cfg!(target_os = "linux") {
        configure_linux();
    } else {
        panic!("Unsupported platform");
    }
    
    generate_bindings();
}

fn configure_windows() {
    // Link with CUDA 12.9 and nvCOMP v4.2 libraries
    println!("cargo:rustc-link-search=native=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\lib\\x64");
    println!("cargo:rustc-link-search=native=C:\\Program Files\\NVIDIA nvCOMP\\v4.2\\lib\\12");
    
    // Link with static libraries
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=static=nvcomp_static");
    println!("cargo:rustc-link-lib=static=nvcomp_device_static");
    
    // Required system libraries for static linking
    println!("cargo:rustc-link-lib=kernel32");
    println!("cargo:rustc-link-lib=user32");
    println!("cargo:rustc-link-lib=advapi32");
}

fn configure_linux() {
    // Try to use pkg-config for CUDA libraries
    match pkg_config::Config::new().probe("cudart-12.6") {
        Ok(_) => {
            println!("cargo:rustc-link-lib=cudart");
        }
        Err(_) => {
            // Fallback to manual configuration
            println!("cargo:rustc-link-search=native=/usr/local/cuda-12.6/targets/x86_64-linux/lib");
            println!("cargo:rustc-link-lib=cudart");
        }
    }
    
    // Link with nvCOMP libraries - these are installed in specific paths
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/nvcomp/12");
    println!("cargo:rustc-link-lib=static=nvcomp_static");
    println!("cargo:rustc-link-lib=static=nvcomp_device_static");
    
    // Required system libraries on Linux
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=rt");
    println!("cargo:rustc-link-lib=stdc++");
}

fn generate_bindings() {
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .allowlist_type("nvcompStatus_t")
        .allowlist_type("cudaStream_t")
        .allowlist_function("nvcompBatchedZstdDecompressAsync")
        .allowlist_function("nvcompBatchedZstdDecompressGetTempSize")
        .allowlist_function("nvcompBatchedLZ4DecompressAsync")
        .allowlist_function("nvcompBatchedLZ4DecompressGetTempSize")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Silence naming convention warnings
        .derive_debug(false)
        .layout_tests(false)
        .generate_comments(false);

    // Add platform-specific include paths
    if cfg!(target_os = "windows") {
        builder = builder
            .clang_arg("-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\include")
            .clang_arg("-IC:\\Program Files\\NVIDIA nvCOMP\\v4.2\\include");
    } else if cfg!(target_os = "linux") {
        builder = builder
            .clang_arg("-I/usr/local/cuda-12.6/targets/x86_64-linux/include")
            .clang_arg("-I/usr/include/nvcomp_12");
    }

    let bindings = builder
        .generate()
        .unwrap_or_else(|_| {
            // Fallback if bindgen fails
            println!("cargo:warning=Failed to generate nvCOMP bindings, using stubs");
            bindgen::Builder::default()
                .header_contents("stub.h", r#"
                    typedef int nvcompStatus_t;
                    typedef void* cudaStream_t;
                    #define nvcompSuccess 0
                "#)
                .generate()
                .expect("Failed to generate stub bindings")
        });

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
