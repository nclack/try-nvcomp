use std::env;
use std::path::PathBuf;

fn main() {
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
    
    // Generate minimal bindings just for the types we need
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\include")
        .clang_arg("-IC:\\Program Files\\NVIDIA nvCOMP\\v4.2\\include")
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
        .generate_comments(false)
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
